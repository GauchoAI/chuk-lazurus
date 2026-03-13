#!/usr/bin/env python3
"""
Residual-stream navigation map extractor — 3D sphere edition.

Captures the residual stream at every layer, for every generated token step.
Projects fact unembedding directions and residuals onto a visible 3D sphere
via SVD on the fact subspace.

Geometry
--------
  Basis   : top-3 right singular vectors of the fact unembedding matrix.
             These are the 3 directions that best span the "fact space".
  Facts   : project → normalise → points ON the unit sphere surface.
  Residuals: project → keep raw magnitude.
             Early layers are nearly orthogonal to all facts → near the centre.
             Late layers align with the target fact → near the sphere surface.
             The trajectory is literally the model navigating from centre to surface.

Output
------
  nav_map_<target>.json  — version 2.0, self-contained, read by nav_map.html

Usage
-----
    # Default: capital of France, Gemma 3-4B, 20 generation steps
    uv run python examples/inference/nav_map_extract.py

    # Smaller model for fast iteration
    uv run python examples/inference/nav_map_extract.py \\
        --model mlx-community/gemma-3-1b-it-bf16 \\
        --steps 15

    # Different target
    uv run python examples/inference/nav_map_extract.py \\
        --prompt "The capital of Japan is" \\
        --target Tokyo --output nav_map_tokyo.json

    # Custom facts
    uv run python examples/inference/nav_map_extract.py \\
        --facts Paris Tokyo Berlin Ottawa Cairo Nairobi Sydney Canberra \\
        --target Paris --steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# ── chuk-lazarus on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_lazarus.models_v2.families.gemma import GemmaConfig
from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[92m"
AMBER = "\033[93m"
DIM   = "\033[2m"
RESET = "\033[0m"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-bf16"

DEFAULT_FACTS = [
    "Paris", "Tokyo", "Canberra", "Sydney",
    "Berlin", "Ottawa", "Nairobi", "Cairo",
]

DEFAULT_COLORS = [
    "#E24B4A", "#1D9E75", "#378ADD", "#534AB7",
    "#D85A30", "#639922", "#D4537E", "#BA7517",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    print(f"  Downloading {model_id} ...", file=sys.stderr)
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    ))


def _apply_weights(model, model_path: Path) -> None:
    from mlx.utils import tree_unflatten
    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))
    sanitized = model.sanitize(raw)
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in sanitized.items()
    }
    model.update(tree_unflatten(list(sanitized.items())))
    mx.eval(model.parameters())


def load_rs_model(model_id: str):
    """Returns (model, tokenizer, config)."""
    from transformers import AutoTokenizer

    print(f"{BOLD}Loading {model_id}{RESET}", file=sys.stderr)
    model_path = _download(model_id)

    with open(model_path / "config.json") as f:
        config_data = json.load(f)
    config = GemmaConfig.from_hf_config(config_data)
    print(
        f"  {config.num_hidden_layers} layers  "
        f"hidden={config.hidden_size}  vocab={config.vocab_size}",
        file=sys.stderr,
    )

    model = GemmaResidualStreamForCausalLM(config)
    _apply_weights(model, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, config


# ── Step 1: Fact unembedding directions ───────────────────────────────────────

def get_fact_unembeddings(model, tokenizer, fact_tokens: list[str]) -> dict:
    """
    lm_head.weight[token_id] = the direction in residual space that predicts
    that token. These are the fixed landmarks — model properties, not inputs.

    Returns {token_str: {token_id, direction (D,), unit (D,), norm}}.
    Skips multi-token entries with a warning.
    """
    weight_mx = model.lm_head.weight  # (vocab_size, hidden_size)
    unembed: dict = {}

    for token_str in fact_tokens:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) != 1:
            print(f"  {AMBER}SKIP{RESET} '{token_str}' → {len(ids)} tokens", file=sys.stderr)
            continue
        token_id = ids[0]
        direction = np.array(weight_mx[token_id].astype(mx.float32), dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        unembed[token_str] = {
            "token_id": token_id,
            "direction": direction,
            "unit": direction / (norm + 1e-12),
            "norm": norm,
        }
        print(f"  {GREEN}✓{RESET} {token_str!r:<12s}  id={token_id:>7d}  ‖e‖={norm:.1f}", file=sys.stderr)

    return unembed


# ── Step 2: Per-layer metrics (shared helper) ─────────────────────────────────

def _cosine_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _layer_metrics(
    r_mx: mx.array,
    unembed: dict,
    model,
    tokenizer,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Compute metrics for the last-token position of one layer's residual.

    Returns (metrics_dict, raw_residual_np, normed_residual_np).

    Key geometry note: lm_head.weight rows live in *normed* residual space.
    All cosine/sharpness computations use the logit-lens normed residual so
    that angles are computed in the same space as the fact directions.
    """
    facts = list(unembed.keys())
    r = np.array(r_mx[0, -1, :].astype(mx.float32), dtype=np.float32)
    r_norm = float(np.linalg.norm(r))

    # Logit lens: apply final RMSNorm → lm_head (same space as lm_head.weight)
    r_t = mx.array(r[None, None, :])
    normed_mx = model.model.norm(r_t)
    normed = np.array(normed_mx[0, 0, :].astype(mx.float32), dtype=np.float32)
    logits = np.array(model.lm_head(normed_mx)[0, 0, :].astype(mx.float32), dtype=np.float32)
    top_idx = int(np.argmax(logits))

    # Angles computed in normed space (same space as fact directions)
    angles: dict[str, float] = {}
    for token_str, udata in unembed.items():
        angles[token_str] = _cosine_angle_deg(normed, udata["direction"])

    # Sharpness: concentration of logit-lens logits over the fact tokens
    fact_ids = [udata["token_id"] for udata in unembed.values()]
    fact_logits = logits[fact_ids].astype(np.float32)
    fact_logits -= fact_logits.max()
    probs = np.exp(fact_logits)
    probs /= probs.sum()
    H = float(-np.sum(probs * np.log(probs + 1e-12)))
    H_max = float(np.log(len(facts)))
    sharpness = float(1.0 - H / H_max) if H_max > 0 else 1.0

    return {
        "angles_to_facts": angles,
        "sharpness": sharpness,
        "residual_norm": r_norm,
        "top_token": tokenizer.decode([top_idx]),
        "top_logit": float(logits[top_idx]),
    }, r, normed


# ── Step 3: Multi-frame extraction (token-by-token generation) ────────────────

def extract_frames(
    model,
    tokenizer,
    prompt: str,
    unembed: dict,
    steps: int = 20,
) -> list[dict]:
    """
    Greedily generate `steps` tokens from `prompt`.
    At each step, run a full forward pass and capture the 34-layer trajectory
    at the current last-token position.

    Each frame has:
      step, token, token_id, predicted_token, predicted_token_id,
      prompt_so_far, layers (list of per-layer dicts with metrics),
      _raw_residuals (list of (D,) np arrays — stripped before JSON output)
    """
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    frames: list[dict] = []

    print(
        f"  Initial prompt: {prompt!r}  ({len(token_ids)} tokens)\n"
        f"  Capturing {steps + 1} frames (prompt + {steps} generated tokens) ...",
        file=sys.stderr,
    )
    t_total = time.monotonic()

    for step in range(steps + 1):
        t0 = time.monotonic()
        input_ids = mx.array(token_ids)[None]
        out = model(input_ids, collect_layer_residuals=True)
        mx.eval(out.logits)
        for r in out.layer_residuals:
            mx.eval(r)
        elapsed = time.monotonic() - t0

        # Per-layer metrics
        layers_data: list[dict] = []
        raw_residuals: list[np.ndarray] = []
        normed_residuals: list[np.ndarray] = []
        for layer_idx, r_mx in enumerate(out.layer_residuals):
            metrics, r_raw, r_normed = _layer_metrics(r_mx, unembed, model, tokenizer)
            layers_data.append({"layer": layer_idx, **metrics})
            raw_residuals.append(r_raw)
            normed_residuals.append(r_normed)

        # Predicted next token (greedy)
        final_logits = np.array(out.logits[0, -1, :].astype(mx.float32), dtype=np.float32)
        pred_id = int(np.argmax(final_logits))
        pred_token = tokenizer.decode([pred_id])
        current_token = tokenizer.decode([token_ids[-1]])

        frames.append({
            "step": step,
            "token": current_token,
            "token_id": int(token_ids[-1]),
            "predicted_token": pred_token,
            "predicted_token_id": pred_id,
            "prompt_so_far": tokenizer.decode(token_ids),
            "layers": layers_data,
            "_raw_residuals": raw_residuals,
            "_normed_residuals": normed_residuals,
        })

        print(
            f"\r  Step {step + 1:>3}/{steps + 1}  "
            f"token={current_token!r:<10s}  "
            f"→ {pred_token!r:<10s}  "
            f"({elapsed:.2f}s)",
            end="",
            file=sys.stderr,
            flush=True,
        )

        if step < steps:
            token_ids = token_ids + [pred_id]

    print(
        f"\n  Total: {time.monotonic() - t_total:.1f}s for {steps + 1} frames",
        file=sys.stderr,
    )
    return frames


# ── Step 4: 3D sphere projection ──────────────────────────────────────────────

def compute_sphere_projection(
    unembed: dict,
    frames: list[dict],
) -> tuple[dict[str, list[float]], list]:
    """
    Project fact directions and residuals onto a visible 3D sphere.

    Basis: top-3 right singular vectors of the (n_facts × D) fact unit matrix.
           These span the 3D subspace that best captures the fact geometry.

    Fact positions:
        project onto basis → normalise to unit sphere surface.
        These are the fixed landmarks. Their positions reflect actual angular
        relationships between fact unembedding directions.

    Residual positions (per layer, per frame):
        project onto basis → keep raw magnitude (do NOT normalise).
        ‖xyz‖ ≈ 0  →  residual is orthogonal to all facts  (dark accumulation)
        ‖xyz‖ ≈ 1  →  residual lies in the fact subspace   (fact explosion)
        direction  →  which fact the residual points toward

    Adds "xyz": [x, y, z] to every layer dict in every frame (in-place).
    Returns (fact_positions_3d, basis_as_nested_list).
    """
    facts = list(unembed.keys())
    fact_units = np.stack([udata["unit"] for udata in unembed.values()]).astype(np.float32)

    # SVD on the fact unit matrix — no centering (we want sphere geometry)
    print("  SVD on fact subspace ...", file=sys.stderr)
    _, _, Vt = np.linalg.svd(fact_units, full_matrices=False)
    basis = Vt[:3].astype(np.float32)  # (3, D)

    # Fact positions: project + normalise to unit sphere
    fact_3d = fact_units @ basis.T   # (n_facts, 3)
    fact_positions_3d: dict[str, list[float]] = {}
    for i, t in enumerate(facts):
        p = fact_3d[i].astype(np.float64)
        n = float(np.linalg.norm(p))
        if n > 1e-8:
            fact_positions_3d[t] = [round(float(v / n), 6) for v in p]
        else:
            fact_positions_3d[t] = [0.0, 1.0, 0.0]

    # Residual positions: project normed residual onto fact subspace, then
    # normalise to the unit sphere. Direction = which fact the residual points toward.
    # Commitment is captured by sharpness (not position magnitude), so the dot is
    # always ON the sphere — dim/small when uncommitted, bright/large when sharp.
    for frame in frames:
        for i, layer_data in enumerate(frame["layers"]):
            r = frame["_normed_residuals"][i]
            r_unit = r / (float(np.linalg.norm(r)) + 1e-12)
            xyz = (basis @ r_unit).astype(np.float64)
            xyz_len = float(np.linalg.norm(xyz))
            if xyz_len > 1e-8:
                xyz_norm = xyz / xyz_len
            else:
                xyz_norm = np.array([0., 1., 0.])
            layer_data["xyz"] = [round(float(v), 6) for v in xyz_norm]

    return fact_positions_3d, basis.tolist()


# ── Step 5: Phase boundary detection ─────────────────────────────────────────

def detect_phase_boundaries(frames: list[dict]) -> list[int]:
    """
    Detect inflection points in the sharpness curve of the first frame.
    Falls back to even thirds.
    """
    trajectory = frames[0]["layers"]
    n = len(trajectory)
    sharpness = np.array([l["sharpness"] for l in trajectory], dtype=np.float32)
    d2 = np.diff(np.diff(sharpness))
    if len(d2) >= 4:
        candidates = np.argsort(np.abs(d2))[-2:]
        boundaries = sorted([int(x + 1) for x in candidates])
        if boundaries[0] > 2 and boundaries[1] < n - 2:
            return boundaries
    return [n // 3, 2 * n // 3]


# ── Full pipeline ─────────────────────────────────────────────────────────────

def extract_nav_map(
    model_id: str,
    prompt: str,
    target_fact: str,
    fact_tokens: list[str] | None = None,
    steps: int = 20,
    output_path: str | Path | None = None,
) -> dict:
    """Full pipeline: load → facts → frames → sphere → JSON."""
    fact_tokens = fact_tokens or DEFAULT_FACTS

    # 1. Load
    model, tokenizer, config = load_rs_model(model_id)

    # 2. Unembedding directions
    print(f"\n{BOLD}Fact unembedding directions{RESET}", file=sys.stderr)
    unembed = get_fact_unembeddings(model, tokenizer, fact_tokens)

    if target_fact not in unembed:
        raise ValueError(
            f"Target '{target_fact}' not available (may be multi-token). "
            f"Available: {list(unembed.keys())}"
        )

    # 3. Multi-frame extraction
    print(f"\n{BOLD}Extracting {steps + 1} frames{RESET}", file=sys.stderr)
    frames = extract_frames(model, tokenizer, prompt, unembed, steps=steps)

    # 4. Sphere projection (adds xyz to each layer in each frame in-place)
    print(f"\n{BOLD}3D sphere projection{RESET}", file=sys.stderr)
    fact_positions_3d, basis = compute_sphere_projection(unembed, frames)

    # 5. Phase boundaries
    phase_boundaries = detect_phase_boundaries(frames)

    # 6. Pairwise angles between facts
    fact_list = list(unembed.keys())
    pairwise_angles = {
        f"{t1}-{t2}": _cosine_angle_deg(unembed[t1]["direction"], unembed[t2]["direction"])
        for i, t1 in enumerate(fact_list)
        for j, t2 in enumerate(fact_list)
        if i < j
    }

    # 7. Facts metadata — include their 3D sphere positions
    facts_meta = [
        {
            "token": t,
            "token_id": unembed[t]["token_id"],
            "is_target": t == target_fact,
            "color": DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            "xyz": fact_positions_3d[t],
        }
        for i, t in enumerate(fact_list)
    ]

    # 8. Strip temporary residual arrays before serialisation
    for frame in frames:
        del frame["_raw_residuals"]
        del frame["_normed_residuals"]

    data = {
        "version": "2.0",
        "model": model_id,
        "hidden_dim": int(config.hidden_size),
        "num_layers": int(config.num_hidden_layers),
        "prompt": prompt,
        "target_fact": target_fact,
        "phase_boundaries": phase_boundaries,
        "facts": facts_meta,
        "pairwise_fact_angles": pairwise_angles,
        "frames": frames,
        # basis omitted from default output (large); add if you need client-side reprojection
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        size_kb = output_path.stat().st_size / 1024
        print(f"\n{GREEN}Saved → {output_path}  ({size_kb:.1f} KB){RESET}", file=sys.stderr)

    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract residual-stream sphere navigation map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local path")
    p.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    p.add_argument(
        "--target", default="Paris",
        help="Target fact token (must be a single token in the vocabulary)",
    )
    p.add_argument("--facts", nargs="+", default=None, help="Landmark fact tokens")
    p.add_argument(
        "--steps", type=int, default=20,
        help="Tokens to generate after the prompt (default: 20)",
    )
    p.add_argument("--output", default=None, help="Output JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output or f"nav_map_{args.target.lower()}.json"
    extract_nav_map(
        model_id=args.model,
        prompt=args.prompt,
        target_fact=args.target,
        fact_tokens=args.facts,
        steps=args.steps,
        output_path=output,
    )
    print(f"\nOpen:  open examples/inference/nav_map.html")
    print(f"Then drag-drop {output} onto the canvas.")


if __name__ == "__main__":
    main()
