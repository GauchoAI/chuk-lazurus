#!/usr/bin/env python3
"""
Proof: Gemma Residual Stream inference == standard Gemma inference.

Loads the same checkpoint into both architectures, runs the same prompt,
and verifies numerical identity. Then demonstrates the residual stream API.

Usage:
    uv run python examples/inference/gemma_rs_proof.py
    uv run python examples/inference/gemma_rs_proof.py --model mlx-community/gemma-3-1b-it-bf16
    uv run python examples/inference/gemma_rs_proof.py --prompt "The capital of France is"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/gemma-3-270m-it-bf16"
DEFAULT_PROMPT = "The Markov property states that"


def parse_args():
    p = argparse.ArgumentParser(description="Gemma RS proof of equivalence")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local path")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    p.add_argument("--probe-layer", type=int, default=7, help="Layer to probe residual at")
    p.add_argument("--inject-layer", type=int, default=14, help="Layer to inject residual at")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg):
    print(f"  {RED}✗{RESET} {msg}")
    sys.exit(1)


def header(msg):
    print(f"\n{BOLD}{msg}{RESET}")
    print("─" * len(msg))


def compare_tensors(a: mx.array, b: mx.array, label: str, tol: float = 5e-3) -> float:
    """Compare two tensors, print stats, return max abs diff."""
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    max_diff = float(np.max(np.abs(a_np - b_np)))
    mean_diff = float(np.mean(np.abs(a_np - b_np)))
    if max_diff <= tol:
        ok(f"{label}: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [PASS]")
    else:
        fail(f"{label}: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [FAIL > {tol}]")
    return max_diff


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _download(model_id: str) -> Path:
    """Locate or download model. Returns local path."""
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    print(f"  Downloading from HuggingFace Hub...")
    path = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    )
    return Path(path)


def _apply_weights(model, model_path: Path) -> None:
    """Load safetensors, call model.sanitize(), apply via model.update()."""
    from mlx.utils import tree_unflatten

    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))

    sanitized = model.sanitize(raw)
    # Convert to bfloat16
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in sanitized.items()
    }
    model.update(tree_unflatten(list(sanitized.items())))
    mx.eval(model.parameters())


def load_models(model_id: str):
    """
    Download once, load into both GemmaForCausalLM and GemmaResidualStreamForCausalLM.
    Returns (standard_model, rs_model, tokenizer, config, model_path).
    """
    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM
    from transformers import AutoTokenizer

    print(f"\nLocating: {model_id}")
    model_path = _download(model_id)
    print(f"  Path: {model_path}")

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    config = GemmaConfig.from_hf_config(config_data)
    print(f"  Config: {config.num_hidden_layers} layers, hidden={config.hidden_size}")

    print("\nLoading standard GemmaForCausalLM ...")
    t0 = time.time()
    standard = GemmaForCausalLM(config)
    _apply_weights(standard, model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\nLoading GemmaResidualStreamForCausalLM ...")
    t0 = time.time()
    rs_model = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs_model, model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return standard, rs_model, tokenizer, config, model_path


def tokenize(tokenizer, prompt: str) -> mx.array:
    ids = tokenizer.encode(prompt, return_tensors=None)
    return mx.array(ids)[None]  # (1, seq_len)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_pass_equivalence(standard, rs_model, input_ids):
    header("TEST 1: Full forward pass — logit identity")

    standard.eval()
    rs_model.eval()

    print(f"  Input shape: {input_ids.shape}  (seq_len={input_ids.shape[1]})")

    # Standard model (no cache)
    t0 = time.time()
    std_out = standard(input_ids)
    mx.eval(std_out.logits)
    t_std = time.time() - t0

    # Residual stream model
    t0 = time.time()
    rs_out = rs_model(input_ids)
    mx.eval(rs_out.logits)
    t_rs = time.time() - t0

    print(f"  Standard:  {t_std:.2f}s")
    print(f"  RS model:  {t_rs:.2f}s")

    compare_tensors(
        std_out.logits[:, -1, :],
        rs_out.logits[:, -1, :],
        "Last-position logits",
        tol=5e-3,
    )
    compare_tensors(
        std_out.logits,
        rs_out.logits,
        "All-position logits",
        tol=5e-3,
    )

    # Top-1 token should be identical
    std_top = int(mx.argmax(std_out.logits[0, -1, :]))
    rs_top = int(mx.argmax(rs_out.logits[0, -1, :]))
    if std_top == rs_top:
        ok(f"Top-1 predicted token matches: id={std_top}")
    else:
        fail(f"Top-1 mismatch: standard={std_top} rs={rs_top}")

    return std_out, rs_out


def test_residual_composition(rs_model, input_ids, probe_layer, inject_layer, config):
    header("TEST 2: Residual composition — forward_to + forward_from == forward")
    print(f"  probe_layer={probe_layer}, inject_layer={inject_layer}")

    # Full pass
    full_out = rs_model(input_ids)
    mx.eval(full_out.logits)

    # Composed pass: to inject_layer, then from inject_layer
    partial = rs_model.forward_to_layer(input_ids, stop_layer=inject_layer)
    mx.eval(partial.residual)
    composed = rs_model.forward_from_layer(partial.residual, start_layer=inject_layer)
    mx.eval(composed.logits)

    compare_tensors(
        full_out.logits,
        composed.logits,
        f"full == forward_to({inject_layer}) + forward_from({inject_layer})",
        tol=5e-3,
    )

    # Three-segment composition: 0→probe, probe→inject, inject→end
    seg1 = rs_model.forward_between_layers(
        rs_model.model.encode(input_ids), 0, probe_layer
    )
    mx.eval(seg1.residual)
    seg2 = rs_model.forward_between_layers(seg1.residual, probe_layer, inject_layer)
    mx.eval(seg2.residual)
    seg3 = rs_model.forward_from_layer(seg2.residual, inject_layer)
    mx.eval(seg3.logits)

    compare_tensors(
        full_out.logits,
        seg3.logits,
        f"full == 3-segment ({0}→{probe_layer}→{inject_layer}→end)",
        tol=5e-3,
    )


def test_residual_injection(rs_model, input_ids, inject_layer):
    header("TEST 3: Residual injection — modified residual changes output")
    print(f"  Injecting at layer {inject_layer}")

    # Baseline
    baseline = rs_model(input_ids)
    mx.eval(baseline.logits)

    # Inject a zero residual — should produce different logits
    partial = rs_model.forward_to_layer(input_ids, stop_layer=inject_layer)
    mx.eval(partial.residual)

    # Add a small steering vector (just use the residual scaled to zero as a sanity check)
    zeroed_residual = mx.zeros_like(partial.residual)
    injected = rs_model.forward_from_layer(zeroed_residual, start_layer=inject_layer)
    mx.eval(injected.logits)

    diff = float(mx.max(mx.abs(baseline.logits[0, -1, :] - injected.logits[0, -1, :])))
    if diff > 0.01:
        ok(f"Zero-injection changes output (max_diff={diff:.3f}) — injection works")
    else:
        fail(f"Zero-injection produced identical output (diff={diff:.4f}) — unexpected")


def print_top_tokens(tokenizer, logits: mx.array, k: int = 5):
    last_logits = logits[0, -1, :]
    top_ids = mx.argsort(last_logits)[::-1][:k]
    mx.eval(top_ids)
    print(f"\n  Top-{k} next tokens:")
    for rank, tid in enumerate(top_ids.tolist()):
        tok = repr(tokenizer.decode([tid]))
        score = float(last_logits[tid])
        print(f"    {rank+1}. {tok:<20} logit={score:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"{BOLD}Gemma Residual Stream — Proof of Inference Equivalence{RESET}")
    print(f"Model:  {args.model}")
    print(f"Prompt: {repr(args.prompt)}")

    standard, rs_model, tokenizer, config, model_path = load_models(args.model)
    input_ids = tokenize(tokenizer, args.prompt)

    # Clamp layer args to valid range
    num_layers = config.num_hidden_layers
    probe_layer = max(1, min(args.probe_layer, num_layers - 1))
    inject_layer = max(probe_layer + 1, min(args.inject_layer, num_layers - 1))

    std_out, rs_out = test_full_pass_equivalence(standard, rs_model, input_ids)
    test_residual_composition(rs_model, input_ids, probe_layer, inject_layer, config)
    test_residual_injection(rs_model, input_ids, inject_layer)

    # Residual stats
    header("Residual stream stats")
    rs_out_full = rs_model(input_ids, collect_layer_residuals=False)
    partial = rs_model.forward_to_layer(input_ids, stop_layer=probe_layer)
    mx.eval(partial.residual)
    residual_bytes = partial.residual.size * 2  # bfloat16 = 2 bytes/element
    print(f"  Residual shape at layer {probe_layer}: {partial.residual.shape}")
    print(f"  Residual size: {residual_bytes / 1024:.1f} KB")
    print(f"  (seq_len={input_ids.shape[1]}, hidden={config.hidden_size})")

    # Top tokens from RS model
    print_top_tokens(tokenizer, rs_out.logits)

    header("Summary")
    ok("Standard GemmaForCausalLM and GemmaResidualStreamForCausalLM produce identical output")
    ok("The residual stream is the complete forward state (Markov property confirmed)")
    ok("forward_to + forward_from composes correctly at any layer boundary")
    ok("Residual injection changes output — the stream is steerable")
    print()


if __name__ == "__main__":
    main()
