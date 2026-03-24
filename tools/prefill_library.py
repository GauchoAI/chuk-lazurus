#!/usr/bin/env python3
"""
Pre-fill a text document into a checkpoint library.

The resulting library directory can be loaded instantly at inference time —
no prefill step required.  Multiple libraries can be loaded and queried
simultaneously via generate_cross_library().

Usage
-----
    # Single library:
    uv run python tools/prefill_library.py \\
        --model mlx-community/gemma-3-270m-it-bf16 \\
        --input texts/meridian_lore.txt \\
        --output libraries/gemma-3-270m-it-bf16/meridian \\
        --window-size 512 \\
        --name "The World of Meridian"

    # All three demo libraries for the 270M model:
    MODEL=mlx-community/gemma-3-270m-it-bf16
    SLUG=gemma-3-270m-it-bf16
    uv run python tools/prefill_library.py --model $MODEL \\
        --input texts/meridian_lore.txt     --output libraries/$SLUG/meridian  --name "The World of Meridian"
    uv run python tools/prefill_library.py --model $MODEL \\
        --input texts/resonance_eng.txt     --output libraries/$SLUG/resonance --name "Resonance Engineering"
    uv run python tools/prefill_library.py --model $MODEL \\
        --input texts/aethermoor_charter.txt --output libraries/$SLUG/charter  --name "Aethermoor City Charter"

Note: the output path is user-controlled, so you can organise libraries however
you like.  The cross_library_demo.py uses libraries/<model-slug>/ by default.

Output format
-------------
    <output>/
    ├── manifest.json         # name, model, token count, window count, ...
    ├── checkpoints.npz       # per-window last-position K,V per layer
    ├── tokens.bin            # raw uint16 token IDs
    └── windows.json          # per-window metadata
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

# Checkpoint library types (no magic strings, no dict goop)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from chuk_lazarus.inference.context import (  # noqa: E402
    LibraryFile,
    LibraryFormatVersion,
    LibraryManifest,
    WindowMeta,
)

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def compute_config_hash(config) -> str:
    """Stable hash of the model config key fields."""
    data = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
    }
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                model_id,
                local_files_only=True,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
            )
        )
    except Exception:
        pass
    print(f"  Downloading {model_id} ...")
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
    )


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


def load_engine(model_id: str):
    """Load model, config, tokenizer, and UnlimitedContextEngine class."""
    from transformers import AutoTokenizer

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config = GemmaConfig.from_hf_config(json.load(f))

    rs = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs, model_path)
    rs.eval()

    # Load inference modules via importlib so relative imports resolve
    inf = Path(__file__).parent.parent / "src/chuk_lazarus/inference"

    def _load(dotted, fpath):
        spec = importlib.util.spec_from_file_location(dotted, fpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("chuk_lazarus.inference.kv_generator", inf / "kv_generator.py")
    eng = _load("chuk_lazarus.inference.unlimited_engine", inf / "unlimited_engine.py")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return rs, config, eng.UnlimitedContextEngine, tokenizer


# ---------------------------------------------------------------------------
# Incremental save / resume helpers
# ---------------------------------------------------------------------------


def _save_partial(
    engine,
    output_path: Path,
    all_token_ids: list[int],
    name: str,
    model_id: str,
    config,
    window_size: int,
    tokenizer,
    created_at: str,
    is_complete: bool = False,
) -> None:
    """
    Write all four library files with the engine's current archived state.

    Safe to call after every window — files are written atomically in order
    (manifest last) so a partial directory is always loadable.
    """
    s = engine.stats()
    num_archived = s.archived_windows
    if num_archived == 0:
        return  # nothing to persist yet

    output_path.mkdir(parents=True, exist_ok=True)

    # ── Collect per-window data ────────────────────────────────────────
    windows: list[WindowMeta] = []
    ckpt_dict: dict[str, mx.array] = {}
    token_offset = 0

    for wid in range(num_archived):
        w_tokens, w_abs = engine.archive.retrieve(wid)
        kv_last, _ = engine.checkpoints.load(wid)
        preview = tokenizer.decode(w_tokens[:30], skip_special_tokens=True)
        windows.append(
            WindowMeta(
                window_id=wid,
                token_offset=token_offset,
                token_count=len(w_tokens),
                abs_offset=w_abs,
                preview=preview.replace("\n", " ")[:80],
            )
        )
        for li, (k, v) in enumerate(kv_last):
            ckpt_dict[f"w{wid}_l{li}_k"] = k
            ckpt_dict[f"w{wid}_l{li}_v"] = v
        token_offset += len(w_tokens)

    total_tokens_to_report = len(all_token_ids) if is_complete else token_offset

    # 1. Checkpoints.npz
    mx.savez(str(output_path / LibraryFile.CHECKPOINTS), **ckpt_dict)

    # 2. Tokens.bin (uint32 — supports vocab > 65535, e.g. Gemma 256 K)
    with open(output_path / LibraryFile.TOKENS, "wb") as f:
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            for tid in w_tokens:
                f.write(struct.pack("<I", tid))

    # 3. Windows.json
    (output_path / LibraryFile.WINDOWS).write_text(
        json.dumps([w.model_dump() for w in windows], indent=2, ensure_ascii=False)
    )

    # 4. Manifest — written last so it's the "committed" marker
    manifest = LibraryManifest(
        name=name,
        model_id=model_id,
        model_config_hash=compute_config_hash(config),
        num_layers=config.num_hidden_layers,
        window_size=window_size,
        total_tokens=total_tokens_to_report,
        num_windows=num_archived,
        checkpoint_bytes=s.checkpoint_bytes,
        archive_bytes=s.archive_bytes,
        created_at=created_at,
        format_version=LibraryFormatVersion.V1,
    )
    (output_path / LibraryFile.MANIFEST).write_text(manifest.model_dump_json(indent=2))


def _restore_engine(engine, output_path: Path) -> int:
    """
    Reload archived windows from a partial library into a fresh engine.

    Returns the number of tokens already processed (= tokens to skip on resume).
    Returns 0 if nothing to restore.
    """
    windows_path = output_path / LibraryFile.WINDOWS
    ckpt_path = output_path / LibraryFile.CHECKPOINTS
    tokens_path = output_path / LibraryFile.TOKENS

    if not (windows_path.exists() and ckpt_path.exists() and tokens_path.exists()):
        return 0

    raw_windows: list[dict] = json.loads(windows_path.read_text())
    if not raw_windows:
        return 0

    raw_ckpts: dict[str, mx.array] = dict(mx.load(str(ckpt_path)))
    num_layers = engine.config.num_hidden_layers

    token_bytes = tokens_path.read_bytes()
    n = len(token_bytes) // 4  # uint32 — 4 bytes per token
    all_saved_tokens = list(struct.unpack(f"<{n}I", token_bytes[: n * 4]))

    token_offset = 0
    for w in raw_windows:
        wid = w["window_id"]
        w_tokens = all_saved_tokens[token_offset : token_offset + w["token_count"]]
        w_abs = w["abs_offset"]

        engine.archive.archive(wid, w_tokens, w_abs)

        kv_last = [
            (raw_ckpts[f"w{wid}_l{li}_k"], raw_ckpts[f"w{wid}_l{li}_v"])
            for li in range(num_layers)
        ]
        abs_last = w_abs + w["token_count"] - 1
        engine.checkpoints.save(wid, kv_last, abs_last)
        token_offset += w["token_count"]

    engine.current_window_id = len(raw_windows)
    engine.abs_offset = token_offset
    engine.kv_store = None
    engine.hot_len = 0
    engine.current_window_tokens = []

    return token_offset


# ---------------------------------------------------------------------------
# Pre-fill a single library
# ---------------------------------------------------------------------------


def prefill_library(
    rs_model,
    config,
    EngineClass,
    tokenizer,
    input_path: Path,
    output_path: Path,
    window_size: int,
    name: str,
    model_id: str,
    resume: bool = False,
) -> None:
    """Tokenise `input_path`, run through the engine, save library to `output_path`.

    Saves incrementally after every completed window so the run can be
    interrupted (Ctrl-C) and resumed with ``--resume``.
    """
    text = input_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    total_windows = (total_tokens + window_size - 1) // window_size

    print(f"  Text:    {len(text):,} chars → {total_tokens:,} tokens")
    print(f"  Windows: ~{total_windows} (window_size={window_size})")

    # ── Build engine and warm-up the compute graph ─────────────────────
    engine = EngineClass(rs_model, config, window_size=window_size)
    _warm = mx.array([[1, 2, 3]])
    _, _kv = engine.kv_gen.prefill(_warm)
    mx.eval()

    created_at = datetime.now(timezone.utc).isoformat()
    resume_tokens = 0

    # ── Resume: restore archived windows from an existing partial save ──
    if resume and (output_path / LibraryFile.WINDOWS).exists():
        resume_tokens = _restore_engine(engine, output_path)
        if resume_tokens > 0:
            manifest_path = output_path / LibraryFile.MANIFEST
            if manifest_path.exists():
                saved = json.loads(manifest_path.read_text())
                created_at = saved.get("created_at", created_at)
            pct = resume_tokens / total_tokens
            print(
                f"  {YELLOW}Resuming from token {resume_tokens}/{total_tokens} "
                f"({pct:.0%}, {engine.current_window_id} windows already done){RESET}"
            )
            if resume_tokens >= total_tokens:
                print("  Already fully prefilled. Nothing to do.")
                return

    # ── Process window-by-window, saving after each completed window ────
    remaining = token_ids[resume_tokens:]
    t0 = time.perf_counter()
    interrupted = False
    tokens_done = resume_tokens

    try:
        for i in range(0, len(remaining), window_size):
            chunk = remaining[i : i + window_size]
            engine.process(chunk)
            tokens_done = resume_tokens + i + len(chunk)

            elapsed = time.perf_counter() - t0
            rate = (tokens_done - resume_tokens) / elapsed if elapsed > 0 else 0.0
            archived = engine.stats().archived_windows
            print(
                f"\r  [{archived}/{total_windows} windows]  "
                f"{tokens_done}/{total_tokens} tokens  {rate:.0f} tok/s",
                end="",
                flush=True,
            )

            # Persist every completed window immediately
            if archived > 0:
                _save_partial(
                    engine, output_path, token_ids,
                    name, model_id, config, window_size,
                    tokenizer, created_at,
                )

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n  {YELLOW}Interrupted — partial save written.{RESET}", flush=True)

    # ── Final flush: close any trailing partial window ──────────────────
    if not interrupted:
        engine.flush()

    elapsed = time.perf_counter() - t0
    _save_partial(
        engine, output_path, token_ids,
        name, model_id, config, window_size,
        tokenizer, created_at,
        is_complete=not interrupted,
    )
    print()  # newline after progress line

    s = engine.stats()
    total_disk = s.checkpoint_bytes + s.archive_bytes
    equiv_kv = s.equivalent_kv_bytes
    ratio = equiv_kv / max(total_disk, 1)

    if interrupted:
        print(
            f"\n  {YELLOW}Partial library: {s.archived_windows} windows, "
            f"{tokens_done}/{total_tokens} tokens saved.{RESET}"
        )
        print(
            f"  Resume with:\n"
            f"    uv run python tools/prefill_library.py "
            f"--model {model_id} --input {input_path} --output {output_path} --resume"
        )
        return

    print(f"  Processed in {elapsed * 1000:.0f} ms")
    print(f"  Archived windows:  {s.archived_windows}")
    print(f"  Checkpoint bytes:  {fmt_bytes(s.checkpoint_bytes)}")
    print(f"  Token archive:     {fmt_bytes(s.archive_bytes)}")
    print(f"\n  {GREEN}Library saved → {output_path}/{RESET}")
    print(f"  Disk (npz+bin):  {fmt_bytes(total_disk)}")
    print(f"  Equivalent KV:   {fmt_bytes(equiv_kv)}")
    print(f"  Compression:     {ratio:.0f}×")
    print("  Windows:")
    for wid in range(s.archived_windows):
        w_tokens, w_abs = engine.archive.retrieve(wid)
        preview = tokenizer.decode(w_tokens[:30], skip_special_tokens=True)
        preview = preview.replace("\n", " ")[:80]
        print(
            f"    [{wid}] offset={w_abs:>5}, "
            f"tokens={len(w_tokens):>4} — {DIM}{preview}{RESET}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-fill a text into a checkpoint library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default="mlx-community/gemma-3-270m-it-bf16",
        help="HuggingFace model ID or local path",
    )
    p.add_argument("--input", required=True, help="Path to input text file")
    p.add_argument("--output", required=True, help="Output directory for the library")
    p.add_argument("--window-size", type=int, default=512, help="Tokens per window (default: 512)")
    p.add_argument(
        "--name",
        default=None,
        help="Human-readable name for the library (defaults to input filename stem)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previously interrupted prefill (skips already-saved windows)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    name = args.name or input_path.stem.replace("_", " ").title()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    print(f"\n{BOLD}Pre-filling library: {name}{RESET}")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Model:       {args.model}")
    print(f"  Window size: {args.window_size}")
    print()

    print("Loading model ...")
    rs, config, EngineClass, tokenizer = load_engine(args.model)
    print(
        f"  {config.num_hidden_layers} layers, "
        f"hidden={config.hidden_size}, "
        f"kv_heads={config.num_key_value_heads}, "
        f"head_dim={config.head_dim}"
    )
    print()

    print("Pre-filling ...")
    prefill_library(
        rs_model=rs,
        config=config,
        EngineClass=EngineClass,
        tokenizer=tokenizer,
        input_path=input_path,
        output_path=output_path,
        window_size=args.window_size,
        name=name,
        model_id=args.model,
        resume=args.resume,
    )
    print()
    print(f"{BOLD}Done.{RESET}  Load with:")
    print("  from chuk_lazarus.inference.context import CheckpointLibrary")
    print(f"  lib = CheckpointLibrary('{output_path}')")


if __name__ == "__main__":
    main()
