#!/usr/bin/env python3
"""
Context Generate Demo — windowed KV inference with live metrics.

Uses UnlimitedContextEngine (Mode 4) to ingest a document with bounded memory,
then generates from a prompt with streaming output and per-token stats.

Instead of loading a 48 GB raw KV checkpoint, this windows the document into
bounded chunks (~150 MB hot KV), checkpoints at each boundary (~174 KB each),
and archives token IDs (~16 KB per window).  370K tokens fits in ~160 MB.

Usage:
    uv run python examples/inference/context_generate_demo.py \
        --model google/gemma-3-4b-it \
        --input docs/apollo11_clean.txt \
        --prompt "Summarize the Apollo 11 mission"

    uv run python examples/inference/context_generate_demo.py \
        --model google/gemma-3-4b-it \
        --input docs/apollo11_clean.txt \
        --prompt "What were the key moments?" \
        --max-tokens 300 --temperature 0.0 --window-size 4096
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Windowed KV inference demo with live metrics"
    )
    p.add_argument("--model", "-m", required=True, help="Model ID or local path")
    p.add_argument("--input", "-i", required=True, help="Input text file")
    p.add_argument("--prompt", "-p", default=None, help="Prompt text")
    p.add_argument("--prompt-file", default=None, help="File containing the prompt")
    p.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--window-size", type=int, default=8192, help="Window size in tokens")
    p.add_argument(
        "--replay", type=int, nargs="*", default=None,
        help="Window IDs to replay for cross-window retrieval (e.g. --replay 0 1)",
    )
    p.add_argument(
        "--no-chat-template", action="store_true",
        help="Send prompt as raw text (no chat template wrapping)",
    )
    p.add_argument(
        "--stats-every", type=int, default=10,
        help="Print stats summary every N tokens (0 = only at end)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    from pathlib import Path

    from chuk_lazarus.inference.unified import UnifiedPipeline
    from chuk_lazarus.inference.context.kv_generator import make_kv_generator
    from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine

    print(f"\n{BOLD}Context Generate Demo (Windowed KV){RESET}")
    print("=" * 56)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\n{DIM}Loading model: {args.model}{RESET}")
    pipeline = UnifiedPipeline.from_pretrained(args.model, verbose=False)
    tokenizer = pipeline.tokenizer

    # ------------------------------------------------------------------
    # 2. Tokenize input
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return

    source_text = input_path.read_text(errors="replace")
    token_ids = tokenizer.encode(source_text, add_special_tokens=False)
    total_tokens = len(token_ids)

    # Model geometry
    kv_gen = make_kv_generator(pipeline.model)
    backbone = kv_gen.backbone
    num_layers = len(backbone.adapted_layers)
    layer0 = backbone.adapted_layers[0]
    num_kv_heads = layer0.num_kv_heads
    head_dim = layer0.head_dim
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
    sw = backbone.sliding_window

    naive_kv_bytes = bytes_per_token * total_tokens
    num_windows = (total_tokens + args.window_size - 1) // args.window_size
    hot_kv_bytes = bytes_per_token * args.window_size

    print(f"\n{BOLD}Input{RESET}")
    print(f"  Source:        {input_path.name}")
    print(f"  Tokens:        {total_tokens:,}")
    print(f"  Naive KV:      {fmt_bytes(naive_kv_bytes)}  (all tokens, all layers)")

    print(f"\n{BOLD}Windowed Mode{RESET}")
    print(f"  Window size:   {args.window_size:,} tokens")
    print(f"  Windows:       {num_windows}")
    print(f"  Sliding window:{f' {sw}' if sw else ' None'}")
    print(f"  HOT (1 window):{fmt_bytes(hot_kv_bytes)}")
    print(f"  Compression:   ~{naive_kv_bytes // max(hot_kv_bytes, 1)}x vs naive KV")

    # ------------------------------------------------------------------
    # 3. Ingest via UnlimitedContextEngine
    # ------------------------------------------------------------------
    engine = UnlimitedContextEngine(
        pipeline.model, pipeline.config, window_size=args.window_size
    )

    print(f"\n{DIM}Ingesting {total_tokens:,} tokens into windowed engine...{RESET}")
    ingest_start = time.time()

    # Feed in chunks to show progress
    chunk_size = args.window_size
    for offset in range(0, total_tokens, chunk_size):
        chunk = token_ids[offset : offset + chunk_size]
        engine.process(chunk)
        done = min(offset + chunk_size, total_tokens)
        elapsed = time.time() - ingest_start
        rate = done / elapsed if elapsed > 0 else 0
        print(
            f"\r  {done:,}/{total_tokens:,} tokens  "
            f"{rate:,.0f} tok/s  "
            f"windows: {engine.current_window_id}\033[K",
            end="", file=sys.stderr, flush=True,
        )

    engine.flush()
    ingest_elapsed = time.time() - ingest_start
    stats = engine.stats()

    print(file=sys.stderr)
    print(f"\n{BOLD}Engine Stats{RESET}")
    print(f"  Ingest time:     {ingest_elapsed:.1f}s ({total_tokens / ingest_elapsed:,.0f} tok/s)")
    print(f"  Windows:         {stats.archived_windows}")
    print(f"  Checkpoints:     {fmt_bytes(stats.checkpoint_bytes)}")
    print(f"  Token archive:   {fmt_bytes(stats.archive_bytes)}")
    print(f"  Total warm+cold: {fmt_bytes(stats.cold_warm_bytes)}")
    print(f"  Equivalent KV:   {fmt_bytes(stats.equivalent_kv_bytes)}")
    print(f"  Compression:     {GREEN}{stats.compression_ratio:.0f}x{RESET}")

    # ------------------------------------------------------------------
    # 4. Generate
    # ------------------------------------------------------------------
    prompt_text = args.prompt
    if not prompt_text and args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()

    if not prompt_text:
        print(f"\n{YELLOW}No prompt specified. Use --prompt to generate.{RESET}")
        return

    # Apply chat template for instruction-tuned models
    if not args.no_chat_template and hasattr(tokenizer, "apply_chat_template"):
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
        print(f"\n{BOLD}Prompt{RESET} ({len(prompt_ids)} tokens, chat template applied)")
        print(f"  {CYAN}{prompt_text}{RESET}")
        print(f"  {DIM}{chat_prompt[:100]}...{RESET}" if len(chat_prompt) > 100 else f"  {DIM}{chat_prompt}{RESET}")
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        print(f"\n{BOLD}Prompt{RESET} ({len(prompt_ids)} tokens, raw)")
        print(f"  {CYAN}{prompt_text}{RESET}")

    replay_ids = args.replay
    if replay_ids:
        print(f"  Replaying windows: {replay_ids}")

    print(f"\n{BOLD}Generating{RESET} (max {args.max_tokens} tokens, temp={args.temperature})")
    print("-" * 56)

    # Build context: replay specified windows (or last window by default)
    if replay_ids is None:
        # Default: replay last window for continuity
        last_wid = stats.archived_windows - 1
        replay_ids = [last_wid] if last_wid >= 0 else []

    # Use engine.generate for the full pipeline
    eos_id = tokenizer.eos_token_id

    gen_start = time.time()

    # For streaming + metrics, we replicate the generate loop manually
    # so we can stream tokens and measure per-step latency
    replay_parts = []
    max_abs_end = -1
    for wid in sorted(replay_ids):
        print(f"  {DIM}Replaying window {wid}...{RESET}", end="", flush=True)
        t0 = time.time()
        w_kv, w_abs_end = engine.replay_window(wid)
        print(f" {(time.time() - t0) * 1000:.0f}ms")
        replay_parts.append(w_kv)
        max_abs_end = max(max_abs_end, w_abs_end)

    # Merge replayed KV
    if replay_parts:
        context_kv = engine._merge_kv_parts(replay_parts)
    else:
        context_kv = engine._make_empty_kv()

    # Extend with prompt
    abs_start = max_abs_end + 1 if max_abs_end >= 0 else 0
    q_ids = mx.array(prompt_ids)[None]
    print(f"  {DIM}Extending with prompt...{RESET}", end="", flush=True)
    t0 = time.time()
    logits, gen_kv = engine.kv_gen.extend(q_ids, context_kv, abs_start=abs_start)
    mx.eval(logits)
    print(f" {(time.time() - t0) * 1000:.0f}ms")
    seq_len = abs_start + len(prompt_ids)

    # Context size for this generation
    ctx_seq = sum(kv[0][0].shape[2] for kv in replay_parts) + len(prompt_ids)
    ctx_bytes = bytes_per_token * ctx_seq
    print(f"  {DIM}Generation context: {ctx_seq:,} tokens ({fmt_bytes(ctx_bytes)}){RESET}")
    print()

    # Autoregressive decode with streaming
    stop_ids = {eos_id} if eos_id is not None else set()
    generated_tokens: list[int] = []
    step_times: list[float] = []

    for i in range(args.max_tokens):
        last_logits = logits[0, -1]

        if args.temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / args.temperature
            probs = mx.softmax(scaled, axis=-1)
            next_token = int(mx.random.categorical(probs[None]).item())

        if next_token in stop_ids:
            break

        generated_tokens.append(next_token)
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()

        t0 = time.time()
        logits, gen_kv = engine.kv_gen.step_uncompiled(
            mx.array([[next_token]]), gen_kv, seq_len
        )
        mx.eval(logits)
        step_ms = (time.time() - t0) * 1000
        step_times.append(step_ms)
        seq_len += 1

        # Periodic inline stats
        n = len(generated_tokens)
        if args.stats_every > 0 and n > 0 and n % args.stats_every == 0:
            avg_ms = sum(step_times[-args.stats_every:]) / args.stats_every
            sys.stdout.write(
                f"\n  {DIM}[{n} tokens | "
                f"{1000 / avg_ms:.1f} tok/s | "
                f"{avg_ms:.1f} ms/tok]{RESET}\n"
            )
            sys.stdout.flush()

    gen_elapsed = time.time() - gen_start
    n_generated = len(generated_tokens)

    print()
    print("-" * 56)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    if n_generated == 0:
        print(f"\n{YELLOW}No tokens generated (hit EOS immediately){RESET}")
        return

    avg_ms = sum(step_times) / n_generated
    p50 = sorted(step_times)[n_generated // 2]
    p99 = sorted(step_times)[min(int(n_generated * 0.99), n_generated - 1)]
    first_token_ms = step_times[0]

    print(f"\n{BOLD}Performance{RESET}")
    print(f"  Document:        {total_tokens:,} tokens ({fmt_bytes(naive_kv_bytes)} naive KV)")
    print(f"  Generation ctx:  {ctx_seq:,} tokens ({fmt_bytes(ctx_bytes)})")
    print(f"  Generated:       {n_generated} tokens")
    print(f"  Throughput:      {GREEN}{n_generated / gen_elapsed:.1f} tok/s{RESET}")
    print(f"  First token:     {first_token_ms:.1f}ms")
    print(f"  Avg latency:     {avg_ms:.1f}ms/tok")
    print(f"  P50 latency:     {p50:.1f}ms")
    print(f"  P99 latency:     {p99:.1f}ms")

    print(f"\n{BOLD}Memory{RESET}")
    print(f"  Naive KV (all):  {fmt_bytes(naive_kv_bytes)}")
    print(f"  Windowed total:  {fmt_bytes(stats.cold_warm_bytes)}")
    print(f"  Gen context KV:  {fmt_bytes(ctx_bytes)}")
    print(f"  Compression:     {GREEN}{stats.compression_ratio:.0f}x{RESET}")
    print()


if __name__ == "__main__":
    main()
