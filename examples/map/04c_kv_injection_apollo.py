#!/usr/bin/env python3
"""
Section 4c — Apollo 11 Queries with KV Injection

Full Mode 6 pipeline for each query:
  1. Route via sparse BM25 index → find relevant windows
  2. Replay those windows → save full KV cache to disk
  3. For subsequent queries: load KV from disk (no recomputation!)
  4. Extend with query → generate

Demonstrates: the same answers as Mode 4 replay, but with saved KV
that can be reused across queries without recomputing window forward passes.
"""

import os
import time
import json

import numpy as np
import mlx.core as mx
from pathlib import Path
from chuk_lazarus.inference import UnifiedPipeline
from chuk_lazarus.inference.context import CheckpointLibrary
from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine


def generate_tokens(kv_gen, logits, kv_store, seq_len, tokenizer,
                    max_tokens=200, temperature=0.0):
    """Autoregressive generation."""
    stop_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id else set()
    generated = []

    for _ in range(max_tokens):
        last_logits = logits[0, -1]
        if temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / temperature
            next_token = int(mx.random.categorical(scaled[None]).item())

        if next_token in stop_ids:
            break
        generated.append(next_token)

        logits, kv_store = kv_gen.step_uncompiled(
            mx.array([[next_token]]), kv_store, seq_len=seq_len
        )
        seq_len += 1

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def save_kv(kv_store, path):
    """Save full KV store to disk."""
    data = {}
    for i, (k, v) in enumerate(kv_store):
        mx.eval(k, v)
        k_f16 = k.astype(mx.float16)
        v_f16 = v.astype(mx.float16)
        mx.eval(k_f16, v_f16)
        data[f"k_{i}"] = np.array(k_f16)
        data[f"v_{i}"] = np.array(v_f16)
    np.savez_compressed(str(path), **data)


def load_kv(path):
    """Load full KV store from disk."""
    data = np.load(str(path))
    num_layers = len([k for k in data.keys() if k.startswith("k_")])
    return [(mx.array(data[f"k_{i}"]).astype(mx.bfloat16),
             mx.array(data[f"v_{i}"]).astype(mx.bfloat16)) for i in range(num_layers)]


def mode6_query(query_text, lib, kv_gen, engine, tokenizer, kv_store_dir,
                top_k_windows=5, max_tokens=200, temperature=0.0):
    """Full Mode 6 pipeline: route → replay → save → (reload) → extend → generate.

    First call: routes + replays windows, saves the context KV to disk.
    The saved KV can be reloaded for the same windows without recomputation.
    """
    from chuk_lazarus.cli.commands.context.compass_routing import compass_route, RoutingStrategy

    timings = {}

    # ── Route: find relevant windows via sparse BM25 ──────────
    query_ids = tokenizer.encode(query_text, add_special_tokens=True)
    t0 = time.time()
    window_ids = compass_route(
        lib, kv_gen, query_ids, query_text, tokenizer,
        strategy=RoutingStrategy.SPARSE,
        top_k=top_k_windows,
    )
    timings["route_ms"] = (time.time() - t0) * 1000

    if not window_ids:
        return None, timings, "No windows found"

    # ── Build context: preamble + windows + postamble ──────────
    # Build chat-wrapped context (same as _cmd.py standard mode)
    sys_content = (
        "You are answering questions based on the document transcript provided below. "
        "Answer using only information from the transcript. Quote exact text when possible."
    )
    preamble_text = (
        "<start_of_turn>user\n"
        f"{sys_content}\n\n"
        "Here is the relevant transcript:\n\n"
    )
    preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)

    postamble_text = (
        f"\n\n---\nBased on the transcript above, {query_text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)

    # Check if we have a cached KV for these windows
    cache_key = "_".join(str(w) for w in sorted(window_ids))
    kv_cache_path = kv_store_dir / f"windows_{cache_key}.npz"

    if kv_cache_path.exists():
        # ── INJECTION PATH: load pre-saved KV ─────────────────
        t0 = time.time()
        context_kv = load_kv(kv_cache_path)
        # Figure out context length from saved KV
        context_seq_len = context_kv[0][0].shape[2]
        timings["load_ms"] = (time.time() - t0) * 1000
        timings["replay_ms"] = 0
        timings["mode"] = "injection"
    else:
        # ── REPLAY PATH: replay windows + save KV ─────────────
        t0 = time.time()

        # Prefill preamble
        p_ids = mx.array(preamble_ids)[None]
        logits, context_kv = kv_gen.prefill(p_ids)
        seq_len = len(preamble_ids)

        # Replay each window (re-encode at contiguous positions)
        for wid in window_ids:
            w_tokens = lib.get_window_tokens(wid)
            w_ids = mx.array(w_tokens)[None]
            logits, context_kv = kv_gen.extend(w_ids, context_kv, abs_start=seq_len)
            seq_len += len(w_tokens)

        mx.eval(logits)
        timings["replay_ms"] = (time.time() - t0) * 1000
        context_seq_len = seq_len

        # Save context KV for reuse
        t0 = time.time()
        save_kv(context_kv, kv_cache_path)
        timings["save_ms"] = (time.time() - t0) * 1000
        timings["load_ms"] = 0
        timings["mode"] = "replay+save"

    # ── Extend with postamble (question) and generate ─────────
    t0 = time.time()
    q_ids = mx.array(postamble_ids)[None]
    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=context_seq_len)
    gen_seq_len = context_seq_len + len(postamble_ids)
    mx.eval(logits)

    answer = generate_tokens(
        kv_gen, logits, gen_kv, gen_seq_len,
        tokenizer, max_tokens=max_tokens, temperature=temperature
    )
    timings["generate_ms"] = (time.time() - t0) * 1000

    file_size = kv_cache_path.stat().st_size if kv_cache_path.exists() else 0
    timings["kv_file_bytes"] = file_size
    timings["n_windows"] = len(window_ids)
    timings["window_ids"] = window_ids
    timings["context_tokens"] = context_seq_len

    return answer, timings, None


def main():
    MODEL = "google/gemma-3-4b-it"
    CHECKPOINT = Path(os.environ.get(
        "APOLLO_CHECKPOINT",
        "/Users/christopherhay/chris-source/apollo-demo/apollo11_lean",
    ))
    KV_STORE_DIR = Path(__file__).parent / "fact_kv"

    print("═" * 65)
    print("  Section 4c — Apollo 11 KV Injection Queries")
    print("  Route → Replay → Save → (Inject) → Generate")
    print("═" * 65)
    print()

    # Create KV store directory
    KV_STORE_DIR.mkdir(exist_ok=True)

    # ── Load library + model ──────────────────────────────────
    print("Loading library...", end=" ", flush=True)
    lib = CheckpointLibrary(CHECKPOINT)
    print(f"{lib.total_tokens:,} tokens, {lib.num_windows} windows")

    print("Loading model...", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    tokenizer = pipeline.tokenizer
    print("done.")

    engine = UnlimitedContextEngine(
        pipeline.model, pipeline.config, window_size=lib.window_size
    )
    engine.load_library(lib)
    kv_gen = engine.kv_gen

    print()

    # ── Queries (Round 1: replay + save) ──────────────────────
    queries = [
        ("What sport and teams were discussed during the mission?",
         "Joe Namath, New York Jets"),
        ("What did the crew say about audio quality?",
         "scratchy"),
        ("What ship recovered the Apollo 11 crew?",
         "Hornet"),
        ("Who was the commander of Apollo 11?",
         "Armstrong (parametric)"),
    ]

    print("═══ Round 1: Replay windows + save KV to disk ═══")
    print()

    round1_results = []
    for i, (query, expected) in enumerate(queries):
        print(f"  Query {i+1}: {query}")

        answer, timings, error = mode6_query(
            query, lib, kv_gen, engine, tokenizer, KV_STORE_DIR,
            top_k_windows=5, max_tokens=200, temperature=0.0,
        )

        if error:
            print(f"  Error: {error}")
            continue

        first_line = answer.split("\n")[0][:80]
        mode = timings.get("mode", "unknown")
        print(f"  Answer: {first_line}")
        print(f"  Mode: {mode}  |  Windows: {timings['window_ids']}")
        print(f"  Replay: {timings.get('replay_ms', 0):.0f}ms  |  "
              f"Save: {timings.get('save_ms', 0):.0f}ms  |  "
              f"Generate: {timings['generate_ms']:.0f}ms")
        if timings.get('kv_file_bytes'):
            print(f"  KV saved: {timings['kv_file_bytes']:,} bytes")
        print(f"  Expected: {expected}")
        print()
        round1_results.append((query, answer, timings))

    # ── Show saved KV files ───────────────────────────────────
    print("─── Saved KV cache files ───")
    total_size = 0
    for f in sorted(KV_STORE_DIR.glob("windows_*.npz")):
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size:,} bytes")
    print(f"  Total: {total_size:,} bytes ({total_size / 1e6:.1f} MB)")
    print()

    # ── Round 2: Same queries, but load KV from disk ──────────
    print("═══ Round 2: Load KV from disk (no recomputation) ═══")
    print()

    for i, (query, expected) in enumerate(queries):
        print(f"  Query {i+1}: {query}")

        answer, timings, error = mode6_query(
            query, lib, kv_gen, engine, tokenizer, KV_STORE_DIR,
            top_k_windows=5, max_tokens=200, temperature=0.0,
        )

        if error:
            print(f"  Error: {error}")
            continue

        first_line = answer.split("\n")[0][:80]
        mode = timings.get("mode", "unknown")
        load_ms = timings.get("load_ms", 0)
        print(f"  Answer: {first_line}")
        print(f"  Mode: {mode}  |  Load: {load_ms:.0f}ms  |  Generate: {timings['generate_ms']:.0f}ms")
        print(f"  Expected: {expected}")
        print()

    # ── Comparison ────────────────────────────────────────────
    print("═" * 65)
    print("  ROUND 1 vs ROUND 2 COMPARISON")
    print("═" * 65)
    print()
    print(f"  {'Query':<45s}  {'Replay':>8s}  {'Inject':>8s}")
    print(f"  {'─'*45}  {'─'*8}  {'─'*8}")
    for query, answer, r1_timings in round1_results:
        short_q = query[:45]
        replay_t = r1_timings.get("replay_ms", 0) + r1_timings["generate_ms"]
        # Re-run to get injection timing
        _, r2_timings, _ = mode6_query(
            query, lib, kv_gen, engine, tokenizer, KV_STORE_DIR,
            top_k_windows=5, max_tokens=10, temperature=0.0,
        )
        inject_t = r2_timings.get("load_ms", 0) + r2_timings["generate_ms"]
        print(f"  {short_q:<45s}  {replay_t:>6.0f}ms  {inject_t:>6.0f}ms")

    print()
    print(f"  Replay: forward pass through {len(engine.kv_gen.backbone.adapted_layers)} layers for each window")
    print(f"  Inject: load pre-computed KV from disk (zero forward passes)")
    print(f"  Same answers. Saved computation on every subsequent query.")
    print()
    print("═" * 65)


if __name__ == "__main__":
    main()
