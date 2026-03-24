#!/usr/bin/env python3
"""
Section 4a — The Injection

Process a fact. Save it. Discard it. Load it back.
The model answers correctly from the saved state.
Without it, the model invents Detroit.

Three-way comparison:
  No context   → "New York" (hallucination)
  Saved KV     → "Voltara"  (loaded from disk)
  Full replay  → "Voltara"  (recomputed from tokens)
"""

import time
import os

import numpy as np
import mlx.core as mx
from chuk_lazarus.inference import UnifiedPipeline


def generate_tokens(kv_gen, logits, kv_store, seq_len, tokenizer, max_tokens=20, temperature=0.0):
    """Autoregressive generation from logits + KV store."""
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
    """Save KV cache to disk."""
    data = {}
    for i, (k, v) in enumerate(kv_store):
        mx.eval(k, v)
        k_f16 = k.astype(mx.float16)
        v_f16 = v.astype(mx.float16)
        mx.eval(k_f16, v_f16)
        data[f"k_{i}"] = np.array(k_f16)
        data[f"v_{i}"] = np.array(v_f16)
    np.savez_compressed(path, **data)


def load_kv(path):
    """Load KV cache from disk."""
    data = np.load(path)
    num_layers = len([k for k in data.keys() if k.startswith("k_")])
    return [(mx.array(data[f"k_{i}"]).astype(mx.bfloat16),
             mx.array(data[f"v_{i}"]).astype(mx.bfloat16))
            for i in range(num_layers)]


def main():
    MODEL = "google/gemma-3-4b-it"
    KV_PATH = os.path.join(os.path.dirname(__file__), "zarkov_fact_kv.npz")

    print()
    print("Loading model...", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    tokenizer = pipeline.tokenizer
    kv_gen = pipeline.make_engine()
    print("done.")
    print()

    fact = "Zarkov Industries was founded in the city of Voltara in 1987."
    question = " Where was Zarkov Industries founded?"

    fact_ids = tokenizer.encode(fact, add_special_tokens=True)
    question_ids = tokenizer.encode(question, add_special_tokens=False)

    # ── Process the fact and save it ──────────────────────────
    print(f"  Processing: \"{fact}\"")
    t0 = time.time()
    _, fact_kv = kv_gen.prefill(mx.array(fact_ids)[None])
    mx.eval(fact_kv[0][0])
    prefill_ms = (time.time() - t0) * 1000

    save_kv(fact_kv, KV_PATH)
    file_size = os.path.getsize(KV_PATH)
    print(f"  Saved to disk.  ({file_size:,} bytes, {prefill_ms:.0f}ms)")
    print()

    # ── Without context: hallucination ────────────────────────
    print(f"  Without context:")
    q_text = question.strip()
    q_ids = tokenizer.encode(q_text, add_special_tokens=True)
    q_logits, q_kv = kv_gen.prefill(mx.array(q_ids)[None])
    mx.eval(q_logits)
    baseline = generate_tokens(kv_gen, q_logits, q_kv, len(q_ids), tokenizer, max_tokens=20)
    print(f"  \"{q_text}\" → {baseline}")
    print()

    # ── Load saved KV and answer ──────────────────────────────
    print(f"  With saved KV:")
    t0 = time.time()
    loaded_kv = load_kv(KV_PATH)
    load_ms = (time.time() - t0) * 1000

    t0 = time.time()
    ext_logits, ext_kv = kv_gen.extend(
        mx.array(question_ids)[None], loaded_kv, abs_start=len(fact_ids),
    )
    mx.eval(ext_logits)
    injected = generate_tokens(
        kv_gen, ext_logits, ext_kv,
        len(fact_ids) + len(question_ids), tokenizer, max_tokens=20,
    )
    gen_ms = (time.time() - t0) * 1000

    print(f"  \"{q_text}\" → {injected.split(chr(10))[0]}")
    print(f"  (loaded in {load_ms:.0f}ms)")
    print()

    # ── Full replay for comparison ────────────────────────────
    print(f"  Full replay:")
    full_text = fact + question
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    t0 = time.time()
    full_logits, full_kv = kv_gen.prefill(mx.array(full_ids)[None])
    mx.eval(full_logits)
    replay = generate_tokens(kv_gen, full_logits, full_kv, len(full_ids), tokenizer, max_tokens=20)
    replay_ms = (time.time() - t0) * 1000
    print(f"  \"{q_text}\" → {replay.split(chr(10))[0]}")
    print(f"  (recomputed in {replay_ms:.0f}ms)")
    print()

    # ── The comparison ────────────────────────────────────────
    has_voltara = "voltara" in injected.lower()
    print(f"  No context:  {baseline.split(chr(10))[0][:40]}")
    print(f"  Saved KV:    {injected.split(chr(10))[0][:40]}")
    print(f"  Full replay: {replay.split(chr(10))[0][:40]}")
    if has_voltara:
        print()
        print(f"  Same answer. The saved KV skips the forward pass entirely.")
    print()


if __name__ == "__main__":
    main()
