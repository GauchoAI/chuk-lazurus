#!/usr/bin/env python3
"""
Section 2 — The Filing Cabinet

Two moments:
  1. Ask about a fact with context vs. without context.
     One gets it right, one invents New York.
  2. Ask about a completely fictional expedition.
     The model never says "I don't know." Not once.
"""

import mlx.core as mx
from chuk_lazarus.inference import UnifiedPipeline


def main():
    MODEL = "google/gemma-3-4b-it"

    print()
    print("Loading model...", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    tokenizer = pipeline.tokenizer
    kv_gen = pipeline.make_engine()
    print("done.")
    print()

    # ══════════════════════════════════════════════════════════
    # The fact exists during processing, gone after
    # ══════════════════════════════════════════════════════════
    fact = "Zarkov Industries was founded in the city of Voltara in 1987."
    question = "Where was Zarkov Industries founded?"

    # WITH the fact in context — extend the fact KV with the question
    print(f"── With the fact in context ──")
    print()
    print(f"  Context:  \"{fact}\"")
    print(f"  Question: \"{question}\"")

    fact_ids = tokenizer.encode(fact, add_special_tokens=True)
    question_ids = tokenizer.encode(" " + question, add_special_tokens=False)

    _, fact_kv = kv_gen.prefill(mx.array(fact_ids)[None])
    ext_logits, ext_kv = kv_gen.extend(
        mx.array(question_ids)[None], fact_kv, abs_start=len(fact_ids),
    )
    mx.eval(ext_logits)

    # Generate answer
    stop_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id else set()
    generated = []
    logits, gen_kv = ext_logits, ext_kv
    seq_len = len(fact_ids) + len(question_ids)
    for _ in range(20):
        last = logits[0, -1]
        tok = int(mx.argmax(last).item())
        if tok in stop_ids:
            break
        generated.append(tok)
        logits, gen_kv = kv_gen.step_uncompiled(mx.array([[tok]]), gen_kv, seq_len=seq_len)
        seq_len += 1
    with_ctx = tokenizer.decode(generated, skip_special_tokens=True).strip().split("\n")[0]
    print(f"  Answer:   {with_ctx}")
    print()

    # WITHOUT context
    print(f"── Without context ──")
    print()
    print(f"  Question: \"{question}\"")
    result = pipeline.generate(question, max_new_tokens=20, temperature=0.0)
    without_ctx = result.text.strip().split("\n")[0]
    print(f"  Answer:   {without_ctx}")
    print()

    print(f"  Same question. One has the fact. One doesn't.")
    print()

    # ══════════════════════════════════════════════════════════
    # The model never says "I don't know"
    # ══════════════════════════════════════════════════════════
    print("── The model never says \"I don't know\" ──")
    print()
    print("  Asking about a completely fictional expedition:")
    print()

    confab_queries = [
        "Who led the Zarkov expedition?",
        "How many team members were in the expedition?",
        "What was discovered during the expedition?",
        "What year did the expedition launch?",
        "What was the budget of the expedition?",
        "Who funded the Zarkov expedition?",
    ]

    hedging_count = 0
    for q in confab_queries:
        result = pipeline.generate(q, max_new_tokens=30, temperature=0.0)
        answer = result.text.strip().split("\n")[0][:70]

        hedging_words = ["i don't know", "not sure", "no information",
                         "cannot", "unclear", "don't have", "not available"]
        hedged = any(h in answer.lower() for h in hedging_words)
        if hedged:
            hedging_count += 1

        print(f"  Q: {q}")
        print(f"  A: {answer}")
        print()

    print(f"  Hedging: {hedging_count} out of {len(confab_queries)}.")
    if hedging_count == 0:
        print(f"  The model never said \"I don't know.\"")
    print()


if __name__ == "__main__":
    main()
