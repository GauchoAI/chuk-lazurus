#!/usr/bin/env python3
"""
Prove that memory bank fact lookup works.

Concept: The model has parametric memory (weights), but we give it an
external memory bank that it should prefer. Like a person checking their
notes instead of going from memory.

This script demonstrates:
1. Model answers from parametric memory (weights) - may hallucinate
2. Model answers from memory bank (injected facts) - deterministic, correct
3. Memory bank overrides parametric memory even for counterfactuals

Run: python experiments/memory_fact_retrieval/scripts/prove_fact_lookup.py
"""

import mlx.core as mx


# =============================================================================
# Memory Bank
# =============================================================================


MEMORY_BANK = {
    # Correct facts
    ("france", "capital"): "Paris",
    ("japan", "capital"): "Tokyo",
    ("germany", "capital"): "Berlin",
    ("australia", "capital"): "Canberra",
    ("brazil", "capital"): "Brasília",
    ("gold", "symbol"): "Au",
    ("silver", "symbol"): "Ag",
    ("iron", "symbol"): "Fe",
    ("apple", "ceo"): "Tim Cook",
    ("microsoft", "ceo"): "Satya Nadella",
}

# Counterfactual overrides (to prove memory bank takes precedence)
MEMORY_BANK_COUNTERFACTUAL = {
    ("france", "capital"): "Lyon",
    ("japan", "capital"): "Osaka",
    ("gold", "symbol"): "Gd",
}


def memory_bank_lookup(entity: str, relation: str, bank: dict) -> str | None:
    """Look up a fact in the memory bank."""
    return bank.get((entity.lower(), relation))


def format_memory_context(facts: list[tuple[str, str, str]]) -> str:
    """Format memory bank entries as context for the model."""
    lines = ["[Memory Bank]"]
    for entity, relation, value in facts:
        lines.append(f"- {entity} | {relation} | {value}")
    lines.append("[End Memory Bank]")
    return "\n".join(lines)


# =============================================================================
# Model
# =============================================================================


def load_model():
    """Load GPT-OSS."""
    from chuk_lazarus.models_v2.loader import load_model as lm
    print("Loading model...")
    loaded = lm("openai/gpt-oss-20b")
    if loaded.tokenizer.pad_token is None:
        loaded.tokenizer.pad_token = loaded.tokenizer.eos_token
    mx.eval(loaded.model.parameters())
    return loaded.model, loaded.tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 30) -> str:
    """Generate response."""
    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = mx.array(tokens["input_ids"])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        next_token = mx.argmax(output.logits[0, -1, :])
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

    return tokenizer.decode(generated).strip()


# =============================================================================
# Experiment
# =============================================================================


def main():
    model, tokenizer = load_model()

    print("\n" + "="*70)
    print("MEMORY BANK FACT LOOKUP - PROOF OF CONCEPT")
    print("="*70)

    # ─────────────────────────────────────────────────────────────────
    # Test 1: Parametric vs Memory Bank (correct facts)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("TEST 1: Memory Bank with correct facts")
    print("─"*70)
    print("Does the model use memory bank entries when provided?\n")

    correct_tests = [
        ("What is the capital of France?", "france", "capital", "Paris"),
        ("What is the capital of Australia?", "australia", "capital", "Canberra"),
        ("What is the chemical symbol for gold?", "gold", "symbol", "Au"),
        ("Who is the CEO of Microsoft?", "microsoft", "ceo", "Satya Nadella"),
    ]

    correct_results = []
    for query, entity, relation, expected in correct_tests:
        value = memory_bank_lookup(entity, relation, MEMORY_BANK)
        memory_context = format_memory_context([(entity.title(), relation, value)])

        prompt = f"{memory_context}\n\nUsing the memory bank above, answer: {query}\nAnswer:"
        answer = generate(model, tokenizer, prompt)
        success = expected.lower() in answer.lower()
        correct_results.append(success)

        print(f"  Q: {query}")
        print(f"  Memory Bank: {entity} | {relation} | {value}")
        print(f"  Answer: {answer[:60]}")
        print(f"  Contains '{expected}': {'✓' if success else '✗'}")
        print()

    print(f"  Result: {sum(correct_results)}/{len(correct_results)}")

    # ─────────────────────────────────────────────────────────────────
    # Test 2: Counterfactual override (the real test)
    # ─────────────────────────────────────────────────────────────────
    print("─"*70)
    print("TEST 2: Memory Bank OVERRIDES parametric memory")
    print("─"*70)
    print("Can the memory bank override what the model 'knows'?\n")

    override_tests = [
        ("What is the capital of France?", "france", "capital", "Lyon"),
        ("What is the capital of Japan?", "japan", "capital", "Osaka"),
        ("What is the chemical symbol for gold?", "gold", "symbol", "Gd"),
    ]

    override_results = []
    for query, entity, relation, counterfactual in override_tests:
        value = memory_bank_lookup(entity, relation, MEMORY_BANK_COUNTERFACTUAL)
        memory_context = format_memory_context([(entity.title(), relation, value)])

        # Without memory bank
        parametric = generate(model, tokenizer, query)

        # With memory bank
        prompt = f"{memory_context}\n\nUsing the memory bank above, answer: {query}\nAnswer:"
        answer = generate(model, tokenizer, prompt)
        success = counterfactual.lower() in answer.lower()
        override_results.append(success)

        print(f"  Q: {query}")
        print(f"  Parametric:  {parametric[:60]}")
        print(f"  Memory Bank: {entity} | {relation} | {value}")
        print(f"  With Bank:   {answer[:60]}")
        print(f"  Override to '{counterfactual}': {'✓ OVERRIDE' if success else '✗ FAILED'}")
        print()

    print(f"  Result: {sum(override_results)}/{len(override_results)}")

    # ─────────────────────────────────────────────────────────────────
    # Test 3: Multi-fact memory bank
    # ─────────────────────────────────────────────────────────────────
    print("─"*70)
    print("TEST 3: Multi-fact memory bank lookup")
    print("─"*70)
    print("Can the model select the right fact from multiple entries?\n")

    # Load several facts into memory
    multi_facts = [
        ("France", "capital", "Lyon"),        # counterfactual
        ("Japan", "capital", "Osaka"),         # counterfactual
        ("Germany", "capital", "Berlin"),      # correct
        ("Gold", "symbol", "Gd"),             # counterfactual
        ("Silver", "symbol", "Ag"),           # correct
    ]
    memory_context = format_memory_context(multi_facts)

    multi_tests = [
        ("What is the capital of France?", "Lyon"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the chemical symbol for silver?", "Ag"),
    ]

    multi_results = []
    for query, expected in multi_tests:
        prompt = f"{memory_context}\n\nUsing the memory bank above, answer: {query}\nAnswer:"
        answer = generate(model, tokenizer, prompt)
        success = expected.lower() in answer.lower()
        multi_results.append(success)

        print(f"  Q: {query}")
        print(f"  Answer: {answer[:60]}")
        print(f"  Contains '{expected}': {'✓' if success else '✗'}")
        print()

    print(f"  Result: {sum(multi_results)}/{len(multi_results)}")

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    total = correct_results + override_results + multi_results
    total_pass = sum(total)
    total_count = len(total)

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Correct fact lookup:     {sum(correct_results)}/{len(correct_results)}")
    print(f"  Counterfactual override: {sum(override_results)}/{len(override_results)}")
    print(f"  Multi-fact selection:    {sum(multi_results)}/{len(multi_results)}")
    print(f"  ─────────────────────────────────")
    print(f"  Total:                   {total_pass}/{total_count} ({total_pass/total_count:.0%})")
    print("="*70)

    if total_pass == total_count:
        print("\n✓ PROOF COMPLETE: Memory bank fact lookup works.")
        print("  The model treats [Memory Bank] entries as authoritative.")
        print("  Overrides parametric memory. Selects from multiple entries.")
    else:
        print(f"\n✗ {total_count - total_pass} tests failed.")


if __name__ == "__main__":
    main()
