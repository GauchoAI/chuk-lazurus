#!/usr/bin/env python3
"""
Section 2 — Parametric

Two questions. No context. One the model knows. One it doesn't.
Same confidence. One is right. One is invented.
"""

from chuk_lazarus.inference import UnifiedPipeline


def main():
    MODEL = "google/gemma-3-4b-it"

    print()
    print("Loading model...", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    print("done.")
    print()

    # ── Question 1: the model knows this ─────────────────────
    q1 = "Who was the commander of Apollo 11?"
    r1 = pipeline.generate(q1, max_new_tokens=20, temperature=0.0)
    a1 = r1.text.strip().split("\n")[0]
    correct = "armstrong" in a1.lower()

    print(f"  Q: {q1}")
    print(f"  A: {a1}")
    print(f"     {'Correct.' if correct else 'Wrong.'}")
    print()

    # ── Question 2: the model does NOT know this ─────────────
    q2 = "What did the Apollo 11 crew say about the audio quality during the early part of the mission?"
    r2 = pipeline.generate(q2, max_new_tokens=80, temperature=0.0)
    a2 = r2.text.strip().split("\n")[0]
    has_scratchy = "scratchy" in a2.lower()

    print(f"  Q: {q2}")
    print(f"  A: {a2}")
    wrong_msg = 'Wrong. The real answer is "scratchy."'
    print(f"     {'Correct.' if has_scratchy else wrong_msg}")
    print()

    # ── The point ────────────────────────────────────────────
    print(f"  Same confidence. The model can't tell the difference.")
    print()


if __name__ == "__main__":
    main()
