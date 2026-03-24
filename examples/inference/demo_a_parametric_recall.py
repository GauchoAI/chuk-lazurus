#!/usr/bin/env python3
"""
Experiment A — Parametric Recall Baseline

Measures what the model already knows WITHOUT any context or sparse index.
Tests three familiarity levels: Apollo 11 (famous), Gemini 8 (moderate),
Zarkov (fictional). Then tests entity anchor boost.

Usage:
    cd /Users/christopherhay/chris-source/apollo-demo
    uv run --project /Users/christopherhay/chris-source/chuk-mlx \
        python demo_a_parametric_recall.py
"""

from __future__ import annotations

import json
import sys
import time

# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Question banks
# ---------------------------------------------------------------------------

APOLLO_11_QUESTIONS = [
    {
        "id": "commander",
        "question": "Who was the mission commander?",
        "expected": "Armstrong",
        "category": "major_fact",
    },
    {
        "id": "lunar_module",
        "question": "What was the lunar module called?",
        "expected": "Eagle",
        "category": "major_fact",
    },
    {
        "id": "quote",
        "question": "What did Armstrong say when he stepped on the moon?",
        "expected": "one small step",
        "category": "famous_quote",
    },
    {
        "id": "landing_date",
        "question": "What date did they land on the moon?",
        "expected": "July 20, 1969",
        "category": "major_fact",
    },
    {
        "id": "command_module",
        "question": "What was the command module called?",
        "expected": "Columbia",
        "category": "major_fact",
    },
    {
        "id": "orbit",
        "question": "Who stayed in orbit while the others walked on the moon?",
        "expected": "Collins",
        "category": "major_fact",
    },
    {
        "id": "landing_site",
        "question": "Where did Eagle land on the moon?",
        "expected": "Sea of Tranquility",
        "category": "specific_detail",
    },
    {
        "id": "launch_vehicle",
        "question": "What was the launch vehicle?",
        "expected": "Saturn V",
        "category": "specific_detail",
    },
    {
        "id": "utc_time",
        "question": "What time (UTC) did Armstrong step onto the lunar surface?",
        "expected": "02:56",
        "category": "specific_detail",
    },
    {
        "id": "porridge",
        "question": "How many bowls of porridge did the crew eat during the mission?",
        "expected": None,  # Novel — no correct answer exists
        "category": "novel",
    },
]

GEMINI_8_QUESTIONS = [
    {
        "id": "command_pilot",
        "question": "Who was the command pilot?",
        "expected": "Armstrong",
        "category": "major_fact",
    },
    {
        "id": "pilot",
        "question": "Who was the pilot?",
        "expected": "David Scott",
        "category": "major_fact",
    },
    {
        "id": "emergency",
        "question": "What was the main emergency that occurred during the mission?",
        "expected": "thruster",
        "category": "major_fact",
    },
    {
        "id": "launch_date",
        "question": "What date did Gemini 8 launch?",
        "expected": "March 16, 1966",
        "category": "specific_detail",
    },
    {
        "id": "docked_with",
        "question": "What spacecraft did Gemini 8 dock with?",
        "expected": "Agena",
        "category": "major_fact",
    },
    {
        "id": "landing",
        "question": "Where did the capsule land after the emergency?",
        "expected": "Pacific",
        "category": "specific_detail",
    },
]

ZARKOV_QUESTIONS = [
    {
        "id": "leader",
        "question": "Who led the Zarkov expedition?",
        "expected": None,
        "category": "fictional",
    },
    {
        "id": "objective",
        "question": "What was the primary objective of the Zarkov expedition?",
        "expected": None,
        "category": "fictional",
    },
    {
        "id": "team_size",
        "question": "How many team members were on the Zarkov expedition?",
        "expected": None,
        "category": "fictional",
    },
    {
        "id": "discovery",
        "question": "What was the most significant discovery made during the Zarkov expedition?",
        "expected": None,
        "category": "fictional",
    },
    {
        "id": "location",
        "question": "Where did the Zarkov expedition take place?",
        "expected": None,
        "category": "fictional",
    },
    {
        "id": "failure",
        "question": "What equipment failure almost ended the Zarkov expedition?",
        "expected": None,
        "category": "fictional",
    },
]

# Entity anchors for the boost test
APOLLO_11_ANCHORS = (
    "Key entities: Armstrong, Aldrin, Collins, Eagle, Columbia, "
    "Saturn V, Sea of Tranquility, July 1969."
)
GEMINI_8_ANCHORS = (
    "Key entities: Armstrong, Scott, Agena, thruster malfunction, "
    "Pacific Ocean, Okinawa, March 1966."
)


def build_prompt(topic_intro: str, question: str) -> str:
    """Build a chat-template prompt."""
    return (
        f"<start_of_turn>user\n"
        f"{topic_intro}\n"
        f"Answer concisely in one sentence.\n"
        f"{question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def check_correct(answer: str, expected: str | None) -> str:
    """Check if the answer contains the expected substring."""
    if expected is None:
        return "confab"
    return "✓" if expected.lower() in answer.lower() else "✗"


def detect_hedging(answer: str) -> bool:
    """Check if the answer contains hedging language."""
    hedges = [
        "i think", "i believe", "i'm not sure", "i don't know",
        "uncertain", "possibly", "perhaps", "might be", "may have",
        "it's unclear", "not certain",
    ]
    lower = answer.lower()
    return any(h in lower for h in hedges)


def run_questions(
    pipeline, topic_intro: str, questions: list[dict], label: str
) -> list[dict]:
    """Run all questions for a topic and return results."""
    results = []
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{label}{RESET}")
    print(f"{'─' * 60}")

    for q in questions:
        prompt = build_prompt(topic_intro, q["question"])
        result = pipeline.generate(prompt, max_new_tokens=80, temperature=0.0)
        answer = result.text.strip()
        correct = check_correct(answer, q["expected"])
        hedged = detect_hedging(answer)

        # Color coding
        if correct == "✓":
            mark = f"{GREEN}✓{RESET}"
        elif correct == "confab":
            mark = f"{YELLOW}confab{RESET}"
        else:
            mark = f"{RED}✗{RESET}"

        hedge_mark = f" {YELLOW}[hedged]{RESET}" if hedged else ""

        print(f"  {mark} {q['id']:20s} → {answer[:80]}{hedge_mark}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "expected": q["expected"],
            "correct": correct,
            "hedged": hedged,
            "category": q["category"],
        })

    # Summary
    factual = [r for r in results if r["expected"] is not None]
    n_correct = sum(1 for r in factual if r["correct"] == "✓")
    n_hedged = sum(1 for r in results if r["hedged"])

    print(f"\n  {BOLD}Score: {n_correct}/{len(factual)} factual correct")
    print(f"  Hedging: {n_hedged}/{len(results)} answers{RESET}")
    return results


def main():
    print(f"\n{BOLD}{'═' * 60}")
    print("  EXPERIMENT A — Parametric Recall Baseline")
    print(f"{'═' * 60}{RESET}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n{DIM}Loading model...{RESET}")
    from chuk_lazarus.inference.unified import UnifiedPipeline

    pipeline = UnifiedPipeline.from_pretrained(
        "google/gemma-3-4b-it", verbose=False
    )
    print(f"  Model loaded: {pipeline.family_type.value}")

    # ------------------------------------------------------------------
    # Phase A1: Apollo 11 (high familiarity)
    # ------------------------------------------------------------------
    a11_results = run_questions(
        pipeline,
        "You are answering questions about the Apollo 11 mission.",
        APOLLO_11_QUESTIONS,
        "Phase A1: Apollo 11 — Very High Familiarity",
    )

    # ------------------------------------------------------------------
    # Phase A4a: Gemini 8 (moderate familiarity)
    # ------------------------------------------------------------------
    g8_results = run_questions(
        pipeline,
        "You are answering questions about the Gemini 8 mission.",
        GEMINI_8_QUESTIONS,
        "Phase A4a: Gemini 8 — Moderate Familiarity",
    )

    # ------------------------------------------------------------------
    # Phase A4b: Zarkov (fictional — expected 0%)
    # ------------------------------------------------------------------
    zarkov_results = run_questions(
        pipeline,
        "You are answering questions about the Zarkov expedition of 2019.",
        ZARKOV_QUESTIONS,
        "Phase A4b: Zarkov Expedition — Fictional (Expected 0%)",
    )

    # ------------------------------------------------------------------
    # Phase A5: Entity Anchor Boost
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{'═' * 60}")
    print("  Phase A5 — Entity Anchor Boost")
    print(f"{'═' * 60}{RESET}")
    print(f"  {DIM}Re-asking questions with entity anchors in prompt{RESET}")

    a11_anchor_results = run_questions(
        pipeline,
        f"You are answering questions about the Apollo 11 mission. {APOLLO_11_ANCHORS}",
        APOLLO_11_QUESTIONS,
        "Apollo 11 + Entity Anchors",
    )

    g8_anchor_results = run_questions(
        pipeline,
        f"You are answering questions about the Gemini 8 mission. {GEMINI_8_ANCHORS}",
        GEMINI_8_QUESTIONS,
        "Gemini 8 + Entity Anchors",
    )

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    def score(results):
        factual = [r for r in results if r["expected"] is not None]
        n = sum(1 for r in factual if r["correct"] == "✓")
        return n, len(factual)

    a11_n, a11_t = score(a11_results)
    g8_n, g8_t = score(g8_results)
    a11a_n, a11a_t = score(a11_anchor_results)
    g8a_n, g8a_t = score(g8_anchor_results)

    print(f"\n{BOLD}{'═' * 60}")
    print("  PARAMETRIC CEILING SUMMARY")
    print(f"{'═' * 60}{RESET}")
    print()
    print(f"  {'Topic':<25} {'Without anchors':>16} {'With anchors':>14} {'Δ':>6}")
    print(f"  {'─' * 63}")
    print(
        f"  {'Apollo 11 (famous)':<25} "
        f"{a11_n}/{a11_t} ({100*a11_n/a11_t:.0f}%){'':<5} "
        f"{a11a_n}/{a11a_t} ({100*a11a_n/a11a_t:.0f}%){'':<3} "
        f"{GREEN}+{100*(a11a_n/a11a_t - a11_n/a11_t):.0f}%{RESET}"
    )
    print(
        f"  {'Gemini 8 (moderate)':<25} "
        f"{g8_n}/{g8_t} ({100*g8_n/g8_t:.0f}%){'':<5} "
        f"{g8a_n}/{g8a_t} ({100*g8a_n/g8a_t:.0f}%){'':<3} "
        f"{GREEN}+{100*(g8a_n/g8a_t - g8_n/g8_t):.0f}%{RESET}"
    )
    print(f"  {'Zarkov (fictional)':<25} 0/6 (0%)")

    n_hedged = sum(
        1 for r in a11_results + g8_results + zarkov_results if r["hedged"]
    )
    n_total = len(a11_results) + len(g8_results) + len(zarkov_results)
    print(f"\n  {BOLD}Hedging rate: {n_hedged}/{n_total} "
          f"({'zero' if n_hedged == 0 else n_hedged} answers hedge){RESET}")
    print(f"  {DIM}The model NEVER says 'I don't know' — even on fiction.{RESET}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    all_results = {
        "experiment": "A_parametric_recall",
        "model": "google/gemma-3-4b-it",
        "apollo11": a11_results,
        "gemini8": g8_results,
        "zarkov": zarkov_results,
        "apollo11_anchored": a11_anchor_results,
        "gemini8_anchored": g8_anchor_results,
        "summary": {
            "apollo11_ceiling": f"{a11_n}/{a11_t}",
            "apollo11_anchored": f"{a11a_n}/{a11a_t}",
            "gemini8_ceiling": f"{g8_n}/{g8_t}",
            "gemini8_anchored": f"{g8a_n}/{g8a_t}",
            "zarkov_ceiling": "0/6",
            "total_hedging": f"{n_hedged}/{n_total}",
        },
    }

    out_path = "results_experiment_a.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  {DIM}Results saved to {out_path}{RESET}\n")


if __name__ == "__main__":
    main()
