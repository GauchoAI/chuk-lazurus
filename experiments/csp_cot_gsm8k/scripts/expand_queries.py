#!/usr/bin/env python3
"""Generate LLM-expanded training data using GPT-OSS via Ollama.

Uses SchemaGenerator's expander callback to rewrite template queries
with diverse natural language while keeping traces and answers fixed.

Supports three expansion strategies (randomly selected per example):
  INVENT     — LLM invents a fresh word problem from the math spec alone
  PARAPHRASE — LLM rewrites the template with scenario/order changes
  TEMPLATE   — Uses the original template as-is (baseline)

Usage:
    # Generate 1500 expanded examples
    python scripts/expand_queries.py --n 1500 --output data/expanded_1500.jsonl

    # Use a different model
    python scripts/expand_queries.py --n 500 --model qwen3:8b

    # Custom strategy weights (INVENT, PARAPHRASE, TEMPLATE)
    python scripts/expand_queries.py --n 500 --strategy-weights 60,20,20
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import random
import re
import sys
import time
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_virtual_expert_arithmetic.generators import (
    ALL_SCHEMAS,
    TraceGenerator,
)
from chuk_virtual_expert_arithmetic.generators.schema_generator import SchemaGenerator
from chuk_virtual_expert_arithmetic.types import WORD_NUMBERS


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class ExpandStrategy(enum.Enum):
    INVENT = "invent"
    PARAPHRASE = "paraphrase"
    TEMPLATE = "template"


# Default weights: INVENT 50%, PARAPHRASE 30%, TEMPLATE 20%
STRATEGY_WEIGHTS: dict[ExpandStrategy, int] = {
    ExpandStrategy.INVENT: 50,
    ExpandStrategy.PARAPHRASE: 30,
    ExpandStrategy.TEMPLATE: 20,
}


# ---------------------------------------------------------------------------
# Schema descriptions for INVENT strategy
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    # Sequential
    "price_chain": "Compute a total price by adding several item costs together.",
    "subtract_chain": "Start with a quantity and subtract several amounts in sequence.",
    "multiply_add": "Multiply two numbers then add another value.",
    "divide_multiply": "Divide a number then multiply the result.",
    "work_rate": "Calculate work output from a rate and time.",
    "combined_rate": "Combine two rates to find a total output.",
    "div_then_add": "Divide a number then add another value to the result.",
    # Interleaved
    "interleaved_mul_mul": "Two separate multiplications whose results are combined.",
    "parallel_merge": "Two parallel computations merged into a final answer.",
    "chained_mul_sum": "Chain of multiplications followed by a sum.",
    "consume_then_sell": "Consume part of a quantity, then sell or give away more.",
    "rate_comparison_total": "Compare two rates and find their combined total.",
    # Long chain
    "long_expense_chain": "Track expenses through a long chain of additions and subtractions.",
    # Division chains
    "sub_sub_div_div": "Subtract twice, then divide twice.",
    "div_chain": "Perform a chain of successive divisions.",
    # Gap-closing
    "half_twice": "One quantity is half (or twice) another; find the total or difference.",
    "fraction_simple": "Apply a simple fraction to a quantity.",
    "shopping_spree": "Buy multiple items at different prices; find total or change.",
    "material_half": "Use a quantity and half that much more material.",
    "material_twice": "Use a quantity and twice that much more material.",
    "decimal_rate_week": "Apply a decimal rate over days to get a weekly total.",
    "decimal_multiply": "Multiply a decimal rate by a whole number.",
    "weekly_sprints": "Compute total distance from sprints per day over a week.",
    "total_minus_given": "Need a total, already have some parts; find the remainder.",
    "distance_rate": "Compute distance from speed and time.",
    "distance_round_trip": "Calculate a round-trip distance.",
    "school_supplies": "Calculate the cost of school supplies.",
    "classroom_groups": "Divide students into groups for a classroom activity.",
    "growth_doubled": "A quantity doubles or triples; find the new amount.",
    "average_three": "Find the average of three values.",
    "remaining_capacity": "Find how many more units fit given a limit and current amount.",
    "multi_item_cost": "Buy different quantities of different items; find total cost.",
    "ratio_split": "Split a total according to a ratio.",
    "time_from_distance": "Find travel time from distance and speed.",
    "twice_relationship": "One quantity is twice another; find a total or difference.",
    "two_period_sum": "Sum outputs across two periods with different rates.",
    "recover_then_multiply": "Recover a consumed amount, then multiply.",
    "nested_groups": "Multiply groups × subgroups × items per subgroup.",
    # Entity tracking
    "entity_simple_transfer": "Transfer items between two people; track who has what.",
    "entity_consume_sequence": "A person consumes items in a sequence; track remaining.",
    "entity_consume_multiply": "Consume some, then multiply the remainder.",
    "entity_bidirectional": "Two-way transfers between people.",
    "entity_find_lose": "Someone finds and then loses items; track the net change.",
    # Rate / equation
    "rate_distance": "Use rate × time to compute a distance.",
    "rate_earning": "Use hourly rate × hours to compute earnings.",
    # Comparison
    "comparison_times_more": "One person has N times more than another; find totals.",
    "comparison_sum_diff": "Compare two quantities using their sum and difference.",
    "comparison_more_less": "One has more, another has less; find amounts.",
    "comparison_half": "One quantity is half of another; compare or combine.",
    # Percentage
    "percent_off": "Apply a percentage discount to a price.",
    "percent_increase": "Increase a value by a given percentage.",
    "percent_tip": "Calculate a tip as a percentage of a bill.",
    "percent_simple": "Find a percentage of a number.",
}


def _auto_describe_schema(schema_name: str, variables: dict) -> str:
    """Fallback: generate a description from schema name and variable names."""
    readable = schema_name.replace("_", " ")
    var_names = ", ".join(variables.keys())
    return f"Perform a {readable} calculation using: {var_names}."


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def choose_strategy(rng: random.Random, weights: dict[ExpandStrategy, int]) -> ExpandStrategy:
    """Weighted random pick of an expansion strategy."""
    strategies = list(weights.keys())
    w = [weights[s] for s in strategies]
    return rng.choices(strategies, weights=w, k=1)[0]


# ---------------------------------------------------------------------------
# Math description builder (for INVENT strategy)
# ---------------------------------------------------------------------------

def describe_math_from_variables(schema_name: str, variables: dict, answer: float | int) -> str:
    """Build a human-readable math description for the INVENT prompt."""
    desc = SCHEMA_DESCRIPTIONS.get(schema_name) or _auto_describe_schema(schema_name, variables)
    var_lines = []
    for name, val in variables.items():
        display = int(val) if isinstance(val, float) and val == int(val) else val
        var_lines.append(f"  {name} = {display}")
    var_block = "\n".join(var_lines)
    answer_display = int(answer) if isinstance(answer, float) and answer == int(answer) else answer
    return (
        f"Math type: {desc}\n"
        f"Variables:\n{var_block}\n"
        f"Correct answer: {answer_display}"
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_invent_prompt(schema_name: str, variables: dict, answer: float | int) -> str:
    """Build an INVENT prompt — LLM creates a fresh word problem from the math spec."""
    math_desc = describe_math_from_variables(schema_name, variables, answer)
    values_list = ", ".join(
        str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
        for v in variables.values()
    )
    return (
        f"Create a NEW grade-school math word problem that uses exactly "
        f"these numbers and produces the given answer.\n\n"
        f"{math_desc}\n\n"
        f"Requirements:\n"
        f"- Use ALL these numbers exactly: {values_list}\n"
        f"- The answer must be {int(answer) if isinstance(answer, float) and answer == int(answer) else answer}\n"
        f"- Write 3-5 natural sentences, 35-50 words\n"
        f"- Add a brief setup sentence with context before the math\n"
        f"- Scatter the numbers — do NOT present them in computation order\n"
        f"- Use a creative, original scenario (shopping, cooking, sports, travel, school, etc.)\n"
        f"- End with a clear question\n\n"
        f"Problem:"
    )


def build_paraphrase_prompt(query: str, variables: dict, answer: float | int) -> str:
    """Build a PARAPHRASE prompt — LLM rewrites the template with changes."""
    values_list = ", ".join(
        str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
        for v in variables.values()
    )
    return (
        f"Rewrite this math word problem with major changes.\n\n"
        f"Original: {query}\n\n"
        f"Numbers that must appear: {values_list}\n"
        f"Correct answer: {int(answer) if isinstance(answer, float) and answer == int(answer) else answer}\n\n"
        f"Rules:\n"
        f"- Use ALL the same numbers (values must be identical)\n"
        f"- Same mathematical operations in the same order\n"
        f"- CHANGE the scenario completely (different names, items, setting)\n"
        f"- Rearrange the information — put some key numbers later in the problem\n"
        f"- Add narrative context (why the character is doing this)\n"
        f"- Write 3-5 natural sentences, 35-50 words\n"
        f"- End with a clear question\n\n"
        f"Rewritten:"
    )


# ---------------------------------------------------------------------------
# Output cleaning and validation
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(
    r"^\s*(problem|rewritten|rewrite|answer|solution|output|result|question)\s*[:：]\s*",
    re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(r"^```[a-z]*\n?|```\s*$|^\*\*.*?\*\*\s*", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_llm_output(text: str) -> str:
    """Strip markdown, labels, thinking tags, and trailing incomplete sentences."""
    # Remove <think>...</think> blocks
    text = _THINK_RE.sub("", text)
    # Remove markdown fences and bold markers
    text = _MARKDOWN_RE.sub("", text)
    # Remove leading labels like "Problem:" or "Rewritten:"
    text = _LABEL_RE.sub("", text)
    # Strip whitespace
    text = text.strip()
    # Remove trailing incomplete sentence (no terminal punctuation)
    sentences = re.split(r"(?<=[.?!])\s+", text)
    if sentences and not re.search(r"[.?!]$", sentences[-1]):
        sentences = sentences[:-1]
    if sentences:
        text = " ".join(sentences)
    return text.strip()


def is_complete_sentence(text: str) -> bool:
    """Check that text ends with sentence-terminal punctuation."""
    return bool(re.search(r"[.?!]\s*$", text))


def is_too_short(text: str, min_words: int = 25) -> bool:
    """Check if text has fewer than min_words words."""
    return len(text.split()) < min_words


# ---------------------------------------------------------------------------
# Word number injection (post-processing)
# ---------------------------------------------------------------------------

def inject_word_numbers(text: str, rng: random.Random, prob: float = 0.30) -> str:
    """Randomly convert eligible digit strings to English words.

    Skips numbers preceded by '$' (prices stay as digits).
    Only converts numbers present in the WORD_NUMBERS mapping (1-25, 30, 40, 50).
    """
    def _replace(match: re.Match) -> str:
        prefix = match.group(1)  # may be '$' or empty
        num_str = match.group(2)
        # Never convert prices
        if prefix == "$":
            return match.group(0)
        try:
            num = int(num_str)
        except ValueError:
            return match.group(0)
        if num in WORD_NUMBERS and rng.random() < prob:
            return prefix + WORD_NUMBERS[num]
        return match.group(0)

    # Match optional $ prefix followed by a whole number (not part of a larger number)
    return re.sub(r"(\$?)(?<!\d)\b(\d+)\b(?!\d)", _replace, text)


# ---------------------------------------------------------------------------
# Expander factory
# ---------------------------------------------------------------------------

def make_ollama_expander(
    model: str = "gpt-oss:20b",
    base_url: str = "http://localhost:11434/v1",
    temperature: float = 0.8,
    strategy_weights: dict[ExpandStrategy, int] | None = None,
    seed: int = 42,
):
    """Create an async expander callback with multi-strategy support."""

    client = openai.AsyncOpenAI(base_url=base_url, api_key="ollama")
    weights = strategy_weights or STRATEGY_WEIGHTS
    rng = random.Random(seed)

    # Stats tracking
    stats = {
        "strategy_counts": {s.value: 0 for s in ExpandStrategy},
        "retries": 0,
        "fallbacks": 0,
    }

    def get_stats() -> dict:
        return stats

    async def expander(query: str, context: dict) -> str:
        schema_name = context.get("schema", "")
        variables = context.get("variables", {})
        answer = context.get("answer", 0)

        strategy = choose_strategy(rng, weights)
        stats["strategy_counts"][strategy.value] += 1

        # TEMPLATE strategy — return original query with word number injection
        if strategy == ExpandStrategy.TEMPLATE:
            return inject_word_numbers(query, rng)

        # Build strategy-specific prompt and system message
        if strategy == ExpandStrategy.INVENT:
            prompt = build_invent_prompt(schema_name, variables, answer)
            system_msg = (
                "You create original math word problems. Output ONLY the "
                "problem text, nothing else. No explanations, no reasoning, "
                "no markdown, no labels."
            )
        else:  # PARAPHRASE
            prompt = build_paraphrase_prompt(query, variables, answer)
            system_msg = (
                "You rewrite math word problems with creative changes. "
                "Output ONLY the rewritten problem, nothing else. "
                "No explanations, no reasoning, no markdown, no labels."
            )

        # Extract expected values for validation
        values = [
            str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
            for v in variables.values()
        ]

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                temp = temperature + (attempt * 0.1)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=512,
                )
                result = response.choices[0].message.content.strip()
                result = clean_llm_output(result)

                # Validation 1: number preservation (≥ half of init values present)
                found = sum(1 for v in values if v in result)
                if found < len(values) / 2:
                    stats["retries"] += 1
                    continue

                # Validation 2: sentence completeness
                if not is_complete_sentence(result):
                    stats["retries"] += 1
                    continue

                # Validation 3: minimum length
                if is_too_short(result):
                    stats["retries"] += 1
                    continue

                # All checks passed — apply word number injection
                return inject_word_numbers(result, rng)

            except Exception as e:
                print(f"  [expander error: {e}] attempt {attempt+1}/{max_attempts}")
                stats["retries"] += 1

        # All retries exhausted — fall back to template with word number injection
        stats["fallbacks"] += 1
        return inject_word_numbers(query, rng)

    expander.get_stats = get_stats  # type: ignore[attr-defined]
    return expander


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

async def generate_expanded_data(
    n: int,
    model: str,
    seed: int,
    output_path: Path,
    strategy_weights: dict[ExpandStrategy, int] | None = None,
) -> list[dict]:
    """Generate n expanded training examples, one at a time with streaming output."""

    gen = SchemaGenerator(
        perturbation_level=0.0,  # No perturbation — LLM handles diversity
        word_number_prob=0.0,    # No word numbers — post-processing handles it
        gsm8k_style_prob=0.0,    # No style switching — LLM rewrites
        messy_vocab_prob=0.0,    # Clean template for LLM reference
        seed=seed,
    )

    expander = make_ollama_expander(
        model=model,
        strategy_weights=strategy_weights,
        seed=seed,
    )
    rng = random.Random(seed)

    print(f"Generating {n} expanded examples using {model}...")
    print(f"  Schemas: {len(ALL_SCHEMAS)} available")
    print(f"  Output: {output_path}")
    weights = strategy_weights or STRATEGY_WEIGHTS
    total_w = sum(weights.values())
    weight_str = ", ".join(
        f"{s.value}={w/total_w:.0%}" for s, w in weights.items()
    )
    print(f"  Strategy weights: {weight_str}")
    print()

    result = []
    start = time.time()

    # Stream results to JSONL as we go (resume-friendly)
    with open(output_path, "w") as f:
        for i in range(n):
            schema_name = rng.choice(ALL_SCHEMAS)
            ex = await gen.generate_with_expander(schema_name, expander)

            trace = [
                {k: v for k, v in step.model_dump(mode="json").items() if v is not None}
                for step in ex.trace
            ]
            record = {
                "expert": ex.expert,
                "query": ex.query,
                "trace": trace,
                "answer": ex.answer,
            }
            result.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()

            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {schema_name:30s} | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - start
    print(f"\nGenerated {len(result)} examples in {elapsed:.1f}s ({elapsed/len(result):.2f}s/ex)")

    # Print strategy stats
    exp_stats = expander.get_stats()  # type: ignore[attr-defined]
    print("\nStrategy distribution:")
    for strategy_name, count in exp_stats["strategy_counts"].items():
        pct = count / n * 100 if n > 0 else 0
        print(f"  {strategy_name:12s}: {count:4d} ({pct:.1f}%)")
    print(f"  {'retries':12s}: {exp_stats['retries']}")
    print(f"  {'fallbacks':12s}: {exp_stats['fallbacks']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-expanded training data")
    parser.add_argument("--n", type=int, default=1500, help="Number of examples")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model name")
    parser.add_argument("--output", default="data/expanded.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--preview", type=int, default=5, help="Print N examples (0=none)")
    parser.add_argument(
        "--strategy-weights",
        default=None,
        help="Comma-separated INVENT,PARAPHRASE,TEMPLATE weights (e.g. '50,30,20')",
    )
    args = parser.parse_args()

    # Parse strategy weights
    strategy_weights = None
    if args.strategy_weights:
        parts = [int(x.strip()) for x in args.strategy_weights.split(",")]
        if len(parts) != 3:
            parser.error("--strategy-weights must have exactly 3 comma-separated values")
        strategy_weights = {
            ExpandStrategy.INVENT: parts[0],
            ExpandStrategy.PARAPHRASE: parts[1],
            ExpandStrategy.TEMPLATE: parts[2],
        }

    # Resolve output path relative to experiment dir
    exp_dir = Path(__file__).parent.parent
    output_path = exp_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate (streams to JSONL as it goes)
    examples = asyncio.run(
        generate_expanded_data(args.n, args.model, args.seed, output_path, strategy_weights)
    )

    # Preview
    if args.preview > 0:
        print(f"\n{'='*70}")
        print(f"  SAMPLE EXPANDED QUERIES")
        print(f"{'='*70}")
        for ex in random.Random(args.seed).sample(examples, min(args.preview, len(examples))):
            print(f"\n  expert: {ex['expert']}")
            print(f"  query:  {ex['query']}")
            print(f"  answer: {ex['answer']}")
            print(f"  steps:  {len(ex['trace'])}")

    print(f"\nWrote {len(examples)} examples to {output_path}")

    # Expert distribution
    expert_counts: dict[str, int] = {}
    for ex in examples:
        expert_counts[ex["expert"]] = expert_counts.get(ex["expert"], 0) + 1
    print("\nExpert distribution:")
    for expert, count in sorted(expert_counts.items()):
        print(f"  {expert}: {count}")


if __name__ == "__main__":
    main()
