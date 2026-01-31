#!/usr/bin/env python3
"""Fact Lookup Virtual Expert - Proof of Concept Experiment.

Tests the hypothesis that:
1. Entity fact queries can be detected via prompt patterns
2. External KB lookup (Wikidata) returns correct facts
3. Context injection with proper framing overrides parametric memory

This validates the V1 approach before building the full L13 probe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FactResult:
    """Result from KB lookup."""
    entity: str
    relation: str
    value: str
    source: str
    confidence: float


@dataclass
class ExperimentResult:
    """Result from a single test case."""
    query: str
    detected: bool
    entity_extracted: str | None
    relation_extracted: str | None
    kb_result: FactResult | None
    parametric_answer: str
    injected_answer: str
    override_success: bool
    correct: bool


# =============================================================================
# Prompt-Based Detector (V1 - No Hidden States)
# =============================================================================


class PromptFactDetector:
    """Detect entity fact queries via prompt patterns."""

    PATTERNS = {
        "capital_of": [
            r"(?:what is |what's )?the capital of (\w+)",
            r"capital (?:of|city of) (\w+)",
        ],
        "president_of": [
            r"(?:who is |who's )?the president of (\w+)",
            r"(\w+)'s president",
        ],
        "ceo_of": [
            r"(?:who is |who's )?the ceo of (\w+)",
            r"ceo of (\w+)",
        ],
        "symbol_of": [
            r"(?:what is |what's )?the (?:chemical )?symbol (?:for|of) (\w+)",
            r"(\w+)'s (?:chemical )?symbol",
        ],
        "author_of": [
            r"(?:who )?(?:wrote|is the author of) (.+?)(?:\?|$)",
            r"author of (.+?)(?:\?|$)",
        ],
        "located_in": [
            r"(?:where is |what country is )(.+?) (?:located|in)",
            r"(.+?) is (?:located )?in what country",
        ],
        "born_in": [
            r"(?:when was )(.+?) born",
            r"(.+?)'s (?:birth ?date|birthday)",
        ],
        "population_of": [
            r"(?:what is )?the population of (\w+)",
            r"how many people (?:live in|are in) (\w+)",
        ],
    }

    def detect(self, query: str) -> tuple[bool, str | None, str | None]:
        """Detect if query is an entity fact query.

        Returns:
            (is_fact_query, entity, relation)
        """
        query_lower = query.lower().strip()

        for relation, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if match := re.search(pattern, query_lower, re.I):
                    entity = match.group(1).strip()
                    return True, entity, relation

        return False, None, None


# =============================================================================
# Mock Knowledge Base (for experiment - would be Wikidata in production)
# =============================================================================


class MockKnowledgeBase:
    """Mock KB with known facts for testing override."""

    # Ground truth facts
    FACTS = {
        ("france", "capital_of"): FactResult("France", "capital_of", "Paris", "Wikidata", 1.0),
        ("japan", "capital_of"): FactResult("Japan", "capital_of", "Tokyo", "Wikidata", 1.0),
        ("germany", "capital_of"): FactResult("Germany", "capital_of", "Berlin", "Wikidata", 1.0),
        ("australia", "capital_of"): FactResult("Australia", "capital_of", "Canberra", "Wikidata", 1.0),
        ("brazil", "capital_of"): FactResult("Brazil", "capital_of", "Brasília", "Wikidata", 1.0),
        ("gold", "symbol_of"): FactResult("Gold", "symbol_of", "Au", "Wikidata", 1.0),
        ("silver", "symbol_of"): FactResult("Silver", "symbol_of", "Ag", "Wikidata", 1.0),
        ("iron", "symbol_of"): FactResult("Iron", "symbol_of", "Fe", "Wikidata", 1.0),
        ("apple", "ceo_of"): FactResult("Apple", "ceo_of", "Tim Cook", "Wikidata", 1.0),
        ("microsoft", "ceo_of"): FactResult("Microsoft", "ceo_of", "Satya Nadella", "Wikidata", 1.0),
        ("1984", "author_of"): FactResult("1984", "author_of", "George Orwell", "Wikidata", 1.0),
    }

    # Counterfactual facts for testing override
    COUNTERFACTUALS = {
        ("france", "capital_of"): FactResult("France", "capital_of", "Lyon", "TestDB", 1.0),
        ("japan", "capital_of"): FactResult("Japan", "capital_of", "Osaka", "TestDB", 1.0),
        ("gold", "symbol_of"): FactResult("Gold", "symbol_of", "Gd", "TestDB", 1.0),
    }

    def query(self, entity: str, relation: str, use_counterfactual: bool = False) -> FactResult | None:
        """Query the knowledge base."""
        key = (entity.lower(), relation)
        if use_counterfactual and key in self.COUNTERFACTUALS:
            return self.COUNTERFACTUALS[key]
        return self.FACTS.get(key)


# =============================================================================
# Context Injector
# =============================================================================


class ContextInjector:
    """Format facts for context injection."""

    TEMPLATES = {
        "capital_of": "In this context, {entity}'s capital is {value} (source: {source}).",
        "symbol_of": "In this context, the chemical symbol for {entity} is {value} (source: {source}).",
        "ceo_of": "In this context, the CEO of {entity} is {value} (source: {source}).",
        "author_of": "In this context, {entity} was written by {value} (source: {source}).",
        "president_of": "In this context, the president of {entity} is {value} (source: {source}).",
        "default": "In this context, {entity} {relation} {value} (source: {source}).",
    }

    def format(self, fact: FactResult) -> str:
        """Format fact for injection."""
        template = self.TEMPLATES.get(fact.relation, self.TEMPLATES["default"])
        return template.format(
            entity=fact.entity,
            relation=fact.relation.replace("_", " "),
            value=fact.value,
            source=fact.source,
        )


# =============================================================================
# Main Experiment
# =============================================================================


class FactLookupExperiment:
    """End-to-end test of fact lookup virtual expert."""

    def __init__(self):
        self.detector = PromptFactDetector()
        self.kb = MockKnowledgeBase()
        self.injector = ContextInjector()
        self.model = None
        self.tokenizer = None

    async def setup(self):
        """Load model."""
        from chuk_lazarus.models_v2.loader import load_model

        logger.info("Loading model: openai/gpt-oss-20b")
        loaded = load_model("openai/gpt-oss-20b")
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        logger.info("Model loaded")

    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        """Generate response."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            next_token = mx.argmax(output.logits[0, -1, :])
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    async def run_single(
        self,
        query: str,
        expected_answer: str,
        use_counterfactual: bool = False,
    ) -> ExperimentResult:
        """Run single test case."""
        # Step 1: Detect
        detected, entity, relation = self.detector.detect(query)

        # Step 2: KB Lookup
        kb_result = None
        if detected and entity and relation:
            kb_result = self.kb.query(entity, relation, use_counterfactual)

        # Step 3: Generate without injection (parametric)
        parametric_answer = self.generate(query)

        # Step 4: Generate with injection
        injected_answer = parametric_answer  # Default if no injection
        if kb_result:
            injection = self.injector.format(kb_result)
            injected_prompt = f"{injection}\n\nQuestion: {query}\nAnswer:"
            injected_answer = self.generate(injected_prompt)

        # Step 5: Evaluate
        expected_lower = expected_answer.lower()
        parametric_has_answer = expected_lower in parametric_answer.lower()
        injected_has_answer = expected_lower in injected_answer.lower()

        # For counterfactual: check if injected answer has the counterfactual value
        if use_counterfactual and kb_result:
            override_success = kb_result.value.lower() in injected_answer.lower()
            correct = override_success  # Success = adopted counterfactual
        else:
            override_success = injected_has_answer and not parametric_has_answer
            correct = injected_has_answer

        return ExperimentResult(
            query=query,
            detected=detected,
            entity_extracted=entity,
            relation_extracted=relation,
            kb_result=kb_result,
            parametric_answer=parametric_answer[:100],
            injected_answer=injected_answer[:100],
            override_success=override_success,
            correct=correct,
        )

    async def run(self) -> dict[str, Any]:
        """Run full experiment."""
        await self.setup()

        results = {
            "timestamp": datetime.now().isoformat(),
            "detection_tests": [],
            "override_tests": [],
            "counterfactual_tests": [],
            "summary": {},
        }

        # Test 1: Detection accuracy
        logger.info("=== Testing Detection ===")
        detection_cases = [
            ("What is the capital of France?", True, "france", "capital_of"),
            ("Who is the CEO of Apple?", True, "apple", "ceo_of"),
            ("What's the chemical symbol for gold?", True, "gold", "symbol_of"),
            ("How do I tie a bowline knot?", False, None, None),  # Procedural
            ("What is 2 + 2?", False, None, None),  # Arithmetic
            ("Tell me a joke", False, None, None),  # Creative
            ("The capital of Japan is", True, "japan", "capital_of"),
        ]

        detection_correct = 0
        for query, expected_detect, expected_entity, expected_relation in detection_cases:
            detected, entity, relation = self.detector.detect(query)
            correct = (detected == expected_detect)
            if expected_entity:
                correct = correct and (entity and entity.lower() == expected_entity.lower())
            if expected_relation:
                correct = correct and (relation == expected_relation)

            detection_correct += int(correct)
            results["detection_tests"].append({
                "query": query,
                "expected": (expected_detect, expected_entity, expected_relation),
                "actual": (detected, entity, relation),
                "correct": correct,
            })
            logger.info(f"  {query[:40]:40} | detect={detected} entity={entity} | {'✓' if correct else '✗'}")

        results["summary"]["detection_accuracy"] = detection_correct / len(detection_cases)
        logger.info(f"Detection accuracy: {detection_correct}/{len(detection_cases)}")

        # Test 2: Override with correct facts
        logger.info("\n=== Testing Override (Correct Facts) ===")
        override_cases = [
            ("What is the capital of France?", "Paris"),
            ("What is the capital of Australia?", "Canberra"),
            ("What is the chemical symbol for gold?", "Au"),
            ("Who is the CEO of Microsoft?", "Satya Nadella"),
        ]

        override_correct = 0
        for query, expected in override_cases:
            result = await self.run_single(query, expected, use_counterfactual=False)
            override_correct += int(result.correct)
            results["override_tests"].append({
                "query": query,
                "expected": expected,
                "parametric": result.parametric_answer,
                "injected": result.injected_answer,
                "correct": result.correct,
            })
            logger.info(f"  {query[:40]:40} | correct={result.correct}")

        results["summary"]["override_accuracy"] = override_correct / len(override_cases)
        logger.info(f"Override accuracy: {override_correct}/{len(override_cases)}")

        # Test 3: Counterfactual override (the real test)
        logger.info("\n=== Testing Counterfactual Override ===")
        counterfactual_cases = [
            ("What is the capital of France?", "Lyon"),  # Counterfactual: Lyon instead of Paris
            ("What is the capital of Japan?", "Osaka"),  # Counterfactual: Osaka instead of Tokyo
            ("What is the chemical symbol for gold?", "Gd"),  # Counterfactual: Gd instead of Au
        ]

        counterfactual_success = 0
        for query, counterfactual_value in counterfactual_cases:
            result = await self.run_single(query, counterfactual_value, use_counterfactual=True)
            counterfactual_success += int(result.override_success)
            results["counterfactual_tests"].append({
                "query": query,
                "counterfactual_value": counterfactual_value,
                "parametric": result.parametric_answer,
                "injected": result.injected_answer,
                "override_success": result.override_success,
            })
            status = "✓ OVERRIDE" if result.override_success else "✗ FAILED"
            logger.info(f"  {query[:40]:40} | {status}")
            logger.info(f"    Parametric: {result.parametric_answer[:60]}...")
            logger.info(f"    Injected:   {result.injected_answer[:60]}...")

        results["summary"]["counterfactual_override_rate"] = counterfactual_success / len(counterfactual_cases)
        logger.info(f"\nCounterfactual override rate: {counterfactual_success}/{len(counterfactual_cases)}")

        # Save results
        output_path = Path(__file__).parent / "results" / f"fact_lookup_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")

        return results


async def main():
    experiment = FactLookupExperiment()
    results = await experiment.run()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Detection Accuracy:        {results['summary']['detection_accuracy']:.1%}")
    print(f"Override Accuracy:         {results['summary']['override_accuracy']:.1%}")
    print(f"Counterfactual Override:   {results['summary']['counterfactual_override_rate']:.1%}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
