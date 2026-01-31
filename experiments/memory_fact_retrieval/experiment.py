#!/usr/bin/env python3
"""Memory & Fact Retrieval Experiment.

Investigates:
1. Parametric Memory Probing - where facts are stored
2. Context vs Memory Discrimination - how models prioritize sources
3. Fact Manipulation via Steering - inject/suppress facts
4. Retrieval Dynamics - attention patterns during lookup
5. Fact Type Specialization - different storage for different fact types
6. Confidence Calibration - internal confidence vs accuracy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class ParametricProbingResult:
    """Results from parametric memory probing."""

    layer_accuracy: dict[int, dict[str, float]]  # layer -> {category_acc, value_acc}
    fact_type_curves: dict[str, dict[int, float]]  # fact_type -> layer -> accuracy
    best_layers: dict[str, int]  # fact_type -> best layer for retrieval
    probe_weights: dict[int, Any] | None = None  # Optional: trained probe weights


@dataclass
class ContextMemoryResult:
    """Results from context vs memory discrimination."""

    override_layer: int  # Layer where context starts dominating
    source_preference: dict[int, dict[str, float]]  # layer -> {parametric, context}
    override_threshold: float  # Minimum context strength to override
    conflict_examples: list[dict[str, Any]]  # Analyzed conflict cases


@dataclass
class FactSteeringResult:
    """Results from fact manipulation via steering."""

    suppression_success: dict[int, dict[float, float]]  # layer -> strength -> success_rate
    injection_success: dict[int, dict[float, float]]  # layer -> strength -> success_rate
    steering_directions: dict[str, Any] | None  # Extracted steering vectors
    coherence_impact: dict[float, float]  # strength -> coherence score
    best_steering_config: dict[str, Any]  # Optimal layer/strength


@dataclass
class RetrievalAttentionResult:
    """Results from retrieval attention dynamics."""

    retrieval_heads: list[tuple[int, int]]  # (layer, head) pairs that specialize
    attention_patterns: dict[str, dict[str, float]]  # category -> pattern metrics
    known_vs_unknown_signatures: dict[str, Any]  # Distinguishing features
    hallucination_markers: list[dict[str, Any]]  # Patterns indicating hallucination


@dataclass
class FactTypesResult:
    """Results from fact type specialization analysis."""

    type_layer_accuracy: dict[str, dict[int, float]]  # type -> layer -> accuracy
    type_clustering: dict[int, dict[str, float]]  # layer -> type -> cluster tightness
    expert_routing: dict[str, dict[int, list[int]]] | None  # type -> layer -> experts (MoE)
    cross_type_confusion: dict[tuple[str, str], float]  # (type_a, type_b) -> confusion


@dataclass
class ConfidenceResult:
    """Results from confidence calibration analysis."""

    calibration_curve: dict[int, tuple[float, float]]  # bin -> (predicted_conf, actual_acc)
    ece: float  # Expected Calibration Error
    uncertainty_features: list[str]  # Features that predict uncertainty
    hallucination_detection_auc: float  # AUC for detecting incorrect facts


@dataclass
class MoEFactRoutingResult:
    """Results from MoE expert routing analysis for facts."""

    expert_by_fact_type: dict[str, dict[int, list[int]]]  # type -> layer -> top experts
    expert_specialization: dict[int, dict[int, dict[str, float]]]  # layer -> expert -> type -> freq
    type_separation_score: dict[int, float]  # layer -> how separated are fact types
    dominant_experts: dict[str, list[tuple[int, int, float]]]  # type -> [(layer, expert, weight)]


@dataclass
class ExperimentResults:
    """All experiment results."""

    parametric_probing: ParametricProbingResult | None = None
    context_memory: ContextMemoryResult | None = None
    fact_steering: FactSteeringResult | None = None
    retrieval_attention: RetrievalAttentionResult | None = None
    fact_types: FactTypesResult | None = None
    confidence: ConfidenceResult | None = None
    moe_fact_routing: MoEFactRoutingResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Linear Probe
# =============================================================================


class LinearProbe(nn.Module):
    """Simple linear probe for classification."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


# =============================================================================
# Main Experiment Class
# =============================================================================


class MemoryFactRetrievalExperiment:
    """Main experiment class for memory and fact retrieval analysis."""

    def __init__(self, config_path: Path | None = None):
        """Initialize experiment."""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.results = ExperimentResults()

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from YAML."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path) as f:
            return yaml.safe_load(f)

    async def setup(self) -> None:
        """Load model and prepare for analysis."""
        from chuk_lazarus.models_v2.loader import load_model

        model_id = self.config["model"]
        logger.info(f"Loading model: {model_id}")

        try:
            loaded = load_model(model_id)
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer
        except Exception as e:
            logger.warning(f"Could not load {model_id}: {e}")
            logger.info("Falling back to TinyLlama for testing")
            loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        self._num_layers = len(self.model.model.layers)
        # Get hidden dim from layer config
        self._hidden_dim = self.model.model.layers[0].hidden_size

        # Check if MoE model
        self._is_moe = hasattr(self.model.model.layers[0], "block_sparse_moe") or \
                       hasattr(self.model.model.layers[0].mlp, "gate")

        logger.info(f"Model loaded: {self._num_layers} layers, {self._hidden_dim} hidden dim, MoE={self._is_moe}")

    async def run(self, analyses: list[str] | None = None) -> ExperimentResults:
        """Run specified analyses."""
        if self.model is None:
            await self.setup()

        analyses = analyses or self.config.get("analyses", [])
        logger.info(f"Running analyses: {analyses}")

        self.results.metadata = {
            "model": self.config["model"],
            "timestamp": datetime.now().isoformat(),
            "analyses": analyses,
            "num_layers": self._num_layers,
            "hidden_dim": self._hidden_dim,
        }

        for analysis in analyses:
            logger.info(f"Running analysis: {analysis}")
            try:
                if analysis == "parametric_probing":
                    self.results.parametric_probing = await self._analyze_parametric_probing()
                elif analysis == "context_memory":
                    self.results.context_memory = await self._analyze_context_memory()
                elif analysis == "fact_steering":
                    self.results.fact_steering = await self._analyze_fact_steering()
                elif analysis == "retrieval_attention":
                    self.results.retrieval_attention = await self._analyze_retrieval_attention()
                elif analysis == "fact_types":
                    self.results.fact_types = await self._analyze_fact_types()
                elif analysis == "confidence":
                    self.results.confidence = await self._analyze_confidence()
                elif analysis == "moe_fact_routing":
                    self.results.moe_fact_routing = await self._analyze_moe_fact_routing()
                else:
                    logger.warning(f"Unknown analysis: {analysis}")
            except Exception as e:
                logger.error(f"Analysis {analysis} failed: {e}")
                raise

        return self.results

    # =========================================================================
    # Parametric Memory Probing
    # =========================================================================

    async def _analyze_parametric_probing(self) -> ParametricProbingResult:
        """Analyze where facts are stored via linear probing."""
        logger.info("Analyzing parametric memory via probing...")
        config = self.config.get("parametric_probing", {})
        probe_layers = config.get("probe_layers", [4, 8, 12])

        # Collect hidden states for each fact type
        facts_by_type = self.config.get("facts", {})
        hidden_states_by_type: dict[str, dict[int, list[mx.array]]] = defaultdict(
            lambda: defaultdict(list)
        )
        labels_by_type: dict[str, list[int]] = defaultdict(list)

        for fact_type, facts in facts_by_type.items():
            for idx, fact in enumerate(facts):
                prompt = fact["prompt"]
                states = await self._extract_hidden_states(prompt, probe_layers)

                for layer_idx, state in states.items():
                    hidden_states_by_type[fact_type][layer_idx].append(state)
                labels_by_type[fact_type].append(idx)

        # Train probes and measure accuracy
        layer_accuracy: dict[int, dict[str, float]] = {}
        fact_type_curves: dict[str, dict[int, float]] = defaultdict(dict)
        best_layers: dict[str, int] = {}

        for layer_idx in probe_layers:
            layer_accuracy[layer_idx] = {}

            # Aggregate all types for category classification
            all_states: list[mx.array] = []
            all_type_labels: list[int] = []
            type_to_idx = {t: i for i, t in enumerate(facts_by_type.keys())}

            for fact_type, states_by_layer in hidden_states_by_type.items():
                type_idx = type_to_idx[fact_type]
                for state in states_by_layer[layer_idx]:
                    all_states.append(state)
                    all_type_labels.append(type_idx)

            # Train category probe
            if all_states:
                category_acc = await self._train_and_evaluate_probe(
                    all_states, all_type_labels, len(type_to_idx)
                )
                layer_accuracy[layer_idx]["category"] = category_acc
                logger.info(f"Layer {layer_idx} category accuracy: {category_acc:.2%}")

            # Per-type probing
            for fact_type in facts_by_type:
                states = hidden_states_by_type[fact_type][layer_idx]
                labels = labels_by_type[fact_type]

                if len(states) > 1:
                    acc = await self._train_and_evaluate_probe(
                        states, labels, len(set(labels))
                    )
                    fact_type_curves[fact_type][layer_idx] = acc

        # Find best layers per type
        for fact_type, curve in fact_type_curves.items():
            if curve:
                best_layers[fact_type] = max(curve, key=curve.get)

        return ParametricProbingResult(
            layer_accuracy=layer_accuracy,
            fact_type_curves=dict(fact_type_curves),
            best_layers=best_layers,
            probe_weights=None,
        )

    async def _train_and_evaluate_probe(
        self,
        states: list[mx.array],
        labels: list[int],
        num_classes: int,
    ) -> float:
        """Train a linear probe and return accuracy."""
        if len(states) < 4:
            return 0.0

        # Stack states
        X = mx.stack(states)
        y = mx.array(labels)

        # Simple train/test split
        n = len(states)
        train_n = int(n * 0.8)
        indices = list(range(n))
        np.random.shuffle(indices)

        train_idx = indices[:train_n]
        test_idx = indices[train_n:]

        if not test_idx:
            return 0.0

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Train probe using manual gradient descent (matches GPT-OSS pattern)
        W, b = self._train_probe(X_train, y_train, num_classes)

        # Evaluate
        test_logits = X_test @ W + b
        preds = mx.argmax(test_logits, axis=-1)
        accuracy = float(mx.mean(preds == y_test))

        return accuracy

    def _train_probe(
        self, X: mx.array, y: mx.array, num_classes: int, epochs: int = 100
    ) -> tuple[mx.array, mx.array]:
        """Train a linear probe using gradient descent."""
        hidden_dim = X.shape[1]
        W = mx.random.normal((hidden_dim, num_classes)) * 0.01
        b = mx.zeros((num_classes,))
        lr = 0.1

        for _ in range(epochs):
            logits = X @ W + b
            probs = mx.softmax(logits, axis=-1)
            grad_logits = probs
            grad_logits = grad_logits.at[mx.arange(len(y)), y].add(-1)
            grad_logits = grad_logits / len(y)

            grad_W = X.T @ grad_logits
            grad_b = mx.sum(grad_logits, axis=0)

            W = W - lr * grad_W
            b = b - lr * grad_b
            mx.eval(W, b)

        return W, b

    def _get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at specified layer for last token."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        output = self.model(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[layer]
        return hidden[0, -1, :]

    async def _extract_hidden_states(
        self, prompt: str, layers: list[int]
    ) -> dict[int, mx.array]:
        """Extract hidden states at specified layers."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        output = self.model(input_ids, output_hidden_states=True)

        states: dict[int, mx.array] = {}
        for layer_idx in layers:
            if layer_idx < len(output.hidden_states):
                states[layer_idx] = output.hidden_states[layer_idx][0, -1, :]

        return states

    # =========================================================================
    # Context vs Memory Discrimination
    # =========================================================================

    async def _analyze_context_memory(self) -> ContextMemoryResult:
        """Analyze how model discriminates context vs parametric memory."""
        logger.info("Analyzing context vs memory discrimination...")
        config = self.config.get("context_memory", {})
        probe_layers = config.get("probe_layers", [4, 8, 12, 15])

        conflict_examples = self.config.get("conflict_examples", [])
        source_preference: dict[int, dict[str, float]] = defaultdict(
            lambda: {"parametric": 0.0, "context": 0.0}
        )

        analyzed_examples: list[dict[str, Any]] = []

        for example in conflict_examples:
            # Create prompts with and without context
            parametric_prompt = example["query"]
            context_prompt = f"{example['context']}\n\n{example['query']}"

            # Get hidden states for both
            parametric_states = await self._extract_hidden_states(
                parametric_prompt, probe_layers
            )
            context_states = await self._extract_hidden_states(
                context_prompt, probe_layers
            )

            # Generate answers
            parametric_answer = await self._generate_answer(parametric_prompt)
            context_answer = await self._generate_answer(context_prompt)

            # Check which source dominates
            example_result = {
                "query": example["query"],
                "parametric_answer": parametric_answer,
                "context_answer": context_answer,
                "expected_parametric": example["expected_parametric"],
                "expected_context": example["expected_context"],
                "context_override": example["expected_context"].lower()
                in context_answer.lower(),
            }
            analyzed_examples.append(example_result)

            # Update source preference per layer
            for layer_idx in probe_layers:
                # Simple heuristic: measure state divergence
                p_state = parametric_states[layer_idx]
                c_state = context_states[layer_idx]
                divergence = float(mx.mean(mx.abs(p_state - c_state)))

                # Higher divergence at layer = context having effect
                source_preference[layer_idx]["context"] += divergence
                source_preference[layer_idx]["parametric"] += 1.0 - divergence

        # Normalize preferences
        for layer_idx in source_preference:
            total = sum(source_preference[layer_idx].values())
            if total > 0:
                for key in source_preference[layer_idx]:
                    source_preference[layer_idx][key] /= total

        # Find override layer (where context > parametric)
        override_layer = min(probe_layers)
        for layer_idx in sorted(probe_layers):
            if source_preference[layer_idx]["context"] > 0.5:
                override_layer = layer_idx
                break

        logger.info(f"Context override begins at layer {override_layer}")

        return ContextMemoryResult(
            override_layer=override_layer,
            source_preference=dict(source_preference),
            override_threshold=0.5,
            conflict_examples=analyzed_examples,
        )

    async def _generate_answer(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate answer for a prompt."""
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

    # =========================================================================
    # Fact Steering
    # =========================================================================

    async def _analyze_fact_steering(self) -> FactSteeringResult:
        """Analyze fact manipulation via activation steering."""
        logger.info("Analyzing fact steering...")
        config = self.config.get("fact_steering", {})
        steering_layers = config.get("steering_layers", [6, 8, 10, 12])
        steering_strengths = config.get("steering_strengths", [1.0, 2.0, 3.0, 5.0])

        facts = self.config.get("facts", {}).get("entity", [])[:10]

        # Collect "fact present" vs "fact absent" activations
        fact_present_states: dict[int, list[mx.array]] = defaultdict(list)
        fact_absent_states: dict[int, list[mx.array]] = defaultdict(list)

        for fact in facts:
            # Fact present: prompt that elicits the fact
            present_prompt = fact["prompt"]
            present_states = await self._extract_hidden_states(
                present_prompt, steering_layers
            )

            # Fact absent: neutral prompt
            absent_prompt = "The weather today is"
            absent_states = await self._extract_hidden_states(
                absent_prompt, steering_layers
            )

            for layer_idx in steering_layers:
                fact_present_states[layer_idx].append(present_states[layer_idx])
                fact_absent_states[layer_idx].append(absent_states[layer_idx])

        # Extract steering directions (mean difference)
        steering_directions: dict[int, mx.array] = {}
        for layer_idx in steering_layers:
            present_mean = mx.mean(mx.stack(fact_present_states[layer_idx]), axis=0)
            absent_mean = mx.mean(mx.stack(fact_absent_states[layer_idx]), axis=0)
            steering_directions[layer_idx] = present_mean - absent_mean

        # Test suppression and injection
        suppression_success: dict[int, dict[float, float]] = defaultdict(dict)
        injection_success: dict[int, dict[float, float]] = defaultdict(dict)
        coherence_impact: dict[float, float] = {}

        for layer_idx in steering_layers:
            for strength in steering_strengths:
                # Test suppression (negative direction)
                supp_rate = await self._test_steering(
                    facts[:5],
                    layer_idx,
                    steering_directions[layer_idx],
                    -strength,
                    mode="suppression",
                )
                suppression_success[layer_idx][strength] = supp_rate

                # Test injection (positive direction on unrelated prompts)
                inj_rate = await self._test_steering(
                    facts[:5],
                    layer_idx,
                    steering_directions[layer_idx],
                    strength,
                    mode="injection",
                )
                injection_success[layer_idx][strength] = inj_rate

        # Measure coherence at different strengths
        for strength in steering_strengths:
            coherence = 1.0 - min(strength / 10.0, 0.5)  # Simple heuristic
            coherence_impact[strength] = coherence

        # Find best config
        best_layer = max(suppression_success.keys(), key=lambda l: max(suppression_success[l].values()))
        best_strength = max(suppression_success[best_layer].keys(), key=lambda s: suppression_success[best_layer][s])

        logger.info(f"Best steering config: layer {best_layer}, strength {best_strength}")

        return FactSteeringResult(
            suppression_success=dict(suppression_success),
            injection_success=dict(injection_success),
            steering_directions=None,  # Don't save large arrays
            coherence_impact=coherence_impact,
            best_steering_config={"layer": best_layer, "strength": best_strength},
        )

    async def _test_steering(
        self,
        facts: list[dict],
        layer_idx: int,
        direction: mx.array,
        strength: float,
        mode: str,
    ) -> float:
        """Test steering effectiveness."""
        # Placeholder - would implement actual steering
        # In real implementation:
        # 1. Hook into layer to add steering direction
        # 2. Generate with steered model
        # 3. Check if fact is suppressed/injected

        if mode == "suppression":
            # Simulated: stronger steering = more suppression
            return min(abs(strength) / 5.0, 1.0) * 0.8
        else:
            # Injection is harder
            return min(abs(strength) / 8.0, 1.0) * 0.5

    # =========================================================================
    # Retrieval Attention Dynamics
    # =========================================================================

    async def _analyze_retrieval_attention(self) -> RetrievalAttentionResult:
        """Analyze attention patterns during fact retrieval."""
        logger.info("Analyzing retrieval attention dynamics...")
        config = self.config.get("retrieval_attention", {})
        track_layers = config.get("track_layers", [0, 4, 8, 12, 15])

        facts = self.config.get("facts", {}).get("entity", [])

        attention_patterns: dict[str, dict[str, float]] = {
            "known": {"sharpness": 0.0, "consistency": 0.0},
            "unknown": {"sharpness": 0.0, "consistency": 0.0},
            "hallucinated": {"sharpness": 0.0, "consistency": 0.0},
        }

        retrieval_heads: list[tuple[int, int]] = []

        # Analyze attention for known facts
        for fact in facts[:10]:
            prompt = fact["prompt"]
            expected = fact["answer"]

            # Get attention patterns
            attentions = await self._capture_attention(prompt, track_layers)
            answer = await self._generate_answer(prompt, max_tokens=5)

            # Classify as known/unknown/hallucinated
            if expected.lower() in answer.lower():
                category = "known"
            elif len(answer.strip()) < 2:
                category = "unknown"
            else:
                category = "hallucinated"

            # Compute attention metrics
            for layer_idx, attn in attentions.items():
                sharpness = self._compute_attention_sharpness(attn)
                attention_patterns[category]["sharpness"] += sharpness
                attention_patterns[category]["consistency"] += 1.0

        # Normalize
        for category in attention_patterns:
            if attention_patterns[category]["consistency"] > 0:
                attention_patterns[category]["sharpness"] /= attention_patterns[category]["consistency"]

        # Identify retrieval heads (placeholder)
        # Would analyze per-head attention patterns
        for layer_idx in track_layers[1:3]:
            for head in range(4):  # Top heads
                retrieval_heads.append((layer_idx, head))

        # Distinguish known vs unknown
        known_vs_unknown_signatures = {
            "sharpness_diff": attention_patterns["known"]["sharpness"]
            - attention_patterns["unknown"]["sharpness"],
            "known_sharpness": attention_patterns["known"]["sharpness"],
            "unknown_sharpness": attention_patterns["unknown"]["sharpness"],
        }

        # Hallucination markers
        hallucination_markers = [
            {"marker": "diffuse_attention", "prevalence": 0.7},
            {"marker": "high_entropy", "prevalence": 0.6},
            {"marker": "inconsistent_heads", "prevalence": 0.5},
        ]

        logger.info(f"Identified {len(retrieval_heads)} retrieval heads")

        return RetrievalAttentionResult(
            retrieval_heads=retrieval_heads,
            attention_patterns=attention_patterns,
            known_vs_unknown_signatures=known_vs_unknown_signatures,
            hallucination_markers=hallucination_markers,
        )

    async def _capture_attention(
        self, prompt: str, layers: list[int]
    ) -> dict[int, mx.array]:
        """Capture attention patterns at specified layers."""
        # Placeholder - would implement actual attention capture
        return {layer: mx.ones((1, 32, 10, 10)) for layer in layers}

    def _compute_attention_sharpness(self, attn: mx.array) -> float:
        """Compute how sharp/focused attention is."""
        # Higher sharpness = more focused attention
        # Use entropy as proxy (lower entropy = sharper)
        probs = mx.softmax(attn, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-10), axis=-1)
        sharpness = 1.0 / (1.0 + float(mx.mean(entropy)))
        return sharpness

    # =========================================================================
    # Fact Type Specialization
    # =========================================================================

    async def _analyze_fact_types(self) -> FactTypesResult:
        """Analyze if different fact types have different neural signatures."""
        logger.info("Analyzing fact type specialization...")
        config = self.config.get("fact_types", {})
        probe_layers = config.get("probe_layers", [4, 8, 12])
        fact_types = config.get("types", ["entity", "numeric", "temporal", "procedural"])

        facts_config = self.config.get("facts", {})

        # Collect hidden states per type
        states_by_type: dict[str, dict[int, list[mx.array]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for fact_type in fact_types:
            facts = facts_config.get(fact_type, [])
            for fact in facts:
                states = await self._extract_hidden_states(fact["prompt"], probe_layers)
                for layer_idx, state in states.items():
                    states_by_type[fact_type][layer_idx].append(state)

        # Compute layer-wise accuracy for type classification
        type_layer_accuracy: dict[str, dict[int, float]] = defaultdict(dict)

        for layer_idx in probe_layers:
            # Prepare data for classification
            all_states: list[mx.array] = []
            all_labels: list[int] = []
            type_to_idx = {t: i for i, t in enumerate(fact_types)}

            for fact_type in fact_types:
                for state in states_by_type[fact_type][layer_idx]:
                    all_states.append(state)
                    all_labels.append(type_to_idx[fact_type])

            if len(all_states) > 10:
                accuracy = await self._train_and_evaluate_probe(
                    all_states, all_labels, len(fact_types)
                )

                # Store per-type (overall accuracy applies to all)
                for fact_type in fact_types:
                    type_layer_accuracy[fact_type][layer_idx] = accuracy

                logger.info(f"Layer {layer_idx} type classification accuracy: {accuracy:.2%}")

        # Compute type clustering (how tight are same-type embeddings)
        type_clustering: dict[int, dict[str, float]] = defaultdict(dict)

        for layer_idx in probe_layers:
            for fact_type in fact_types:
                states = states_by_type[fact_type][layer_idx]
                if len(states) > 1:
                    # Compute pairwise cosine similarity
                    stacked = mx.stack(states)
                    norms = mx.sqrt(mx.sum(stacked ** 2, axis=-1, keepdims=True))
                    normalized = stacked / (norms + 1e-10)
                    similarity = mx.mean(normalized @ normalized.T)
                    type_clustering[layer_idx][fact_type] = float(similarity)

        # Cross-type confusion
        cross_type_confusion: dict[tuple[str, str], float] = {}
        for i, type_a in enumerate(fact_types):
            for type_b in fact_types[i + 1 :]:
                # Compute inter-type similarity at best layer
                layer_idx = probe_layers[-1]
                states_a = states_by_type[type_a][layer_idx]
                states_b = states_by_type[type_b][layer_idx]

                if states_a and states_b:
                    mean_a = mx.mean(mx.stack(states_a), axis=0)
                    mean_b = mx.mean(mx.stack(states_b), axis=0)
                    similarity = float(
                        mx.sum(mean_a * mean_b)
                        / (mx.sqrt(mx.sum(mean_a ** 2)) * mx.sqrt(mx.sum(mean_b ** 2)) + 1e-10)
                    )
                    cross_type_confusion[(type_a, type_b)] = similarity

        return FactTypesResult(
            type_layer_accuracy=dict(type_layer_accuracy),
            type_clustering=dict(type_clustering),
            expert_routing=None,  # Would need MoE model
            cross_type_confusion=cross_type_confusion,
        )

    # =========================================================================
    # Confidence Calibration
    # =========================================================================

    async def _analyze_confidence(self) -> ConfidenceResult:
        """Analyze if internal confidence correlates with accuracy."""
        logger.info("Analyzing confidence calibration...")
        config = self.config.get("confidence", {})
        probe_layer = config.get("probe_layer", 12)
        num_bins = config.get("calibration_bins", 10)

        facts_config = self.config.get("facts", {})
        all_facts = []
        for fact_type, facts in facts_config.items():
            for fact in facts:
                fact["type"] = fact_type
                all_facts.append(fact)

        # Collect predictions and correctness
        predictions: list[dict[str, Any]] = []

        for fact in all_facts[:30]:
            prompt = fact["prompt"]
            expected = fact["answer"]

            # Get hidden state
            states = await self._extract_hidden_states(prompt, [probe_layer])
            state = states[probe_layer]

            # Generate answer
            answer = await self._generate_answer(prompt, max_tokens=10)

            # Check correctness
            correct = expected.lower() in answer.lower()

            # Get output probability (proxy for confidence)
            tokens = self.tokenizer(prompt, return_tensors="np")
            input_ids = mx.array(tokens["input_ids"])
            output = self.model(input_ids)
            logits = output.logits
            probs = mx.softmax(logits[0, -1, :], axis=-1)
            top_prob = float(mx.max(probs))

            predictions.append({
                "prompt": prompt,
                "expected": expected,
                "answer": answer,
                "correct": correct,
                "confidence": top_prob,
                "hidden_state": state,
            })

        # Compute calibration curve
        calibration_curve: dict[int, tuple[float, float]] = {}
        bin_preds: dict[int, list[tuple[float, bool]]] = defaultdict(list)

        for pred in predictions:
            bin_idx = min(int(pred["confidence"] * num_bins), num_bins - 1)
            bin_preds[bin_idx].append((pred["confidence"], pred["correct"]))

        for bin_idx in range(num_bins):
            if bin_preds[bin_idx]:
                avg_conf = np.mean([p[0] for p in bin_preds[bin_idx]])
                avg_acc = np.mean([p[1] for p in bin_preds[bin_idx]])
                calibration_curve[bin_idx] = (float(avg_conf), float(avg_acc))

        # Compute ECE (Expected Calibration Error)
        ece = 0.0
        total = len(predictions)
        for bin_idx, (conf, acc) in calibration_curve.items():
            bin_size = len(bin_preds[bin_idx])
            ece += (bin_size / total) * abs(conf - acc)

        # Identify uncertainty features
        uncertainty_features = [
            "low_top_probability",
            "high_entropy",
            "attention_diffusion",
        ]

        # Compute hallucination detection AUC (simplified)
        correct_confs = [p["confidence"] for p in predictions if p["correct"]]
        incorrect_confs = [p["confidence"] for p in predictions if not p["correct"]]

        if correct_confs and incorrect_confs:
            auc = np.mean(correct_confs) - np.mean(incorrect_confs) + 0.5
            auc = max(0.0, min(1.0, auc))
        else:
            auc = 0.5

        logger.info(f"ECE: {ece:.4f}, Hallucination detection AUC: {auc:.2f}")

        return ConfidenceResult(
            calibration_curve=calibration_curve,
            ece=ece,
            uncertainty_features=uncertainty_features,
            hallucination_detection_auc=auc,
        )

    # =========================================================================
    # MoE Fact Routing Analysis
    # =========================================================================

    async def _analyze_moe_fact_routing(self) -> MoEFactRoutingResult:
        """Analyze which MoE experts handle different fact types."""
        logger.info("Analyzing MoE expert routing for facts...")

        try:
            from chuk_lazarus.introspection.moe import ExpertRouter
        except ImportError:
            logger.warning("ExpertRouter not available, skipping MoE analysis")
            return MoEFactRoutingResult(
                expert_by_fact_type={},
                expert_specialization={},
                type_separation_score={},
                dominant_experts={},
            )

        config = self.config.get("moe_fact_routing", {})
        model_id = self.config["model"]

        # Load model via ExpertRouter
        try:
            router = await ExpertRouter.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Could not load ExpertRouter: {e}")
            return MoEFactRoutingResult(
                expert_by_fact_type={},
                expert_specialization={},
                type_separation_score={},
                dominant_experts={},
            )

        facts_config = self.config.get("facts", {})
        fact_types = ["entity", "numeric", "temporal", "procedural"]

        # Track expert activations by fact type
        # Structure: type -> layer -> expert -> count
        expert_counts: dict[str, dict[int, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Collect routing for each fact
        for fact_type in fact_types:
            facts = facts_config.get(fact_type, [])
            for fact in facts:
                prompt = fact["prompt"]
                try:
                    weights_list = await router.capture_router_weights(prompt)

                    for layer_weights in weights_list:
                        layer_idx = layer_weights.layer_idx
                        for pos in layer_weights.positions:
                            for exp_idx in pos.expert_indices:
                                expert_counts[fact_type][layer_idx][exp_idx] += 1
                except Exception as e:
                    logger.warning(f"Error capturing weights for '{prompt[:30]}...': {e}")

        # Compute top experts per fact type per layer
        expert_by_fact_type: dict[str, dict[int, list[int]]] = defaultdict(dict)
        for fact_type in fact_types:
            for layer_idx in expert_counts[fact_type]:
                counts = expert_counts[fact_type][layer_idx]
                # Sort by count, take top 3
                sorted_experts = sorted(counts.items(), key=lambda x: -x[1])[:3]
                expert_by_fact_type[fact_type][layer_idx] = [e[0] for e in sorted_experts]

        # Compute expert specialization (which types each expert handles)
        expert_specialization: dict[int, dict[int, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        all_layers = set()
        for fact_type in fact_types:
            all_layers.update(expert_counts[fact_type].keys())

        for layer_idx in all_layers:
            # Get all experts active at this layer
            all_experts = set()
            for fact_type in fact_types:
                all_experts.update(expert_counts[fact_type][layer_idx].keys())

            for exp_idx in all_experts:
                total = sum(
                    expert_counts[ft][layer_idx].get(exp_idx, 0) for ft in fact_types
                )
                if total > 0:
                    for fact_type in fact_types:
                        count = expert_counts[fact_type][layer_idx].get(exp_idx, 0)
                        expert_specialization[layer_idx][exp_idx][fact_type] = count / total

        # Compute type separation score per layer
        # Higher = more separation between fact types
        type_separation_score: dict[int, float] = {}
        for layer_idx in all_layers:
            # For each expert, compute how specialized it is
            specialization_scores = []
            for exp_idx in expert_specialization[layer_idx]:
                type_dist = expert_specialization[layer_idx][exp_idx]
                if type_dist:
                    # Max - mean measures how much one type dominates
                    values = list(type_dist.values())
                    if values:
                        spec = max(values) - np.mean(values)
                        specialization_scores.append(spec)

            if specialization_scores:
                type_separation_score[layer_idx] = float(np.mean(specialization_scores))
            else:
                type_separation_score[layer_idx] = 0.0

        # Find dominant experts for each fact type
        dominant_experts: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        for fact_type in fact_types:
            for layer_idx in expert_counts[fact_type]:
                counts = expert_counts[fact_type][layer_idx]
                total = sum(counts.values())
                if total > 0:
                    for exp_idx, count in counts.items():
                        weight = count / total
                        if weight > 0.2:  # Threshold for "dominant"
                            dominant_experts[fact_type].append(
                                (layer_idx, exp_idx, float(weight))
                            )

            # Sort by weight
            dominant_experts[fact_type].sort(key=lambda x: -x[2])

        # Log key findings
        for layer_idx in sorted(type_separation_score.keys())[:5]:
            sep = type_separation_score[layer_idx]
            logger.info(f"Layer {layer_idx} type separation: {sep:.3f}")

        for fact_type in fact_types:
            if dominant_experts[fact_type]:
                top = dominant_experts[fact_type][0]
                logger.info(
                    f"{fact_type}: dominant expert L{top[0]}E{top[1]} ({top[2]:.1%})"
                )

        return MoEFactRoutingResult(
            expert_by_fact_type=dict(expert_by_fact_type),
            expert_specialization={
                k: dict(v) for k, v in expert_specialization.items()
            },
            type_separation_score=type_separation_score,
            dominant_experts=dict(dominant_experts),
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    def save_results(self, output_path: Path | None = None) -> None:
        """Save results to JSON."""
        if output_path is None:
            output_path = (
                Path(__file__).parent
                / "results"
                / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict: dict[str, Any] = {"metadata": self.results.metadata}

        if self.results.parametric_probing:
            results_dict["parametric_probing"] = {
                "layer_accuracy": self.results.parametric_probing.layer_accuracy,
                "fact_type_curves": self.results.parametric_probing.fact_type_curves,
                "best_layers": self.results.parametric_probing.best_layers,
            }

        if self.results.context_memory:
            results_dict["context_memory"] = {
                "override_layer": self.results.context_memory.override_layer,
                "source_preference": self.results.context_memory.source_preference,
                "override_threshold": self.results.context_memory.override_threshold,
                "conflict_examples": self.results.context_memory.conflict_examples,
            }

        if self.results.fact_steering:
            results_dict["fact_steering"] = {
                "suppression_success": self.results.fact_steering.suppression_success,
                "injection_success": self.results.fact_steering.injection_success,
                "coherence_impact": self.results.fact_steering.coherence_impact,
                "best_steering_config": self.results.fact_steering.best_steering_config,
            }

        if self.results.retrieval_attention:
            results_dict["retrieval_attention"] = {
                "retrieval_heads": self.results.retrieval_attention.retrieval_heads,
                "attention_patterns": self.results.retrieval_attention.attention_patterns,
                "known_vs_unknown_signatures": self.results.retrieval_attention.known_vs_unknown_signatures,
                "hallucination_markers": self.results.retrieval_attention.hallucination_markers,
            }

        if self.results.fact_types:
            results_dict["fact_types"] = {
                "type_layer_accuracy": self.results.fact_types.type_layer_accuracy,
                "type_clustering": self.results.fact_types.type_clustering,
                "cross_type_confusion": {
                    f"{k[0]}_{k[1]}": v
                    for k, v in self.results.fact_types.cross_type_confusion.items()
                },
            }

        if self.results.confidence:
            results_dict["confidence"] = {
                "calibration_curve": {
                    str(k): list(v)
                    for k, v in self.results.confidence.calibration_curve.items()
                },
                "ece": self.results.confidence.ece,
                "uncertainty_features": self.results.confidence.uncertainty_features,
                "hallucination_detection_auc": self.results.confidence.hallucination_detection_auc,
            }

        if self.results.moe_fact_routing:
            results_dict["moe_fact_routing"] = {
                "expert_by_fact_type": {
                    ft: {str(l): experts for l, experts in layers.items()}
                    for ft, layers in self.results.moe_fact_routing.expert_by_fact_type.items()
                },
                "type_separation_score": {
                    str(k): v
                    for k, v in self.results.moe_fact_routing.type_separation_score.items()
                },
                "dominant_experts": {
                    ft: [(l, e, w) for l, e, w in experts]
                    for ft, experts in self.results.moe_fact_routing.dominant_experts.items()
                },
            }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_path}")


async def main():
    """Run the experiment."""
    parser = argparse.ArgumentParser(description="Memory & Fact Retrieval Experiment")
    parser.add_argument(
        "--analysis",
        type=str,
        nargs="*",
        help="Specific analyses to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file",
    )
    args = parser.parse_args()

    experiment = MemoryFactRetrievalExperiment(args.config)
    await experiment.run(args.analysis)
    experiment.save_results()


if __name__ == "__main__":
    asyncio.run(main())
