"""
Gemma Residual Stream architecture.

Architectural principle: the Markov property.

The residual tensor at any layer is the complete forward state for all subsequent
layers. No information exists outside the residual stream. This means:

  - K and V at layer N are deterministic functions of the residual at layer N.
  - The KV cache is redundant: it stores what the residual already encodes.
  - For single-pass inference, cache=None IS residual-stream inference.

This module makes that principle explicit and deployable:

  forward(input_ids)
      Full single-pass inference. Returns logits + final residual.
      Identical output to GemmaForCausalLM with cache=None.

  forward_to_layer(input_ids, stop_layer)
      Run layers 0..stop_layer-1. Return residual at that depth.
      Enables probe readout at any intermediate layer.

  forward_from_layer(residual, start_layer)
      Accept an external residual, run layers start_layer..N-1, return logits.
      Enables injection: swap the residual at any depth and continue.

  forward_between_layers(residual, start_layer, end_layer)
      Run a contiguous slice of layers. Compose freely.

Weight layout is identical to GemmaForCausalLM — same .safetensors files load.

No KV cache parameter. No cache return value. The residual is the only state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..gemma.config import GemmaConfig
from ..gemma.model import (
    GemmaAttention,
    GemmaMLP,
    GemmaRMSNorm,
    clip_residual,
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class ResidualStreamOutput:
    """Output of a full forward pass through the residual stream."""

    logits: mx.array
    residual: mx.array
    layer_residuals: list[mx.array] | None = None


@dataclass
class PartialResidualOutput:
    """Output of a partial forward pass (to or between layers)."""

    residual: mx.array
    layer_idx: int


# ---------------------------------------------------------------------------
# Block — identical computation to GemmaBlock, no cache
# ---------------------------------------------------------------------------


class GemmaRSBlock(nn.Module):
    """
    Gemma transformer block, residual-stream edition.

    Computationally identical to GemmaBlock. The only difference:
    - No cache parameter accepted or returned.
    - K and V are computed from the current residual and immediately discarded.
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # Attention: compute K,V fresh from x, use them, discard them.
        r, _discarded_kv = self.self_attn(self.input_layernorm(x), mask, cache=None)
        h = clip_residual(x, self.post_attention_layernorm(r))

        # FFN
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = clip_residual(h, self.post_feedforward_layernorm(r))

        return out


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class GemmaResidualStream(nn.Module):
    """
    Gemma backbone, residual stream only.

    Weight names match GemmaModel exactly:
      embed_tokens, layers[i].*, norm

    The forward path accepts start_layer / stop_layer so callers can run
    any contiguous slice of the 34-layer stack.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers
        self.sliding_window = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GemmaRSBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _global_mask(self, h: mx.array) -> mx.array:
        _, seq_len, _ = h.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        return mask.astype(h.dtype)

    def _sliding_mask(self, h: mx.array) -> mx.array:
        _, seq_len, _ = h.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)
        if seq_len > 1:
            positions = mx.arange(seq_len)
            window_mask = mx.where(
                positions[:, None] - positions[None, :] >= self.sliding_window,
                float("-inf"),
                0.0,
            )
            mask = mask + window_mask.astype(h.dtype)
        return mask

    def _mask_for_layer(self, layer_idx: int, h: mx.array) -> mx.array | None:
        if h.shape[1] <= 1:
            return None
        if self.config.is_global_layer(layer_idx):
            return self._global_mask(h)
        return self._sliding_mask(h)

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def _embed(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        scale = mx.array(self.config.hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)
        return h * scale

    # ------------------------------------------------------------------
    # Core forward: run layers [start, stop)
    # ------------------------------------------------------------------

    def _run_layers(
        self,
        h: mx.array,
        start_layer: int,
        stop_layer: int,
        collect: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """
        Run layers [start_layer, stop_layer) on residual h.

        Returns (residual_after_last_layer, optional_per_layer_list).
        Does NOT apply final norm — that is the caller's responsibility.
        """
        layer_residuals = [] if collect else None

        for i in range(start_layer, stop_layer):
            mask = self._mask_for_layer(i, h)
            h = self.layers[i](h, mask=mask)
            if collect:
                layer_residuals.append(h)

        return h, layer_residuals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, input_ids: mx.array) -> mx.array:
        """Embed input tokens into the residual stream (layer 0 input)."""
        return self._embed(input_ids)

    def forward(
        self,
        input_ids: mx.array,
        collect_layer_residuals: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """
        Full single-pass forward. Returns (final_residual_after_norm, layer_residuals).

        final_residual is after the final RMSNorm, ready for the LM head.
        """
        h = self._embed(input_ids)
        h, layer_residuals = self._run_layers(h, 0, len(self.layers), collect=collect_layer_residuals)
        h = self.norm(h)
        return h, layer_residuals

    def forward_to_layer(
        self,
        input_ids: mx.array,
        stop_layer: int,
    ) -> PartialResidualOutput:
        """
        Run layers 0..stop_layer-1. Return residual before layer stop_layer.

        stop_layer=7 means: embed + run layers 0,1,2,3,4,5,6 → return residual.
        The returned residual is the complete state for layers 7..N.
        """
        assert 0 <= stop_layer <= len(self.layers), (
            f"stop_layer {stop_layer} out of range [0, {len(self.layers)}]"
        )
        h = self._embed(input_ids)
        h, _ = self._run_layers(h, 0, stop_layer)
        return PartialResidualOutput(residual=h, layer_idx=stop_layer)

    def forward_from_layer(
        self,
        residual: mx.array,
        start_layer: int,
        apply_norm: bool = True,
    ) -> mx.array:
        """
        Accept an external residual at start_layer, run to end, return final residual.

        The residual may be:
        - The output of forward_to_layer (probe then continue)
        - A modified/injected residual (steer the forward pass)
        - A residual from a different prompt (template branching)

        apply_norm: if True, apply the final RMSNorm (default for LM head use).
        """
        assert 0 <= start_layer <= len(self.layers), (
            f"start_layer {start_layer} out of range [0, {len(self.layers)}]"
        )
        h, _ = self._run_layers(residual, start_layer, len(self.layers))
        if apply_norm:
            h = self.norm(h)
        return h

    def forward_between_layers(
        self,
        residual: mx.array,
        start_layer: int,
        end_layer: int,
    ) -> PartialResidualOutput:
        """
        Run layers [start_layer, end_layer). Return residual before end_layer.

        For composing multi-segment passes.
        """
        assert 0 <= start_layer <= end_layer <= len(self.layers)
        h, _ = self._run_layers(residual, start_layer, end_layer)
        return PartialResidualOutput(residual=h, layer_idx=end_layer)


# ---------------------------------------------------------------------------
# Full model with LM head
# ---------------------------------------------------------------------------


class GemmaResidualStreamForCausalLM(nn.Module):
    """
    Gemma 3 for causal LM, residual stream architecture.

    Weight layout is identical to GemmaForCausalLM:
      model.embed_tokens, model.layers[i].*, model.norm, lm_head

    So the same .safetensors checkpoint loads into either class.

    No KV cache. The residual tensor IS the state.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self._config = config
        self.tie_word_embeddings = False

        self.model = GemmaResidualStream(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def config(self) -> GemmaConfig:
        return self._config

    @property
    def layers(self):
        return self.model.layers

    # ------------------------------------------------------------------
    # Unembed helper
    # ------------------------------------------------------------------

    def _unembed(self, residual: mx.array) -> mx.array:
        if self.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(residual)
        return self.lm_head(residual)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        collect_layer_residuals: bool = False,
        **_ignored,
    ) -> ResidualStreamOutput:
        """
        Full single-pass inference on the residual stream.

        For a prompt of length L, runs all N layers once. K and V are
        computed fresh at each layer from the current residual over all L
        positions and immediately discarded. No cache is built or returned.

        Output is numerically identical to GemmaForCausalLM(input_ids, cache=None).
        """
        residual, layer_residuals = self.model.forward(
            input_ids, collect_layer_residuals=collect_layer_residuals
        )
        logits = self._unembed(residual)
        return ResidualStreamOutput(
            logits=logits,
            residual=residual,
            layer_residuals=layer_residuals,
        )

    def forward_to_layer(
        self,
        input_ids: mx.array,
        stop_layer: int,
    ) -> PartialResidualOutput:
        """
        Single pass to layer stop_layer. Return the residual before that layer.

        Use for probe readout:
            partial = model.forward_to_layer(input_ids, stop_layer=7)
            probe_value = linear_probe(partial.residual[:, -1, :])
        """
        return self.model.forward_to_layer(input_ids, stop_layer)

    def forward_from_layer(
        self,
        residual: mx.array,
        start_layer: int,
    ) -> ResidualStreamOutput:
        """
        Continue a forward pass from an injected residual at start_layer.

        Use for injection:
            partial = model.forward_to_layer(input_ids, stop_layer=10)
            modified = partial.residual + steering_vector
            out = model.forward_from_layer(modified, start_layer=10)

        Use for template branching:
            template_partial = model.forward_to_layer(template_ids, stop_layer=5)
            out = model.forward_from_layer(template_partial.residual, start_layer=5)
        """
        residual_normed = self.model.forward_from_layer(residual, start_layer, apply_norm=True)
        logits = self._unembed(residual_normed)
        return ResidualStreamOutput(logits=logits, residual=residual_normed)

    def forward_between_layers(
        self,
        residual: mx.array,
        start_layer: int,
        end_layer: int,
    ) -> PartialResidualOutput:
        """
        Run a contiguous slice of layers. Compose freely with forward_to_layer
        and forward_from_layer.
        """
        return self.model.forward_between_layers(residual, start_layer, end_layer)

    # ------------------------------------------------------------------
    # Weight loading compatibility
    # ------------------------------------------------------------------

    def sanitize(self, weights: dict) -> dict:
        """
        Same sanitize logic as GemmaForCausalLM.

        Handles VLM-style weights with 'language_model.' prefix,
        filters vision components, detects tied embeddings.
        """
        has_language_model_prefix = any(k.startswith("language_model.") for k in weights)

        if has_language_model_prefix:
            weights = {
                k.replace("language_model.", "", 1): v
                for k, v in weights.items()
                if k.startswith("language_model.")
            }
        else:
            vlm_prefixes = ("vision_tower.", "multi_modal_projector.", "image_")
            weights = {
                k: v
                for k, v in weights.items()
                if not any(k.startswith(p) for p in vlm_prefixes)
            }

        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True

        return weights

    @classmethod
    def from_config(cls, config: GemmaConfig) -> GemmaResidualStreamForCausalLM:
        return cls(config)
