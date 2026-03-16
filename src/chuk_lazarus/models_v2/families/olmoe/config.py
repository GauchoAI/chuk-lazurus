"""
OLMoE configuration.

Based on Llama config with MoE-specific additions.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import ConfigField, DefaultNormEps, DefaultRoPETheta, HFModelType


class OLMoEConfig(ModelConfig):
    """
    Configuration for OLMoE models.

    OLMoE is Allen AI's open MoE model based on Llama architecture.
    Key differences from dense Llama:
    - num_experts: Total number of experts per layer (typically 64)
    - num_experts_per_tok: Number of active experts per token (typically 8)
    - intermediate_size: Size of each expert's FFN (smaller than dense equivalent)

    Example:
        >>> config = OLMoEConfig(
        ...     vocab_size=50304,
        ...     hidden_size=2048,
        ...     num_hidden_layers=16,
        ...     num_attention_heads=16,
        ...     num_key_value_heads=16,
        ...     intermediate_size=1024,  # Per expert
        ...     num_experts=64,
        ...     num_experts_per_tok=8,
        ... )
    """

    model_type: str = "olmoe"

    # Llama-like defaults
    hidden_act: str = "silu"
    rope_theta: float = DefaultRoPETheta.LLAMA2.value
    rms_norm_eps: float = DefaultNormEps.LLAMA.value

    # MoE configuration
    num_experts: int = 64
    num_experts_per_tok: int = 8

    # Router configuration
    router_aux_loss_coef: float = 0.01
    norm_topk_prob: bool = False  # Whether to normalize top-k probabilities
    output_router_logits: bool = False

    # Optional RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def olmoe_1b_7b(cls) -> OLMoEConfig:
        """Create OLMoE-1B-7B configuration (7B total, 1B active)."""
        return cls(
            vocab_size=50304,
            hidden_size=2048,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=1024,  # Per expert
            max_position_embeddings=4096,
            num_experts=64,
            num_experts_per_tok=8,
            tie_word_embeddings=False,  # OLMoE doesn't tie embeddings
        )

    @classmethod
    def tiny(cls) -> OLMoEConfig:
        """Create tiny OLMoE for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=32,
            max_position_embeddings=256,
            num_experts=8,
            num_experts_per_tok=2,
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        weights: dict[str, Any] | None = None,
    ) -> OLMoEConfig:
        """
        Create config from HuggingFace config.json dict.

        Args:
            hf_config: Dict loaded from config.json
            weights: Optional weights dict (not used)

        Returns:
            OLMoEConfig instance
        """
        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE.value, HFModelType.OLMOE.value),
            vocab_size=hf_config[ConfigField.VOCAB_SIZE.value],
            hidden_size=hf_config[ConfigField.HIDDEN_SIZE.value],
            num_hidden_layers=hf_config[ConfigField.NUM_HIDDEN_LAYERS.value],
            num_attention_heads=hf_config[ConfigField.NUM_ATTENTION_HEADS.value],
            num_key_value_heads=hf_config.get(
                ConfigField.NUM_KEY_VALUE_HEADS.value,
                hf_config[ConfigField.NUM_ATTENTION_HEADS.value],
            ),
            intermediate_size=hf_config[ConfigField.INTERMEDIATE_SIZE.value],
            max_position_embeddings=hf_config.get(ConfigField.MAX_POSITION_EMBEDDINGS.value, 4096),
            rope_theta=hf_config.get(ConfigField.ROPE_THETA.value, DefaultRoPETheta.LLAMA2.value),
            rms_norm_eps=hf_config.get(ConfigField.RMS_NORM_EPS.value, DefaultNormEps.LLAMA.value),
            tie_word_embeddings=hf_config.get(ConfigField.TIE_WORD_EMBEDDINGS.value, False),
            bos_token_id=hf_config.get(ConfigField.BOS_TOKEN_ID.value, 1),
            eos_token_id=hf_config.get(ConfigField.EOS_TOKEN_ID.value, 50279),
            pad_token_id=hf_config.get(ConfigField.PAD_TOKEN_ID.value, 1),
            # MoE specific
            num_experts=hf_config.get(ConfigField.NUM_EXPERTS.value, 64),
            num_experts_per_tok=hf_config.get(ConfigField.NUM_EXPERTS_PER_TOK.value, 8),
            router_aux_loss_coef=hf_config.get(ConfigField.ROUTER_AUX_LOSS_COEF.value, 0.01),
            norm_topk_prob=hf_config.get(ConfigField.NORM_TOPK_PROB.value, False),
            output_router_logits=hf_config.get(ConfigField.OUTPUT_ROUTER_LOGITS.value, False),
            rope_scaling=hf_config.get(ConfigField.ROPE_SCALING.value),
        )
