"""
Constants and enums for model families.

All model-type strings, architecture names, and other identifiers
are defined here to avoid magic strings throughout the codebase.
"""

from enum import Enum


class HFModelType(str, Enum):
    """HuggingFace model_type values from config.json."""

    # Llama family
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    CODELLAMA = "codellama"

    # Llama 4
    LLAMA4 = "llama4"

    # Gemma family
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    GEMMA3 = "gemma3"
    GEMMA3_TEXT = "gemma3_text"

    # Granite family
    GRANITE = "granite"
    GRANITE_MOE_HYBRID = "granitemoehybrid"

    # Jamba
    JAMBA = "jamba"

    # Mamba
    MAMBA = "mamba"

    # StarCoder2
    STARCODER2 = "starcoder2"

    # Qwen family
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"

    # GPT-2 family
    GPT2 = "gpt2"
    GPT_NEO = "gpt_neo"
    GPT_NEOX = "gpt_neox"

    # GPT-OSS (OpenAI open source MoE)
    GPT_OSS = "gpt_oss"
    GPT_OSS_LITE = "gpt_oss_lite"

    # GPT BigCode
    GPT_BIGCODE = "gpt_bigcode"

    # OLMoE
    OLMOE = "olmoe"


class HFArchitecture(str, Enum):
    """HuggingFace architecture class names from config.json."""

    # Llama family
    LLAMA_FOR_CAUSAL_LM = "LlamaForCausalLM"
    MISTRAL_FOR_CAUSAL_LM = "MistralForCausalLM"
    MIXTRAL_FOR_CAUSAL_LM = "MixtralForCausalLM"

    # Llama 4
    LLAMA4_FOR_CAUSAL_LM = "Llama4ForCausalLM"
    LLAMA4_FOR_CONDITIONAL_GENERATION = "Llama4ForConditionalGeneration"

    # Gemma family
    GEMMA_FOR_CAUSAL_LM = "GemmaForCausalLM"
    GEMMA2_FOR_CAUSAL_LM = "Gemma2ForCausalLM"
    GEMMA3_FOR_CAUSAL_LM = "Gemma3ForCausalLM"  # Text-only
    GEMMA3_FOR_CONDITIONAL_GENERATION = "Gemma3ForConditionalGeneration"  # VLM (not native)
    PALIGEMMA_FOR_CONDITIONAL_GENERATION = "PaliGemmaForConditionalGeneration"

    # Granite family
    GRANITE_FOR_CAUSAL_LM = "GraniteForCausalLM"
    GRANITE_MOE_HYBRID_FOR_CAUSAL_LM = "GraniteMoeHybridForCausalLM"

    # Jamba
    JAMBA_FOR_CAUSAL_LM = "JambaForCausalLM"

    # Mamba
    MAMBA_FOR_CAUSAL_LM = "MambaForCausalLM"

    # StarCoder2
    STARCODER2_FOR_CAUSAL_LM = "Starcoder2ForCausalLM"

    # Qwen family
    QWEN2_FOR_CAUSAL_LM = "Qwen2ForCausalLM"
    QWEN3_FOR_CAUSAL_LM = "Qwen3ForCausalLM"

    # GPT-2 family
    GPT2_LM_HEAD_MODEL = "GPT2LMHeadModel"
    GPT_NEO_FOR_CAUSAL_LM = "GPTNeoForCausalLM"
    GPT_NEOX_FOR_CAUSAL_LM = "GPTNeoXForCausalLM"

    # GPT-OSS (OpenAI open source MoE)
    GPT_OSS_FOR_CAUSAL_LM = "GptOssForCausalLM"
    GPT_OSS_LITE_FOR_CAUSAL_LM = "GptOssLiteForCausalLM"
    GPT_OSS_LITE_FOR_CAUSAL_LM_ALT = "GPTOSSLiteForCausalLM"

    # GPT BigCode (StarCoder1)
    GPT_BIGCODE_FOR_CAUSAL_LM = "GPTBigCodeForCausalLM"

    # Mamba alt
    MAMBA_LM_HEAD_MODEL = "MambaLMHeadModel"

    # OLMoE
    OLMOE_FOR_CAUSAL_LM = "OlmoeForCausalLM"


class DefaultVocabSize(int, Enum):
    """Default vocabulary sizes for different model families."""

    LLAMA2 = 32000
    LLAMA3 = 128256
    GEMMA = 256000
    GEMMA3 = 262144
    GPT2 = 50257
    STARCODER2 = 49152
    JAMBA = 65536
    MAMBA = 50280
    QWEN = 151936


class DefaultPositionEmbeddings(int, Enum):
    """Default max position embeddings for different model families."""

    GPT2 = 1024
    LLAMA2 = 4096
    LLAMA3 = 8192
    GEMMA = 8192
    GEMMA3 = 32768
    STARCODER2 = 16384
    JAMBA = 262144
    MAMBA = 2048


class DefaultRoPETheta(float, Enum):
    """Default RoPE theta values for different model families."""

    LLAMA2 = 10000.0
    LLAMA3 = 500000.0
    GEMMA3 = 1000000.0
    STARCODER2 = 100000.0


class DefaultNormEps(float, Enum):
    """Default normalization epsilon values."""

    LLAMA = 1e-5
    GEMMA = 1e-6
    GPT2 = 1e-5
    JAMBA = 1e-6
    MAMBA = 1e-5


class SpecialTokenId(int, Enum):
    """Common special token IDs."""

    # GPT-2 style
    GPT2_EOS = 50256
    GPT2_BOS = 50256

    # Llama style
    LLAMA_BOS = 1
    LLAMA_EOS = 2

    # Gemma style
    GEMMA_END_OF_TURN = 106


# Config field names (to avoid typos)
class ConfigField(str, Enum):
    """Field names in HuggingFace config.json."""

    MODEL_TYPE = "model_type"
    ARCHITECTURES = "architectures"
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "hidden_size"
    NUM_HIDDEN_LAYERS = "num_hidden_layers"
    NUM_ATTENTION_HEADS = "num_attention_heads"
    NUM_KEY_VALUE_HEADS = "num_key_value_heads"
    INTERMEDIATE_SIZE = "intermediate_size"
    MAX_POSITION_EMBEDDINGS = "max_position_embeddings"
    ROPE_THETA = "rope_theta"
    RMS_NORM_EPS = "rms_norm_eps"
    TIE_WORD_EMBEDDINGS = "tie_word_embeddings"
    BOS_TOKEN_ID = "bos_token_id"
    EOS_TOKEN_ID = "eos_token_id"
    PAD_TOKEN_ID = "pad_token_id"
    SLIDING_WINDOW = "sliding_window"

    # GPT-2 specific
    N_EMBD = "n_embd"
    N_LAYER = "n_layer"
    N_HEAD = "n_head"
    N_INNER = "n_inner"
    N_POSITIONS = "n_positions"
    LAYER_NORM_EPSILON = "layer_norm_epsilon"

    # Gemma specific
    HEAD_DIM = "head_dim"
    SLIDING_WINDOW_PATTERN = "sliding_window_pattern"
    QUERY_PRE_ATTN_SCALAR = "query_pre_attn_scalar"

    # Common optional fields
    HIDDEN_ACT = "hidden_act"
    HIDDEN_ACTIVATION = "hidden_activation"
    ATTENTION_BIAS = "attention_bias"
    MLP_BIAS = "mlp_bias"
    ATTENTION_DROPOUT = "attention_dropout"
    ROPE_SCALING = "rope_scaling"
    USE_BIAS = "use_bias"

    # Jamba specific
    ATTN_LAYER_PERIOD = "attn_layer_period"
    ATTN_LAYER_OFFSET = "attn_layer_offset"
    EXPERT_LAYER_PERIOD = "expert_layer_period"
    EXPERT_LAYER_OFFSET = "expert_layer_offset"
    NUM_EXPERTS = "num_experts"
    NUM_EXPERTS_PER_TOK = "num_experts_per_tok"
    MAMBA_D_STATE = "mamba_d_state"
    MAMBA_D_CONV = "mamba_d_conv"
    MAMBA_EXPAND = "mamba_expand"
    MAMBA_DT_RANK = "mamba_dt_rank"
    MAMBA_CONV_BIAS = "mamba_conv_bias"
    MAMBA_PROJ_BIAS = "mamba_proj_bias"

    # Mamba standalone
    D_MODEL = "d_model"
    STATE_SIZE = "state_size"
    CONV_KERNEL = "conv_kernel"
    EXPAND_FACTOR = "expand_factor"

    # Gemma extended
    ROPE_LOCAL_BASE_FREQ = "rope_local_base_freq"
    TEXT_CONFIG = "text_config"

    # Granite specific
    EMBEDDING_MULTIPLIER = "embedding_multiplier"
    ATTENTION_MULTIPLIER = "attention_multiplier"
    RESIDUAL_MULTIPLIER = "residual_multiplier"
    LOGITS_SCALING = "logits_scaling"
    POSITION_EMBEDDING_TYPE = "position_embedding_type"

    # Granite hybrid / Mamba2
    LAYER_TYPES = "layer_types"
    MAMBA_N_HEADS = "mamba_n_heads"
    MAMBA_D_HEAD = "mamba_d_head"
    MAMBA_N_GROUPS = "mamba_n_groups"
    MAMBA_CHUNK_SIZE = "mamba_chunk_size"
    SHARED_INTERMEDIATE_SIZE = "shared_intermediate_size"
    ROUTER_AUX_LOSS_COEF = "router_aux_loss_coef"
    OUTPUT_ROUTER_LOGITS = "output_router_logits"

    # OLMoE specific
    NORM_TOPK_PROB = "norm_topk_prob"
    NUM_LOCAL_EXPERTS = "num_local_experts"

    # Llama4 specific
    NO_ROPE_LAYERS = "no_rope_layers"
    NOPE_LAYER_INTERVAL = "nope_layer_interval"
    INTERMEDIATE_SIZE_MLP = "intermediate_size_mlp"
    MOE_ROUTER_TOPK = "moe_router_topk"
    USE_QK_NORM = "use_qk_norm"
    ATTN_TEMPERATURE_TUNING = "attn_temperature_tuning"

    # StarCoder2 specific
    NORM_EPSILON = "norm_epsilon"

    # GPT-OSS specific
    SWIGLU_LIMIT = "swiglu_limit"

    # GPT-OSS-Lite specific
    SOURCE_MODEL = "source_model"
    REDUCTION_PERCENT = "reduction_percent"
    ORIGINAL_EXPERTS = "original_experts"
    TOTAL_EXPERTS = "total_experts"

    # MLX community format aliases
    DIM = "dim"
    N_LAYERS = "n_layers"
    N_HEADS = "n_heads"
    N_KV_HEADS = "n_kv_heads"
    HIDDEN_DIM = "hidden_dim"
    NORM_EPS = "norm_eps"
