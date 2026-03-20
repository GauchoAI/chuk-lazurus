#!/usr/bin/env python3
"""
Copy-head discovery for ArchitectureConfig calibration.

Two-part behavioral calibration:

  Part A — Injection layer scan:
    For each layer L, compute c = dot(h_L[answer_pos], embed(answer)) at the
    answer token's position in each fact document.  The layer where c peaks is
    the injection layer (retrieval_layer = injection_layer - 1).  This directly
    measures where the vec_inject coefficient signal is strongest.

  Part B — Causal ablation:
    At retrieval_layer, for each query head H:
      1. Recompute the layer's SDPA with head H zeroed (full FFN re-run, not
         a linear subtraction — the FFN sees the ablated residual).
      2. Continue the forward pass from retrieval_layer + 1.
      3. delta = P(answer | baseline) - P(answer | head_H_zeroed)
    The head with the largest causal delta is the copy head.

    On Gemma 4B this should recover L29 H4 with delta ≈ 0.19 (96.9%→78.1%).
    Run against 4B first to validate, then trust the 1B result.

Supported model families:
  - Gemma (gemma, gemma2, gemma3, gemma3_text): uses GemmaResidualStream
  - Llama (llama, mistral, codellama): uses LlamaForCausalLM with manual iteration

Usage
-----
    uv run python examples/inference/calibrate_arch.py
    uv run python examples/inference/calibrate_arch.py --model mlx-community/gemma-3-4b-it-bf16
    uv run python examples/inference/calibrate_arch.py --model mlx-community/gemma-3-1b-it-bf16
    uv run python examples/inference/calibrate_arch.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct
    uv run python examples/inference/calibrate_arch.py --model /path/to/local/model
    uv run python examples/inference/calibrate_arch.py --top-layers 8   # faster scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Probes — paired (document, answer).
# The answer word appears verbatim in the document so we can locate it.
# Novel entities to avoid parametric knowledge interference.
# ---------------------------------------------------------------------------

# Probes use a plain cloze-completion format (no instruct template).
# The prompt ends immediately before the answer token so the model must
# predict it next.  This gives non-trivial P(answer) values that the causal
# ablation can meaningfully change.
# Part A still uses the full instruct-template prompt to locate answer
# positions in context; Part B uses only the cloze_prefix field.
PROBES = [
    {
        "doc": (
            "Zarkov Industries was established in the mid-1990s as a pioneering manufacturer "
            "of industrial filtration systems. Its headquarters, built on a former industrial "
            "lot, became a landmark of Voltara's commercial district. Today Zarkov Industries "
            "employs over 2,400 people.\n\n"
            "What city was Zarkov Industries founded in?"
        ),
        # Repeat-completion format: state the fact, then start repeating it.
        # The model must copy the entity name from the preceding context.
        "cloze_prefix": (
            "Zarkov Industries is based in Voltara. "
            "Zarkov Industries is based in"
        ),
        "answer": "Voltara",
    },
    {
        "doc": (
            "Nexaris Corporation began as a small software consultancy before pivoting to "
            "enterprise data management. The founding team worked out of a converted warehouse "
            "in Cerulion's technology corridor. Their flagship product attracted its first "
            "Fortune 500 customer within eighteen months.\n\n"
            "What city was Nexaris Corporation founded in?"
        ),
        "cloze_prefix": (
            "Nexaris Corporation is headquartered in Cerulion. "
            "Nexaris Corporation is headquartered in"
        ),
        "answer": "Cerulion",
    },
    {
        "doc": (
            "Helion Systems traces its origins to a university spin-out that relocated from "
            "campus to leased office space in Dravenport's innovation quarter in 2003. "
            "Within a decade the company had grown from eight employees to more than six hundred.\n\n"
            "What city was Helion Systems founded in?"
        ),
        "cloze_prefix": (
            "Helion Systems is located in Dravenport. "
            "Helion Systems is located in"
        ),
        "answer": "Dravenport",
    },
    {
        "doc": (
            "Keltara Dynamics was incorporated in Solmere following a management buyout. "
            "Keltara's Solmere plant has received three national manufacturing excellence awards.\n\n"
            "What city was Keltara Dynamics founded in?"
        ),
        "cloze_prefix": (
            "Keltara Dynamics is incorporated in Solmere. "
            "Keltara Dynamics is incorporated in"
        ),
        "answer": "Solmere",
    },
    {
        "doc": (
            "Joe Namath was approached by Fabergé Inc. regarding a promotional campaign. "
            "After negotiations, Namath agreed to endorse Brut cologne, appearing in a "
            "television advertisement that became one of the most recognised sports endorsements.\n\n"
            "What did Joe Namath agree to do?"
        ),
        "cloze_prefix": (
            "Joe Namath agreed to endorse Brut cologne. "
            "Joe Namath agreed to"
        ),
        "answer": "endorse",
    },
    {
        "doc": (
            "Sylvia Marchand, a retired art dealer based in Geneva, followed advice from "
            "her estate lawyers. She agreed to sell her painting through a major auction house "
            "rather than in a private transaction.\n\n"
            "What did Sylvia Marchand agree to do?"
        ),
        "cloze_prefix": (
            "Sylvia Marchand agreed to sell her painting. "
            "Sylvia Marchand agreed to"
        ),
        "answer": "sell",
    },
]


# ---------------------------------------------------------------------------
# Model loading — supports Gemma and Llama families
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    try:
        cached = snapshot_download(
            model_id,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
        return Path(cached)
    except Exception:
        pass
    print(f"  Downloading {model_id}...")
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
    )


def _detect_family(config_data: dict) -> str:
    """Return 'gemma' or 'llama' based on config.json model_type."""
    mt = config_data.get("model_type", "").lower()
    if mt in ("gemma", "gemma2", "gemma3", "gemma3_text"):
        return "gemma"
    if mt in ("llama", "mistral", "codellama", "smollm", "llama4"):
        return "llama"
    # Default: try Llama (it's the broader family)
    print(f"  {YELLOW}Warning: unknown model_type={mt!r}, trying Llama family loader{RESET}")
    return "llama"


def _load_gemma(model_path: Path, config_data: dict):
    """Load Gemma model, return (GemmaResidualStreamForCausalLM, GemmaConfig)."""
    from mlx.utils import tree_unflatten

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    config = GemmaConfig.from_hf_config(config_data)

    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))

    std = GemmaForCausalLM(config)
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in std.sanitize(raw).items()
    }
    std.update(tree_unflatten(list(sanitized.items())))
    mx.eval(std.parameters())
    std.eval()

    rs = GemmaResidualStreamForCausalLM(config)
    rs.update(std.parameters())
    mx.eval(rs.parameters())
    rs.eval()

    return rs, config


def _load_llama(model_path: Path, config_data: dict):
    """Load Llama model, return (LlamaForCausalLM, LlamaConfig)."""
    from mlx.utils import tree_unflatten

    from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM
    from chuk_lazarus.models_v2.families.llama.convert import convert_hf_weights

    config = LlamaConfig.from_hf_config(config_data)

    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))

    model = LlamaForCausalLM(config)
    converted = convert_hf_weights(
        raw, tie_word_embeddings=config.tie_word_embeddings
    )
    # Cast floats to bfloat16 for memory efficiency
    converted = {
        k: v.astype(mx.bfloat16) if hasattr(v, "dtype") and v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in converted.items()
    }
    model.update(tree_unflatten(list(converted.items())))
    mx.eval(model.parameters())
    model.eval()

    return model, config


def load_model_runner(model_id: str) -> "CalibRunner":
    """Load model and return a CalibRunner wrapping it."""
    print(f"  Loading {model_id}…")
    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family = _detect_family(config_data)
    print(f"  Detected family: {family}")

    if family == "gemma":
        rs, config = _load_gemma(model_path, config_data)
        return CalibRunner(family="gemma", gemma_rs=rs, config=config)
    else:
        model, config = _load_llama(model_path, config_data)
        return CalibRunner(family="llama", llama_model=model, config=config)


# ---------------------------------------------------------------------------
# CalibRunner — family-agnostic interface for calibration forward passes
# ---------------------------------------------------------------------------


class CalibRunner:
    """Uniform calibration interface for Gemma (RS) and Llama models."""

    def __init__(
        self,
        family: str,
        gemma_rs=None,
        llama_model=None,
        config=None,
    ) -> None:
        self.family = family
        self._rs = gemma_rs
        self._llama = llama_model
        self.config = config

    @property
    def num_layers(self) -> int:
        if self.family == "gemma":
            return len(self._rs.model.layers)
        return len(self._llama.model.layers)

    def num_heads_at(self, layer_idx: int) -> int:
        if self.family == "gemma":
            return self._rs.model.layers[layer_idx].self_attn.num_heads
        return self._llama.model.layers[layer_idx].self_attn.num_heads

    def get_embedding(self, tok_id: int) -> mx.array:
        """Return embedding vector for tok_id, shape (hidden_size,), float32."""
        if self.family == "gemma":
            e = self._rs.model.embed_tokens(mx.array([[tok_id]]))[0, 0, :]
        else:
            e = self._llama.model.embed_tokens(mx.array([[tok_id]]))[0, 0, :]
        return e.astype(mx.float32)

    def forward_full(self, ids: mx.array) -> mx.array:
        """Return logits (batch, seq, vocab)."""
        if self.family == "gemma":
            return self._rs(ids).logits
        return self._llama(ids).logits

    def forward_to_layer(self, ids: mx.array, stop_layer: int) -> mx.array:
        """Return residual stream before layer stop_layer (layers 0..stop_layer-1 run)."""
        if self.family == "gemma":
            return self._rs.forward_to_layer(ids, stop_layer=stop_layer).residual
        # Llama: manual layer iteration
        h = self._llama.model.embed_tokens(ids)
        S = ids.shape[1]
        for i in range(stop_layer):
            mask = self._llama_mask(i, S, h.dtype)
            out = self._llama.model.layers[i](h, mask=mask)
            h = out.hidden_states
        return h

    def forward_from_layer(self, h: mx.array, start_layer: int) -> mx.array:
        """Continue from h at start_layer, return logits."""
        if self.family == "gemma":
            return self._rs.forward_from_layer(h, start_layer=start_layer).logits
        # Llama: manual layer continuation + final norm + lm_head
        S = h.shape[1]
        for i in range(start_layer, len(self._llama.model.layers)):
            mask = self._llama_mask(i, S, h.dtype)
            out = self._llama.model.layers[i](h, mask=mask)
            h = out.hidden_states
        h = self._llama.model.norm(h)
        return self._llama.lm_head(hidden_states=h).logits

    def run_layer_ablated(self, layer_idx: int, head_idx: int, h_in: mx.array) -> mx.array:
        """Run layer with head head_idx zeroed; re-run FFN on ablated residual."""
        if self.family == "gemma":
            return _run_gemma_layer_ablated(self._rs, layer_idx, head_idx, h_in)
        return _run_llama_layer_ablated(self._llama, layer_idx, head_idx, h_in)

    def _llama_mask(self, layer_idx: int, S: int, dtype) -> mx.array:
        block = self._llama.model.layers[layer_idx]
        from chuk_lazarus.models_v2.components.attention.sliding_window import SlidingWindowAttention
        if isinstance(block.self_attn, SlidingWindowAttention):
            from chuk_lazarus.models_v2.components.attention.base import create_sliding_window_mask
            return create_sliding_window_mask(S, block.self_attn.window_size, dtype=dtype)
        return nn.MultiHeadAttention.create_additive_causal_mask(S).astype(dtype)

    def build_doc_prompt(self, doc_text: str) -> str:
        """Wrap doc text in model-appropriate instruct format for Part A."""
        if self.family == "gemma":
            return (
                "<bos><start_of_turn>user\n"
                f"{doc_text}"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        # Llama/Mistral: plain text is sufficient for injection layer scan
        return doc_text


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def find_answer_position(tokenizer, doc_ids: list[int], answer: str) -> int | None:
    """Return position of the first token of the answer word, or None."""
    for candidate in (" " + answer, answer):
        cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if not cand_ids:
            continue
        first_tok = cand_ids[0]
        for pos, tok in enumerate(doc_ids):
            if tok == first_tok:
                return pos
    return None


def get_answer_first_token(tokenizer, answer: str) -> int:
    for candidate in (" " + answer, answer):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return ids[0]
    raise ValueError(f"Cannot tokenize answer: {answer!r}")


# ---------------------------------------------------------------------------
# Layer ablation — Gemma (4-norm, clip_residual on deltas)
# ---------------------------------------------------------------------------


def _run_gemma_layer_ablated(rs, layer_idx: int, head_idx: int, h_in: mx.array) -> mx.array:
    """Run Gemma layer with head_idx zeroed. Re-runs FFN on ablated residual.

    Gemma block structure:
      h_after_attn = clip_residual(h_in, post_attention_layernorm(attn_delta))
      h_out        = clip_residual(h_after_attn, post_feedforward_layernorm(ffn_delta))
    """
    from chuk_lazarus.models_v2.families.gemma.model import clip_residual

    block = rs.model.layers[layer_idx]
    attn = block.self_attn
    B, S, _ = h_in.shape
    nq = attn.num_heads
    nkv = attn.num_kv_heads
    dh = attn.head_dim
    n_rep = nq // nkv

    # --- Replicate GemmaAttention forward, keeping SDPA output per-head ---
    x = block.input_layernorm(h_in)
    q = attn.q_proj(x).reshape(B, S, nq, dh).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    q = attn.rope(q)
    k = attn.rope(k)
    if n_rep > 1:
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    mask = rs.model._mask_for_layer(layer_idx, h_in)
    sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    # sdpa: (B, nq, S, dh)

    # Zero head_idx — build ablated SDPA output
    heads = [
        mx.zeros_like(sdpa[:, h : h + 1, :, :]) if h == head_idx else sdpa[:, h : h + 1, :, :]
        for h in range(nq)
    ]
    sdpa_ablated = mx.concatenate(heads, axis=1)  # (B, nq, S, dh)

    r_ablated = attn.o_proj(sdpa_ablated.transpose(0, 2, 1, 3).reshape(B, S, nq * dh))

    # Residual + FFN (on ablated attention output — NOT a linear approximation)
    h_after_attn = clip_residual(h_in, block.post_attention_layernorm(r_ablated))
    r_ffn = block.mlp(block.pre_feedforward_layernorm(h_after_attn))
    h_out = clip_residual(h_after_attn, block.post_feedforward_layernorm(r_ffn))
    return h_out


# ---------------------------------------------------------------------------
# Layer ablation — Llama (2-norm, plain residual adds)
# ---------------------------------------------------------------------------


def _run_llama_layer_ablated(
    model, layer_idx: int, head_idx: int, h_in: mx.array
) -> mx.array:
    """Run Llama layer with head_idx zeroed. Re-runs FFN on ablated residual.

    Llama block structure:
      h_after_attn = h_in + attn_delta          (plain add, no norm on delta)
      h_out        = h_after_attn + ffn_delta   (plain add, no norm on delta)
      post_attention_layernorm is applied to h_after_attn BEFORE the FFN (pre-FFN norm).
    """
    block = model.model.layers[layer_idx]
    attn = block.self_attn
    B, S, _ = h_in.shape
    nq = attn.num_heads
    nkv = attn.num_kv_heads
    dh = attn.head_dim
    n_rep = attn.n_rep  # query heads per KV head

    # Pre-attention norm
    x = block.input_layernorm(h_in)

    # QKV projection — Llama has NO q_norm/k_norm
    q = attn.q_proj(x).reshape(B, S, nq, dh).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)

    # Apply RoPE (uses _apply_rope which calls attn.rope internally)
    q, k = attn._apply_rope(q, k, offset=0)

    # Repeat KV heads to match query heads
    if n_rep > 1:
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    # Build causal mask (sliding window if applicable)
    from chuk_lazarus.models_v2.components.attention.sliding_window import SlidingWindowAttention
    if isinstance(attn, SlidingWindowAttention):
        from chuk_lazarus.models_v2.components.attention.base import create_sliding_window_mask
        mask = create_sliding_window_mask(S, attn.window_size, dtype=h_in.dtype)
    else:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(S).astype(h_in.dtype)

    sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    # sdpa: (B, nq, S, dh)

    # Zero head_idx
    heads = [
        mx.zeros_like(sdpa[:, h : h + 1, :, :]) if h == head_idx else sdpa[:, h : h + 1, :, :]
        for h in range(nq)
    ]
    sdpa_ablated = mx.concatenate(heads, axis=1)  # (B, nq, S, dh)

    r_ablated = attn.o_proj(sdpa_ablated.transpose(0, 2, 1, 3).reshape(B, S, nq * dh))

    # Llama: plain residual add (no norm on the attention delta)
    h_after_attn = h_in + r_ablated

    # Pre-FFN norm applied to full residual state, then plain add
    r_ffn = block.mlp(block.post_attention_layernorm(h_after_attn))
    h_out = h_after_attn + r_ffn
    return h_out


# ---------------------------------------------------------------------------
# Part A: Injection layer scan — c = dot(h[answer_pos], embed(answer))
# ---------------------------------------------------------------------------


def scan_injection_layers(
    runner: CalibRunner, tokenizer, probes: list[dict], first_layer: int
) -> tuple[int, int]:
    """Find the layer where the injection coefficient signal peaks.

    Returns (inject_layer, retrieval_layer) where inject_layer = argmax(avg_c).
    """
    num_layers = runner.num_layers

    print(f"\n{BOLD}Part A: Injection layer scan (layers {first_layer}–{num_layers - 1}){RESET}")
    print(f"  c = dot(h_layer[answer_pos], embed(answer)), avg over {len(probes)} probes")

    # Pre-compute answer token embeddings (outside the layer loop)
    answer_embeds: dict[int, np.ndarray] = {}
    for probe in probes:
        tok = get_answer_first_token(tokenizer, probe["answer"])
        if tok not in answer_embeds:
            e = runner.get_embedding(tok)
            mx.eval(e)
            answer_embeds[tok] = np.array(e.tolist(), dtype=np.float32)

    layer_scores: list[tuple[int, float]] = []

    for layer_idx in range(first_layer, num_layers):
        probe_cs = []
        for probe in probes:
            doc_text = runner.build_doc_prompt(probe["doc"])
            doc_ids = tokenize(tokenizer, doc_text)
            answer_pos = find_answer_position(tokenizer, doc_ids, probe["answer"])
            if answer_pos is None:
                continue

            answer_tok = get_answer_first_token(tokenizer, probe["answer"])
            ids_mx = mx.array(doc_ids, dtype=mx.int32)[None]

            # h leaving layer_idx = h entering layer_idx+1
            # forward_to_layer(stop_layer=L) runs layers 0..L-1, returns h before L
            # We want h AFTER layer_idx, so stop_layer = layer_idx + 1
            h = runner.forward_to_layer(ids_mx, stop_layer=layer_idx + 1)
            h_pos = h[0, answer_pos, :].astype(mx.float32)
            mx.eval(h_pos)

            h_np = np.array(h_pos.tolist(), dtype=np.float32)
            e_np = answer_embeds[answer_tok]
            probe_cs.append(float(np.dot(h_np, e_np)))

        avg_c = float(np.mean(probe_cs)) if probe_cs else 0.0
        layer_scores.append((layer_idx, avg_c))

    # Print table with peak marker
    best_layer = max(layer_scores, key=lambda x: x[1])[0]
    print(f"\n  {'Layer':>6}  {'avg c':>10}")
    for layer_idx, avg_c in layer_scores:
        marker = f"  {GREEN}← peak{RESET}" if layer_idx == best_layer else ""
        print(f"  L{layer_idx:>4}:  {avg_c:>+10.2f}{marker}")

    # best_layer is the index used in forward_to_layer(stop_layer=layer_idx+1),
    # which returns h after layer_idx = h entering layer_idx+1.
    # So c peaks at h entering (best_layer + 1) → inject_layer = best_layer + 1.
    inject_layer = best_layer + 1
    retrieval_layer = best_layer
    print(f"\n  {BOLD}→ inject_layer={inject_layer}, retrieval_layer={retrieval_layer}{RESET}")
    return inject_layer, retrieval_layer


# ---------------------------------------------------------------------------
# Part B: Causal ablation — zero head H, measure P(answer) drop
# ---------------------------------------------------------------------------


def scan_routing_heads(
    runner: CalibRunner, tokenizer, probes: list[dict], retrieval_layer: int
) -> tuple[int, float]:
    """Find the head at retrieval_layer with the largest causal delta on P(answer).

    Returns (best_head, best_score).
    """
    num_heads = runner.num_heads_at(retrieval_layer)

    print(f"\n{BOLD}Part B: Causal ablation at L{retrieval_layer} ({num_heads} heads){RESET}")
    print(f"  delta = P(answer | baseline) - P(answer | head_H_zeroed)")
    print(f"  Probes: {len(probes)}")

    # Baseline: run cloze_prefix prompts — these end just before the answer
    # token, so P(answer_tok) is directly measurable and usually non-trivial.
    baselines: list[tuple[mx.array, int, float]] = []  # (ids, answer_tok, P_baseline)
    print(f"\n  Baseline predictions (cloze format):")
    for probe in probes:
        cloze_ids = tokenize(tokenizer, probe["cloze_prefix"])
        ids_mx = mx.array(cloze_ids, dtype=mx.int32)[None]
        answer_tok = get_answer_first_token(tokenizer, probe["answer"])

        logits = runner.forward_full(ids_mx)
        logits_last = logits[0, -1, :].astype(mx.float32)
        mx.eval(logits_last)
        probs_np = np.array(nn.softmax(logits_last).tolist(), dtype=np.float32)
        top_tok = int(np.argmax(probs_np))
        p_answer = float(probs_np[answer_tok])
        p_top = float(probs_np[top_tok])
        top_str = tokenizer.decode([top_tok])
        print(
            f"    [{probe['answer']:>12}]  "
            f"P(answer)={p_answer:.4f}  "
            f"top={top_str!r}({p_top:.3f})"
        )
        baselines.append((ids_mx, answer_tok, p_answer))

    # Per-head ablation.
    # Score = mean_delta × coverage, where coverage = fraction of probes where
    # delta >= COVERAGE_THRESHOLD.  Pure mean_delta is dominated by large
    # single-probe effects (downstream amplifier heads like L30 H7).
    # Pure coverage misses magnitude.  The product rewards consistent originators.
    COVERAGE_THRESHOLD = 0.05  # min delta to count a probe as "affected"

    print(f"\n  {'Head':>5}  {'mean Δ':>8}  {'cov':>5}  {'score':>8}  per-probe Δ")
    # (head_idx, score, avg_delta, coverage)
    head_results: list[tuple[int, float, float, float]] = []

    for head_idx in range(num_heads):
        deltas = []
        probe_strs = []

        for probe_data, probe in zip(baselines, probes):
            ids_mx, answer_tok, p_base = probe_data

            # h entering retrieval_layer
            h_in = runner.forward_to_layer(ids_mx, stop_layer=retrieval_layer)
            mx.eval(h_in)

            # Layer with head zeroed
            h_ablated = runner.run_layer_ablated(retrieval_layer, head_idx, h_in)
            mx.eval(h_ablated)

            # Continue from retrieval_layer + 1
            logits_abl = runner.forward_from_layer(h_ablated, start_layer=retrieval_layer + 1)
            logits_last = logits_abl[0, -1, :].astype(mx.float32)
            mx.eval(logits_last)
            probs_abl = np.array(nn.softmax(logits_last).tolist(), dtype=np.float32)
            p_abl = float(probs_abl[answer_tok])

            delta = p_base - p_abl
            deltas.append(delta)
            probe_strs.append(f"{p_base:.3f}→{p_abl:.3f}({delta:+.3f})")

        avg_delta = float(np.mean(deltas))
        n_affected = sum(1 for d in deltas if d >= COVERAGE_THRESHOLD)
        coverage = n_affected / len(deltas) if deltas else 0.0
        score = avg_delta * coverage
        head_results.append((head_idx, score, avg_delta, coverage))
        print(
            f"  H{head_idx}:  {avg_delta:>+7.4f}  {coverage:>4.0%}  {score:>+8.4f}  "
            + "  ".join(probe_strs)
        )

    best_head, best_score, best_delta, best_cov = max(head_results, key=lambda x: x[1])
    print(
        f"\n  {BOLD}→ routing_head={best_head} "
        f"(score={best_score:+.4f}, mean Δ={best_delta:+.4f}, coverage={best_cov:.0%}){RESET}"
    )
    return best_head, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def calibrate(runner: CalibRunner, tokenizer, top_layers: int | None = None) -> None:
    from chuk_lazarus.inference.context.arch_config import (
        ArchitectureConfig,
        ArchitectureNotCalibrated,
    )

    config = runner.config
    num_layers = runner.num_layers
    model_type = getattr(config, "model_type", "?").lower()
    hidden_size = getattr(config, "hidden_size", "?")
    num_heads = runner.num_heads_at(0)

    print(
        f"\n{BOLD}Model:{RESET} {model_type}, {num_layers} layers, "
        f"hidden={hidden_size}, {num_heads} query heads/layer"
    )

    try:
        ac = ArchitectureConfig.from_model_config(config)
        print(f"  {GREEN}Already in registry:{RESET} {ac}")
        print(f"  Running to validate…")
    except ArchitectureNotCalibrated:
        print(f"  Not in registry — running discovery")

    first_layer = (num_layers * 2) // 3
    if top_layers is not None:
        first_layer = max(0, num_layers - top_layers)

    # Part A: injection layer
    inject_layer, retrieval_layer = scan_injection_layers(
        runner, tokenizer, PROBES, first_layer
    )

    # Part B: routing head via causal ablation
    routing_head, routing_score = scan_routing_heads(
        runner, tokenizer, PROBES, retrieval_layer
    )

    # Normalise model_type for registry key
    norm_type = model_type
    if norm_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
        norm_type = "gemma"
    elif norm_type in ("mistral", "codellama", "smollm", "llama4"):
        norm_type = "llama"

    print(f"\n{'═' * 60}")
    print(f"{BOLD}Result:{RESET}")
    print(f"  retrieval_layer = {retrieval_layer}")
    print(f"  query_head      = {routing_head}  (consistency score={routing_score:+.4f})")
    print(f"  injection_layer = {inject_layer}")
    print(f"\n{BOLD}Add to arch_config.py:{RESET}")
    print(f"""
  ArchitectureConfig._KNOWN[("{norm_type}", {num_layers})] = ArchitectureConfig(
      retrieval_layer={retrieval_layer},
      query_head={routing_head},
      injection_layer={inject_layer},
  )
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate ArchitectureConfig for Gemma or Llama models"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-1b-it-bf16",
        help="Model ID or local path",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=None,
        help="Scan only the top N layers (default: top third)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_path = _download(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    runner = load_model_runner(args.model)
    calibrate(runner, tokenizer, top_layers=args.top_layers)


if __name__ == "__main__":
    main()
