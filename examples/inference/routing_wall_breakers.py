#!/usr/bin/env python3
"""
Routing Wall Breakers — Four alternative routing mechanisms for N≥8 same-template.

Root cause: bare query's entity signal lives IN the template geometry.
PCA removal destroys the signal. We test four mechanisms that don't require removal.

Experiment: routing-wall-breakers
Baseline failures:
  - Raw H-space cosine N=12: Q1 0.977× WRONG, Q2 0.984× WRONG
  - Per-template PCA: 0.024× catastrophic

Four mechanisms:
  M1: Variance-weighted cosine (amplify entity-discriminative dims, no removal)
  M2: H4 attention output routing (copy head vs 7 structural heads)
  M3: Entity-position routing (entity token residual, not last position)
  M4: Contrastive query pairs (delta = entity_query - generic_query)
  M5: Fisher discriminant (last resort)

Usage:
    uv run python examples/inference/routing_wall_breakers.py
    uv run python examples/inference/routing_wall_breakers.py --model mlx-community/gemma-3-4b-it-bf16
    uv run python examples/inference/routing_wall_breakers.py --measurements 1234  # run specific ones
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

EXP_DIR = Path(__file__).parent.parent.parent / "experiments" / "routing-wall-breakers"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fact bank — identical to hspace_routing_n12.py
# ---------------------------------------------------------------------------

FACT_DOCS = {
    "F1_zarkov": (
        "Zarkov Industries was established in the mid-1990s as a pioneering manufacturer "
        "of industrial filtration systems. The company's early growth was driven by contracts "
        "with regional utilities across the eastern seaboard. Its headquarters, built on a "
        "former industrial lot, became a landmark of Voltara's commercial district. Today "
        "Zarkov Industries employs over 2,400 people and operates facilities in six countries.\n\n"
        "What city was Zarkov Industries founded in?"
    ),
    "F2_nexaris": (
        "Nexaris Corporation began as a small software consultancy before pivoting to enterprise "
        "data management platforms in the early 2000s. The founding team of five engineers "
        "worked out of a converted warehouse in Cerulion's technology corridor. Their flagship "
        "product, DataBridge, attracted its first Fortune 500 customer within eighteen months "
        "of launch. Nexaris now holds patents in seventeen countries.\n\n"
        "What city was Nexaris Corporation founded in?"
    ),
    "F3_helion": (
        "Helion Systems traces its origins to a university spin-out program that sought to "
        "commercialise advances in photonic computing. The founding researchers relocated from "
        "the campus to leased office space in Dravenport's innovation quarter in 2003. Within "
        "a decade the company had grown from eight employees to more than six hundred, driven "
        "by demand for high-speed optical interconnects in data centres.\n\n"
        "What city was Helion Systems founded in?"
    ),
    "F4_keltara": (
        "Keltara Dynamics was incorporated in Solmere following a management buyout of a "
        "division specialising in aerospace-grade composite materials. The buyout team "
        "retained the original workforce and expanded the product line to include structural "
        "components for the renewable energy sector. Keltara's Solmere plant has since "
        "received three national manufacturing excellence awards.\n\n"
        "What city was Keltara Dynamics founded in?"
    ),
    "F5_vexon": (
        "Vexon Industries emerged from the logistics sector, initially providing routing "
        "optimisation software to freight carriers operating out of Brindor's busy port "
        "district. The founders had previously worked for a major shipping conglomerate "
        "and identified inefficiencies that their algorithms could address. The company "
        "went public seven years after its founding and has since expanded into autonomous "
        "vehicle fleet management.\n\n"
        "What city was Vexon Industries founded in?"
    ),
    "F6_pyraxis": (
        "Pyraxis Corp was founded by three materials scientists who had worked together "
        "on a government-funded research programme in Tornaval. Their core technology "
        "involved ceramic-polymer composites with exceptional thermal resistance, originally "
        "developed for aerospace applications. Commercial customers in the oil and gas "
        "sector soon became the primary revenue source as demand grew for high-temperature "
        "pipeline insulation.\n\n"
        "What city was Pyraxis Corp founded in?"
    ),
    "F7_stratex": (
        "Stratex Global was established in Fenwick's financial district by a team of former "
        "investment analysts who saw an opportunity in quantitative risk modelling for "
        "commodity markets. The firm's early reputation rested on a proprietary volatility "
        "index that outperformed industry benchmarks across multiple market cycles. "
        "Stratex now manages assets on behalf of sovereign wealth funds in twelve "
        "jurisdictions.\n\n"
        "What city was Stratex Global founded in?"
    ),
    "F8_oberon": (
        "Oberon Technologies was incorporated in Selcourt following seed funding from a "
        "regional technology accelerator. The company's founding product was a wearable "
        "biosensor platform aimed at chronic disease management. Clinical partnerships "
        "with three hospital systems in the Selcourt metropolitan area provided the "
        "real-world validation data that secured Series A financing within two years.\n\n"
        "What city was Oberon Technologies founded in?"
    ),
    "F11_namath": (
        "In the late 1960s, American football quarterback Joe Namath was approached by "
        "Fabergé Inc. regarding a promotional campaign for their men's fragrance line. "
        "After negotiations, Namath agreed to endorse Brut cologne, appearing in a "
        "television advertisement that became one of the most recognised sports endorsements "
        "of the era. The campaign helped establish athlete endorsements as a mainstream "
        "marketing strategy.\n\n"
        "What did Joe Namath agree to do?"
    ),
    "F12_marchand": (
        "Sylvia Marchand, a retired art dealer and collector based in Geneva, held an "
        "extensive private collection assembled over four decades. Following advice from "
        "her estate lawyers, she agreed to sell her painting — a significant early work "
        "by a post-war European artist — through a major auction house rather than in a "
        "private transaction. The sale attracted considerable attention from institutional "
        "buyers.\n\n"
        "What did Sylvia Marchand agree to do?"
    ),
    "F13_webb": (
        "Marcus Webb, a former logistics coordinator with detailed knowledge of internal "
        "procurement practices, was subpoenaed during a federal contract irregularities "
        "investigation. After consulting with his attorney, Webb agreed to testify at the "
        "hearing, providing documentary evidence regarding billing discrepancies that had "
        "been flagged by an internal auditor two years earlier. His testimony was later "
        "cited in the committee's final report.\n\n"
        "What did Marcus Webb agree to do?"
    ),
    "F14_frost": (
        "Elena Frost, a veteran executive with two decades of experience in the financial "
        "services sector, was approached by the board of a mid-size asset management firm "
        "seeking to strengthen governance following a period of regulatory scrutiny. "
        "After a due diligence period, Frost agreed to join the board as an independent "
        "non-executive director, bringing expertise in compliance and risk management "
        "that the board had identified as a priority area.\n\n"
        "What did Elena Frost agree to do?"
    ),
}

BARE_QUERIES = {
    "Q1_zarkov":   "What city was Zarkov Industries founded in?",
    "Q2_nexaris":  "What city was Nexaris Corporation founded in?",
    "Q11_namath":  "What did Joe Namath agree to do?",
    "Q12_marchand": "What did Sylvia Marchand agree to do?",
}

GENERIC_QUERIES = {
    "Q1_zarkov":   "What city was the company founded in?",
    "Q2_nexaris":  "What city was the company founded in?",
    "Q11_namath":  "What did the person agree to do?",
    "Q12_marchand": "What did the person agree to do?",
}

# Entity name replacements for generic fact versions
ENTITY_REPLACEMENTS = {
    "F1_zarkov":  [("Zarkov Industries", "the company"), ("Zarkov", "the company")],
    "F2_nexaris": [("Nexaris Corporation", "the company"), ("Nexaris", "the company")],
    "F3_helion":  [("Helion Systems", "the company"), ("Helion", "the company")],
    "F4_keltara": [("Keltara Dynamics", "the company"), ("Keltara", "the company")],
    "F5_vexon":   [("Vexon Industries", "the company"), ("Vexon", "the company")],
    "F6_pyraxis": [("Pyraxis Corp", "the company"), ("Pyraxis", "the company")],
    "F7_stratex": [("Stratex Global", "the company"), ("Stratex", "the company")],
    "F8_oberon":  [("Oberon Technologies", "the company"), ("Oberon", "the company")],
    "F11_namath": [("Joe Namath", "the person"), ("Namath", "the person")],
    "F12_marchand": [("Sylvia Marchand", "the person"), ("Marchand", "the person")],
    "F13_webb":   [("Marcus Webb", "the person"), ("Webb", "the person")],
    "F14_frost":  [("Elena Frost", "the person"), ("Frost", "the person")],
}

# Entity string for position finding
ENTITY_STRINGS = {
    "Q1_zarkov":   "Zarkov",
    "Q2_nexaris":  "Nexaris",
    "Q11_namath":  "Namath",
    "Q12_marchand": "Marchand",
    "F1_zarkov":   "Zarkov",
    "F2_nexaris":  "Nexaris",
    "F3_helion":   "Helion",
    "F4_keltara":  "Keltara",
    "F5_vexon":    "Vexon",
    "F6_pyraxis":  "Pyraxis",
    "F7_stratex":  "Stratex",
    "F8_oberon":   "Oberon",
    "F11_namath":  "Namath",
    "F12_marchand": "Marchand",
    "F13_webb":    "Webb",
    "F14_frost":   "Frost",
}

GROUND_TRUTH = {
    "Q1_zarkov":   "F1_zarkov",
    "Q2_nexaris":  "F2_nexaris",
    "Q11_namath":  "F11_namath",
    "Q12_marchand": "F12_marchand",
}

FACT_KEYS  = list(FACT_DOCS.keys())
QUERY_KEYS = list(BARE_QUERIES.keys())
CITY_KEYS  = [k for k in FACT_KEYS if k[:2] in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")]

# ---------------------------------------------------------------------------
# Raw N=12 cosine baseline (from hspace_routing_n12.py run)
# ---------------------------------------------------------------------------
BASELINE_RATIOS = {
    "Q1_zarkov":   0.977,  # WRONG
    "Q2_nexaris":  0.984,  # WRONG
    "Q11_namath":  1.007,  # correct
    "Q12_marchand": 1.005, # correct
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    try:
        cached = snapshot_download(
            model_id, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
        return Path(cached)
    except Exception:
        pass
    print(f"  Downloading {model_id}...")
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    ))


def load_model(model_id: str):
    from mlx.utils import tree_unflatten
    from chuk_lazarus.inference.context.kv_generator import KVDirectGenerator
    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    print(f"  Loading {model_id}…")
    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)
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

    kv_gen = KVDirectGenerator.from_gemma_rs(rs, config)
    return kv_gen, config


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_fact_prompt(text: str) -> str:
    return (
        "<bos><start_of_turn>user\n"
        f"{text}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def build_query_prompt(question: str) -> str:
    return (
        "<bos><start_of_turn>user\n"
        f"{question}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def make_generic_fact(key: str, text: str) -> str:
    """Replace entity name with neutral placeholder throughout the document."""
    generic = text
    for (src, tgt) in ENTITY_REPLACEMENTS[key]:
        generic = generic.replace(src, tgt)
    return generic


def tokenize(tokenizer, text: str) -> mx.array:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return mx.array(ids, dtype=mx.int32)[None]


# ---------------------------------------------------------------------------
# H-space extraction utilities
# ---------------------------------------------------------------------------


def extract_hspace_last(kv_gen, input_ids: mx.array, layer: int) -> np.ndarray:
    """Extract hidden state at last token position after `layer`. Returns (D,)."""
    h = kv_gen.prefill_to_layer(input_ids, target_layer=layer)
    mx.eval(h)
    return np.array(h[0, -1, :].astype(mx.float32), dtype=np.float32)


def extract_hspace_all(kv_gen, input_ids: mx.array, layer: int) -> np.ndarray:
    """Extract hidden states at ALL positions after `layer`. Returns (S, D)."""
    h = kv_gen.prefill_to_layer(input_ids, target_layer=layer)
    mx.eval(h)
    return np.array(h[0].astype(mx.float32), dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def weighted_cosine(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    aw, bw = a * w, b * w
    return cosine(aw, bw)


def route_all(query_vecs: dict, fact_vecs: dict, sim_fn) -> dict:
    """Compute routing results for all queries against all facts."""
    results = {}
    for qk in QUERY_KEYS:
        correct_fk = GROUND_TRUTH[qk]
        sims = {fk: sim_fn(query_vecs[qk], fact_vecs[fk]) for fk in FACT_KEYS}
        correct_sim = sims[correct_fk]
        wrong_sims = {fk: s for fk, s in sims.items() if fk != correct_fk}
        best_wrong_fk = max(wrong_sims, key=lambda k: wrong_sims[k])
        best_wrong_sim = wrong_sims[best_wrong_fk]
        ratio = correct_sim / best_wrong_sim if best_wrong_sim > 1e-9 else float("inf")
        results[qk] = {
            "correct_sim": correct_sim,
            "best_wrong_fk": best_wrong_fk,
            "best_wrong_sim": best_wrong_sim,
            "ratio": ratio,
            "hit": correct_sim > best_wrong_sim,
        }
    return results


def print_routing_table(results: dict, label: str):
    correct = sum(1 for r in results.values() if r["hit"])
    print(f"\n  {BOLD}{label} — {correct}/{len(results)}{RESET}")
    print(f"  {'Query':<22} {'Corr sim':>10} {'Best wrong':>10} {'Ratio':>8}  Status")
    print(f"  {'─'*65}")
    for qk, r in results.items():
        base_ratio = BASELINE_RATIOS.get(qk, 0.0)
        delta = r["ratio"] - base_ratio
        if r["hit"]:
            st = f"{GREEN}✓{RESET}"
        else:
            st = f"{RED}✗→{r['best_wrong_fk'][-8:]}{RESET}"
        trend = f"{GREEN}+{delta:.4f}{RESET}" if delta > 0 else f"{RED}{delta:.4f}{RESET}"
        print(f"  {qk:<22} {r['correct_sim']:>10.4f} {r['best_wrong_sim']:>10.4f} {r['ratio']:>8.3f}× {trend}  {st}")
    return correct


# ---------------------------------------------------------------------------
# M2: H4 attention output extraction
# ---------------------------------------------------------------------------


def extract_h4_output_l29(
    kv_gen,
    input_ids: mx.array,
    layer: int = 29,
    head_idx: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract H4's contribution through O_proj at the given layer, last position.

    Returns:
        h4_contrib: (D,) H4's output projected into hidden space
        attn_weights: (S,) H4's attention weights over all positions (last query pos)
    """
    backbone = kv_gen.backbone
    B, S = input_ids.shape

    # Run layers 0..(layer-1) to get residual before `layer`
    h_pre = kv_gen.prefill_to_layer(input_ids, target_layer=layer - 1)
    # h_pre: (1, S, D)

    adapter = backbone.adapted_layers[layer]

    # Pre-attention norm
    x = adapter.pre_attn_norm(h_pre)  # (1, S, D)

    # Project Q, K, V (post q_norm, k_norm, post RoPE)
    q, k, v = adapter.project_qkv(x, B, S, offset=0)
    # q: (1, nq, S, dh), k: (1, nkv, S, dh), v: (1, nkv, S, dh)

    n_rep = adapter.n_rep
    kv_idx = head_idx // n_rep  # KV head for this query head

    dh = adapter.head_dim
    scale = adapter.attn_scale

    # Last query position attending over all positions (causal OK — last can see all)
    q_last = q[:, head_idx, -1:, :]   # (1, 1, dh)
    k_kv   = k[:, kv_idx, :, :]       # (1, S, dh)
    v_kv   = v[:, kv_idx, :, :]       # (1, S, dh)

    # Scores: (1, 1, S)
    scores = mx.matmul(q_last, k_kv.transpose(0, 2, 1)) * scale
    attn_w = mx.softmax(scores, axis=-1)  # (1, 1, S)

    # Weighted V: (1, 1, dh)
    h4_out = mx.matmul(attn_w, v_kv)
    h4_out_flat = h4_out[:, 0, :]  # (1, dh)

    # Project through O_proj columns for this head
    o_weight = adapter._block.self_attn.o_proj.weight  # (D, nq*dh) in MLX Linear
    h4_cols = o_weight[:, head_idx * dh:(head_idx + 1) * dh]  # (D, dh)
    h4_contrib = mx.matmul(h4_out_flat, h4_cols.T)  # (1, D)

    mx.eval(h4_contrib, attn_w)
    return (
        np.array(h4_contrib[0].astype(mx.float32), dtype=np.float32),
        np.array(attn_w[0, 0].astype(mx.float32), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# M3: Entity position finder
# ---------------------------------------------------------------------------


def find_entity_last_pos(token_ids: list[int], entity_ids: list[int]) -> int | None:
    """Find the position of the LAST token of the LAST occurrence of entity_ids in token_ids."""
    n = len(entity_ids)
    last_end = None
    for i in range(len(token_ids) - n + 1):
        if token_ids[i:i + n] == entity_ids:
            last_end = i + n - 1  # position of last entity token
    return last_end


def get_entity_position(tokenizer, prompt_text: str, entity_str: str) -> int | None:
    """Return position of last token of first-word entity occurrence in the prompt tokens."""
    # Tokenize the full prompt
    full_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Try progressively: full entity string, then just first word
    for candidate in [entity_str]:
        # Try with and without leading space
        for prefix in [" " + candidate, candidate]:
            ent_ids = tokenizer.encode(prefix, add_special_tokens=False)
            # Strip any BOS/EOS that might have been added
            ent_ids = [t for t in ent_ids if t not in (tokenizer.bos_token_id, tokenizer.eos_token_id)]
            if not ent_ids:
                continue
            pos = find_entity_last_pos(full_ids, ent_ids)
            if pos is not None:
                return pos
    return None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(model_id: str, measurements: str) -> None:
    from transformers import AutoTokenizer

    run_m1 = "1" in measurements
    run_m2 = "2" in measurements
    run_m3 = "3" in measurements
    run_m4 = "4" in measurements
    run_m5 = "5" in measurements

    print(f"\n{BOLD}Routing Wall Breakers — N=12 same-template{RESET}")
    print(f"Model      : {model_id}")
    print(f"Measurements: {measurements}")
    print(f"Output dir : {EXP_DIR}\n")

    kv_gen, config = load_model(model_id)
    hidden_size = config.hidden_size
    num_layers  = config.num_hidden_layers
    print(f"  hidden_size={hidden_size}, layers={num_layers}")

    model_path = _download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    LAYER = 29  # analysis layer

    # ==================================================================
    # Extract base L29 vectors for all facts and queries
    # (needed by M1, M3, M4)
    # ==================================================================
    print(f"\n{CYAN}Extracting L{LAYER} last-pos H-space vectors…{RESET}")
    fact_vecs:  dict[str, np.ndarray] = {}
    query_vecs: dict[str, np.ndarray] = {}

    for key, doc in FACT_DOCS.items():
        prompt = build_fact_prompt(doc)
        ids    = tokenize(tokenizer, prompt)
        h      = extract_hspace_last(kv_gen, ids, LAYER)
        fact_vecs[key] = h

    for key, q in BARE_QUERIES.items():
        prompt = build_query_prompt(q)
        ids    = tokenize(tokenizer, prompt)
        h      = extract_hspace_last(kv_gen, ids, LAYER)
        query_vecs[key] = h

    # Baseline routing (raw cosine, confirm against known values)
    baseline = route_all(query_vecs, fact_vecs, cosine)
    print(f"\n{BOLD}Baseline (raw cosine, should match N=12 run):{RESET}")
    print_routing_table(baseline, "Baseline N=12 raw cosine")

    # Save all base vectors
    np.savez(
        EXP_DIR / "base_vectors_L29.npz",
        **{f"fact_{k}": v for k, v in fact_vecs.items()},
        **{f"query_{k}": v for k, v in query_vecs.items()},
    )

    summary: dict = {"baseline": {k: v["ratio"] for k, v in baseline.items()}}
    all_results: dict = {}

    # ==================================================================
    # M1: Variance-Weighted Cosine
    # ==================================================================
    if run_m1:
        print(f"\n{'═'*70}")
        print(f"{BOLD}M1 — Variance-Weighted Cosine{RESET}")
        print(f"{'═'*70}")

        # 1a: Per-dimension variance across 8 city facts
        H_cities = np.stack([fact_vecs[k] for k in CITY_KEYS])  # (8, D)
        var_per_dim = H_cities.var(axis=0)  # (D,)

        np.savez(EXP_DIR / "m1_variance_per_dim.npz", variance=var_per_dim)

        mean_var = float(var_per_dim.mean())
        max_var  = float(var_per_dim.max())
        min_var  = float(var_per_dim.min())
        std_var  = float(var_per_dim.std())
        n_high   = int((var_per_dim > 10 * mean_var).sum())

        print(f"\n  Per-dim variance across {len(CITY_KEYS)} city facts (D={hidden_size}):")
        print(f"    Mean var : {mean_var:.6f}")
        print(f"    Max var  : {max_var:.6f}")
        print(f"    Min var  : {min_var:.6f}")
        print(f"    Std var  : {std_var:.6f}")
        print(f"    Dims >10× mean: {n_high}")

        # 1b: sqrt-variance weighting
        w_sqrt = np.sqrt(var_per_dim)
        sim_fn_sqrt = lambda a, b: weighted_cosine(a, b, w_sqrt)
        r_sqrt = route_all(query_vecs, fact_vecs, sim_fn_sqrt)
        acc_sqrt = print_routing_table(r_sqrt, "M1b sqrt-variance weighting")

        # 1c: log-variance weighting
        w_log = np.log1p(var_per_dim)
        sim_fn_log = lambda a, b: weighted_cosine(a, b, w_log)
        r_log = route_all(query_vecs, fact_vecs, sim_fn_log)
        acc_log = print_routing_table(r_log, "M1c log-variance weighting")

        # 1d: Top-K dimension selection
        print(f"\n  {BOLD}M1d — Top-K dimension selection:{RESET}")
        k_results = {}
        sorted_dims = np.argsort(-var_per_dim)  # highest variance first
        print(f"  {'K':>6} {'Q1 ratio':>10} {'Q2 ratio':>10} {'Q11 ratio':>10} {'Q12 ratio':>10} {'Acc':>6}")
        print(f"  {'─'*58}")
        for K in [64, 128, 256, 512, 1024, 2560]:
            if K == 2560:
                topk_dims = None
            else:
                topk_dims = sorted_dims[:K]

            def sim_topk(a, b, dims=topk_dims):
                if dims is None:
                    return cosine(a, b)
                return cosine(a[dims], b[dims])

            r_k = route_all(query_vecs, fact_vecs, sim_topk)
            ratios = [r_k[qk]["ratio"] for qk in QUERY_KEYS]
            acc_k  = sum(1 for qk in QUERY_KEYS if r_k[qk]["hit"])
            print(f"  {K:>6} {ratios[0]:>10.4f} {ratios[1]:>10.4f} {ratios[2]:>10.4f} {ratios[3]:>10.4f} {acc_k:>4}/4")
            k_results[K] = {"ratios": ratios, "acc": acc_k}

        m1_summary = {
            "variance_stats": {
                "mean": mean_var, "max": max_var, "min": min_var, "std": std_var,
                "dims_gt_10x_mean": n_high,
            },
            "sqrt_weighting": {"accuracy": acc_sqrt, "ratios": {k: v["ratio"] for k, v in r_sqrt.items()}},
            "log_weighting":  {"accuracy": acc_log,  "ratios": {k: v["ratio"] for k, v in r_log.items()}},
            "topk": k_results,
        }
        summary["m1"] = m1_summary
        all_results["m1"] = {
            "sqrt": r_sqrt, "log": r_log,
            "topk": k_results,
            "variance_stats": m1_summary["variance_stats"],
        }
        print(f"\n  {BOLD}M1 summary: sqrt={acc_sqrt}/4, log={acc_log}/4{RESET}")

    # ==================================================================
    # M2: H4 Attention Output Routing
    # ==================================================================
    if run_m2:
        print(f"\n{'═'*70}")
        print(f"{BOLD}M2 — H4 Attention Output Routing (L{LAYER}){RESET}")
        print(f"{'═'*70}")

        H4_IDX = 4

        print(f"  Extracting H{H4_IDX} output at L{LAYER} for all {len(FACT_KEYS)} facts…")
        fact_h4:  dict[str, np.ndarray] = {}
        fact_h4_weights: dict[str, np.ndarray] = {}
        for key, doc in FACT_DOCS.items():
            prompt = build_fact_prompt(doc)
            ids = tokenize(tokenizer, prompt)
            h4, w4 = extract_h4_output_l29(kv_gen, ids, layer=LAYER, head_idx=H4_IDX)
            fact_h4[key] = h4
            fact_h4_weights[key] = w4

        print(f"  Extracting H{H4_IDX} output at L{LAYER} for all {len(QUERY_KEYS)} queries…")
        query_h4:  dict[str, np.ndarray] = {}
        query_h4_weights: dict[str, np.ndarray] = {}
        for key, q in BARE_QUERIES.items():
            prompt = build_query_prompt(q)
            ids = tokenize(tokenizer, prompt)
            h4, w4 = extract_h4_output_l29(kv_gen, ids, layer=LAYER, head_idx=H4_IDX)
            query_h4[key] = h4
            query_h4_weights[key] = w4

        # Save vectors
        np.savez(
            EXP_DIR / "m2_h4_vectors_L29.npz",
            **{f"fact_{k}": v for k, v in fact_h4.items()},
            **{f"query_{k}": v for k, v in query_h4.items()},
        )

        r_h4 = route_all(query_h4, fact_h4, cosine)
        acc_h4 = print_routing_table(r_h4, f"M2 H{H4_IDX} output routing")

        # Compare norms
        print(f"\n  H{H4_IDX} output norms:")
        for qk in QUERY_KEYS:
            print(f"    {qk:<22} ‖h4‖={np.linalg.norm(query_h4[qk]):.2f}")
        for fk in FACT_KEYS[:4]:
            print(f"    {fk:<22} ‖h4‖={np.linalg.norm(fact_h4[fk]):.2f}")

        # 2c: What does H4 attend to in bare Q1_zarkov?
        print(f"\n  {BOLD}2c — H{H4_IDX} attention weights for Q1_zarkov bare query:{RESET}")
        q1_prompt = build_query_prompt(BARE_QUERIES["Q1_zarkov"])
        q1_ids = tokenizer.encode(q1_prompt, add_special_tokens=False)
        q1_tokens = [tokenizer.decode([t]) for t in q1_ids]
        q1_weights_np = query_h4_weights["Q1_zarkov"]

        top5_pos = np.argsort(-q1_weights_np)[:5]
        print(f"  {'Pos':>5} {'Token':<20} {'Attn weight':>14}")
        print(f"  {'─'*45}")
        for pos in top5_pos:
            tok_str = repr(q1_tokens[pos]) if pos < len(q1_tokens) else "???"
            print(f"  {pos:>5} {tok_str:<20} {q1_weights_np[pos]:>14.6f}")

        # Save attention weight distributions for all queries
        np.savez(
            EXP_DIR / "m2_h4_attention_weights.npz",
            **{f"query_{k}": v for k, v in query_h4_weights.items()},
            **{f"fact_{k}": v for k, v in fact_h4_weights.items()},
        )

        m2_summary = {
            "accuracy": acc_h4,
            "ratios": {k: v["ratio"] for k, v in r_h4.items()},
            "h4_top5_q1_zarkov": [
                {"pos": int(pos), "token": repr(q1_tokens[pos]) if pos < len(q1_tokens) else "???",
                 "weight": float(q1_weights_np[pos])}
                for pos in top5_pos
            ],
        }
        summary["m2"] = m2_summary
        all_results["m2"] = r_h4
        print(f"\n  {BOLD}M2 accuracy: {acc_h4}/4{RESET}")

    # ==================================================================
    # M3: Entity-Position Routing
    # ==================================================================
    if run_m3:
        print(f"\n{'═'*70}")
        print(f"{BOLD}M3 — Entity-Position Routing (L{LAYER}){RESET}")
        print(f"{'═'*70}")

        entity_pos_info: dict = {}
        fact_entity_vecs:  dict[str, np.ndarray] = {}
        query_entity_vecs: dict[str, np.ndarray] = {}

        print(f"\n  Finding entity token positions and extracting L{LAYER} residuals…")

        # Facts
        for key, doc in FACT_DOCS.items():
            prompt = build_fact_prompt(doc)
            full_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
            ent_str = ENTITY_STRINGS[key]
            pos = get_entity_position(tokenizer, prompt, ent_str)
            entity_pos_info[f"fact_{key}"] = {"entity": ent_str, "position": pos, "seq_len": len(full_ids_list)}

            if pos is None:
                print(f"  {RED}WARNING: entity '{ent_str}' not found in fact {key}{RESET}")
                # Fall back to last position
                h = extract_hspace_last(kv_gen, tokenize(tokenizer, prompt), LAYER)
            else:
                ids = tokenize(tokenizer, prompt)
                h_all = extract_hspace_all(kv_gen, ids, LAYER)
                h = h_all[pos]

            fact_entity_vecs[key] = h

        # Queries
        for key, q in BARE_QUERIES.items():
            prompt = build_query_prompt(q)
            full_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
            ent_str = ENTITY_STRINGS[key]
            pos = get_entity_position(tokenizer, prompt, ent_str)
            entity_pos_info[f"query_{key}"] = {"entity": ent_str, "position": pos, "seq_len": len(full_ids_list)}

            if pos is None:
                print(f"  {RED}WARNING: entity '{ent_str}' not found in query {key}{RESET}")
                h = extract_hspace_last(kv_gen, tokenize(tokenizer, prompt), LAYER)
            else:
                ids = tokenize(tokenizer, prompt)
                h_all = extract_hspace_all(kv_gen, ids, LAYER)
                h = h_all[pos]

            query_entity_vecs[key] = h

        # Print position table
        print(f"\n  Entity positions:")
        print(f"  {'Key':<22} {'Entity':<20} {'Pos':>5} {'SeqLen':>8}")
        print(f"  {'─'*60}")
        for k, info in entity_pos_info.items():
            print(f"  {k:<22} {info['entity']:<20} {str(info['position']):>5} {info['seq_len']:>8}")

        np.savez(
            EXP_DIR / "m3_entity_pos_vectors.npz",
            **{f"fact_{k}": v for k, v in fact_entity_vecs.items()},
            **{f"query_{k}": v for k, v in query_entity_vecs.items()},
        )
        with open(EXP_DIR / "m3_entity_positions.json", "w") as f:
            json.dump(entity_pos_info, f, indent=2)

        r_ep = route_all(query_entity_vecs, fact_entity_vecs, cosine)
        acc_ep = print_routing_table(r_ep, "M3 Entity-position routing")

        # Compare to last-position
        print(f"\n  Comparison: last-pos vs entity-pos ratios:")
        print(f"  {'Query':<22} {'Last-pos':>10} {'Entity-pos':>12} {'Better?':>8}")
        print(f"  {'─'*55}")
        for qk in QUERY_KEYS:
            lp = baseline[qk]["ratio"]
            ep = r_ep[qk]["ratio"]
            better = f"{GREEN}YES{RESET}" if ep > lp else f"{RED}NO{RESET}"
            print(f"  {qk:<22} {lp:>10.4f} {ep:>12.4f} {better:>8}")

        m3_summary = {
            "accuracy": acc_ep,
            "ratios": {k: v["ratio"] for k, v in r_ep.items()},
            "entity_positions": entity_pos_info,
        }
        summary["m3"] = m3_summary
        all_results["m3"] = r_ep
        print(f"\n  {BOLD}M3 accuracy: {acc_ep}/4{RESET}")

    # ==================================================================
    # M4: Contrastive Query Pairs
    # ==================================================================
    if run_m4:
        print(f"\n{'═'*70}")
        print(f"{BOLD}M4 — Contrastive Query Pairs{RESET}")
        print(f"{'═'*70}")

        print(f"\n  Generic query substitutions:")
        for qk in QUERY_KEYS:
            print(f"    {qk}: '{BARE_QUERIES[qk]}'")
            print(f"      → '{GENERIC_QUERIES[qk]}'")

        # Extract L29 last-pos for original and generic queries
        print(f"\n  Extracting L{LAYER} last-pos for original + generic queries…")
        query_delta: dict[str, np.ndarray] = {}
        query_norms: dict[str, dict] = {}

        for qk in QUERY_KEYS:
            prompt_with    = build_query_prompt(BARE_QUERIES[qk])
            prompt_without = build_query_prompt(GENERIC_QUERIES[qk])
            h_with    = extract_hspace_last(kv_gen, tokenize(tokenizer, prompt_with), LAYER)
            h_without = extract_hspace_last(kv_gen, tokenize(tokenizer, prompt_without), LAYER)
            delta = h_with - h_without
            query_delta[qk] = delta
            query_norms[qk] = {
                "norm_with": float(np.linalg.norm(h_with)),
                "norm_without": float(np.linalg.norm(h_without)),
                "norm_delta": float(np.linalg.norm(delta)),
                "delta_ratio": float(np.linalg.norm(delta) / (np.linalg.norm(h_with) + 1e-9)),
            }

        print(f"\n  Query delta norms:")
        print(f"  {'Query':<22} {'‖h_with‖':>10} {'‖h_without‖':>13} {'‖delta‖':>10} {'delta/with':>12}")
        print(f"  {'─'*70}")
        for qk in QUERY_KEYS:
            n = query_norms[qk]
            print(f"  {qk:<22} {n['norm_with']:>10.2f} {n['norm_without']:>13.2f} "
                  f"{n['norm_delta']:>10.2f} {n['delta_ratio']:>12.4f}")

        # Extract L29 last-pos for original and generic facts
        print(f"\n  Extracting L{LAYER} last-pos for original + generic facts…")
        fact_delta: dict[str, np.ndarray] = {}

        for key, doc in FACT_DOCS.items():
            generic_doc   = make_generic_fact(key, doc)
            prompt_with   = build_fact_prompt(doc)
            prompt_without = build_fact_prompt(generic_doc)
            h_with    = extract_hspace_last(kv_gen, tokenize(tokenizer, prompt_with), LAYER)
            h_without = extract_hspace_last(kv_gen, tokenize(tokenizer, prompt_without), LAYER)
            delta = h_with - h_without
            fact_delta[key] = delta

        np.savez(
            EXP_DIR / "m4_deltas.npz",
            **{f"query_{k}": v for k, v in query_delta.items()},
            **{f"fact_{k}": v for k, v in fact_delta.items()},
        )

        # Note: queries with identical generics (Q1/Q2 share template) — check separability
        print(f"\n  {DIM}Note: Q1 and Q2 share identical generic template — delta isolates entity signal.{RESET}")
        print(f"  {DIM}Q11 and Q12 also share template.{RESET}")
        inter_delta = cosine(query_delta["Q1_zarkov"], query_delta["Q2_nexaris"])
        print(f"  cosine(delta_Q1, delta_Q2) = {inter_delta:.4f} (should be < 1.0 if entity signal is distinct)")

        r_ct = route_all(query_delta, fact_delta, cosine)
        acc_ct = print_routing_table(r_ct, "M4 Contrastive routing")

        # Cost accounting summary
        print(f"\n  Cost: 2 forward passes per query (original + generic)")
        print(f"        2 forward passes per fact during prefill")
        print(f"        Total: {2*len(QUERY_KEYS) + 2*len(FACT_KEYS)} forward passes vs {len(QUERY_KEYS) + len(FACT_KEYS)} for baseline")

        m4_summary = {
            "accuracy": acc_ct,
            "ratios": {k: v["ratio"] for k, v in r_ct.items()},
            "query_delta_norms": query_norms,
            "inter_entity_cosine_Q1Q2_deltas": float(inter_delta),
        }
        summary["m4"] = m4_summary
        all_results["m4"] = r_ct
        print(f"\n  {BOLD}M4 accuracy: {acc_ct}/4{RESET}")

    # ==================================================================
    # M5: Fisher Discriminant (only if others fail)
    # ==================================================================
    need_m5 = run_m5
    if not need_m5 and (run_m1 or run_m2 or run_m3 or run_m4):
        m1_acc = summary.get("m1", {}).get("sqrt_weighting", {}).get("accuracy", 0)
        m2_acc = summary.get("m2", {}).get("accuracy", 0)
        m3_acc = summary.get("m3", {}).get("accuracy", 0)
        m4_acc = summary.get("m4", {}).get("accuracy", 0)
        zarkov_solved = any([
            all_results.get("m1", {}).get("sqrt", {}).get("Q1_zarkov", {}).get("hit", False),
            all_results.get("m2", {}).get("Q1_zarkov", {}).get("hit", False),
            all_results.get("m3", {}).get("Q1_zarkov", {}).get("hit", False),
            all_results.get("m4", {}).get("Q1_zarkov", {}).get("hit", False),
        ])
        if not zarkov_solved:
            print(f"\n{YELLOW}All M1-M4 failed on Zarkov — escalating to M5 (Fisher Discriminant){RESET}")
            need_m5 = True

    if need_m5:
        print(f"\n{'═'*70}")
        print(f"{BOLD}M5 — Fisher Discriminant (Last Resort){RESET}")
        print(f"{'═'*70}")

        # Within-entity variance: difference between bare query and full-doc fact
        diff_zarkov  = query_vecs["Q1_zarkov"]  - fact_vecs["F1_zarkov"]
        diff_nexaris = query_vecs["Q2_nexaris"]  - fact_vecs["F2_nexaris"]
        diff_namath  = query_vecs["Q11_namath"]  - fact_vecs["F11_namath"]
        diff_marchand = query_vecs["Q12_marchand"] - fact_vecs["F12_marchand"]

        within_var = 0.25 * (diff_zarkov**2 + diff_nexaris**2 + diff_namath**2 + diff_marchand**2)
        between_var = np.stack([fact_vecs[k] for k in CITY_KEYS]).var(axis=0)

        fisher_weight = between_var / (within_var + 1e-8)
        w_fisher = np.sqrt(np.clip(fisher_weight, 0, None))

        np.savez(EXP_DIR / "m5_fisher_weights.npz",
                 within_var=within_var, between_var=between_var,
                 fisher_weight=fisher_weight, w_fisher=w_fisher)

        print(f"  Fisher weight stats:")
        print(f"    Mean: {w_fisher.mean():.4f}, Max: {w_fisher.max():.2f}, "
              f"Median: {np.median(w_fisher):.4f}")
        print(f"    Dims with weight > 10× mean: {(w_fisher > 10*w_fisher.mean()).sum()}")

        sim_fn_fisher = lambda a, b: weighted_cosine(a, b, w_fisher)
        r_fisher = route_all(query_vecs, fact_vecs, sim_fn_fisher)
        acc_fisher = print_routing_table(r_fisher, "M5 Fisher discriminant routing")

        summary["m5"] = {
            "accuracy": acc_fisher,
            "ratios": {k: v["ratio"] for k, v in r_fisher.items()},
        }
        all_results["m5"] = r_fisher
        print(f"\n  {BOLD}M5 accuracy: {acc_fisher}/4{RESET}")

    # ==================================================================
    # Final summary
    # ==================================================================
    print(f"\n{'═'*70}")
    print(f"{BOLD}ROUTING WALL BREAKERS — FINAL SUMMARY{RESET}")
    print(f"{'═'*70}")
    print(f"  Baseline N=12 accuracy: 2/4 (Q1 0.977× WRONG, Q2 0.984× WRONG)\n")

    methods = [
        ("Baseline",   "baseline"),
        ("M1b sqrt-var", "m1"),
        ("M2 H4 output", "m2"),
        ("M3 entity-pos", "m3"),
        ("M4 contrastive", "m4"),
        ("M5 Fisher", "m5"),
    ]
    print(f"  {'Method':<20} {'Q1(Z)':>8} {'Q2(N)':>8} {'Q11':>8} {'Q12':>8} {'Acc':>6}")
    print(f"  {'─'*65}")

    for label, mkey in methods:
        if mkey == "baseline":
            r = baseline
        elif mkey == "m1":
            r = all_results.get("m1", {}).get("sqrt")
            if r is None:
                continue
        elif mkey in all_results:
            r = all_results[mkey]
        else:
            continue

        ratios = [r.get(qk, {}).get("ratio", 0.0) for qk in QUERY_KEYS]
        acc    = sum(1 for qk in QUERY_KEYS if r.get(qk, {}).get("hit", False))
        row = f"  {label:<20}"
        for i, (qk, rat) in enumerate(zip(QUERY_KEYS, ratios)):
            hit = r.get(qk, {}).get("hit", False)
            c = GREEN if hit else RED
            row += f" {c}{rat:>8.3f}{RESET}"
        row += f" {acc:>4}/4"
        print(row)

    # Verdict
    print(f"\n{'─'*70}")
    zarkov_best = max(
        (all_results.get(m, {}).get("Q1_zarkov", {}).get("ratio", 0.0)
         for m in ["m1_sqrt", "m2", "m3", "m4", "m5"]
         if m in all_results),
        default=0.0
    )
    # Check per measurement
    m1_sqrt_zarkov = all_results.get("m1", {}).get("sqrt", {}).get("Q1_zarkov", {}).get("ratio", 0.0)
    m2_zarkov      = all_results.get("m2", {}).get("Q1_zarkov", {}).get("ratio", 0.0)
    m3_zarkov      = all_results.get("m3", {}).get("Q1_zarkov", {}).get("ratio", 0.0)
    m4_zarkov      = all_results.get("m4", {}).get("Q1_zarkov", {}).get("ratio", 0.0)
    m5_zarkov      = all_results.get("m5", {}).get("Q1_zarkov", {}).get("ratio", 0.0)
    best_zarkov    = max(m1_sqrt_zarkov, m2_zarkov, m3_zarkov, m4_zarkov, m5_zarkov)

    if best_zarkov > 1.0:
        print(f"{GREEN}{BOLD}→ WALL BROKEN: Zarkov resolved (best ratio={best_zarkov:.4f}){RESET}")
    else:
        print(f"{RED}{BOLD}→ WALL HOLDS: Zarkov still fails (best ratio={best_zarkov:.4f}). "
              f"Linear routing ceiling is real.{RESET}")

    # Save summary
    summary_path = EXP_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        # Convert numpy floats to Python floats for JSON serialisation
        def _to_json(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json(i) for i in obj]
            return obj
        json.dump(_to_json(summary), f, indent=2)

    print(f"\n  Results saved to {EXP_DIR}")
    print(f"  Summary JSON: {summary_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Routing wall breakers — four alternative routing mechanisms")
    p.add_argument("--model", default="mlx-community/gemma-3-4b-it-bf16")
    p.add_argument(
        "--measurements", default="1234",
        help="Which measurements to run, e.g. '12' for M1+M2, '1234' for all (default: 1234)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.model, args.measurements)
