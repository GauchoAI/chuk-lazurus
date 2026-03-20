#!/usr/bin/env python3
"""
Hierarchical routing — Experiments 1, 2, 3.

Motivation
----------
N=12 test (hspace_routing_n12.py) showed raw H-space cosine fails within
same-template clusters: 8 city facts crowd the correct fact out of rank-1
(Zarkov→Keltara 0.977×, Nexaris→Pyraxis 0.984×). Verb cluster survived
(N=4, less crowding). Per-template PCA is the proposed fix.

Experiment 1 — K-space cluster separation
  Verify K-vectors (256D from L29 H4) form separable city/verb clusters.
  W_K amplifies template differences. If clusters are separable in K-space,
  K-space Q·K can be used for coarse cluster assignment in stage 1.

Experiment 2 — Per-template PCA within city cluster (full-doc queries)
  Fit PCA on the 8 city fact H-vectors.
  Subtract top-k PCs (shared city template component).
  Test whether entity routing in the PCA residual recovers correct ranking.
  Uses full-document queries (no format gap). This is the clean baseline.

Experiment 3 — Per-template PCA format gap survival (bare queries)
  Same PCA as Experiment 2, applied to bare query vectors.
  Key question: is the 7-12° format gap in the template-shared PCs or in
  the entity-discriminating residual?
  If the format gap lives in the top PCs → PCA helps bare-query routing.
  If the format gap lives in the residual → per-template PCA fails.

Usage
-----
    uv run python examples/inference/hierarchical_routing_experiments.py
    uv run python examples/inference/hierarchical_routing_experiments.py --exp 1
    uv run python examples/inference/hierarchical_routing_experiments.py --exp 2 --n-pcs 3
    uv run python examples/inference/hierarchical_routing_experiments.py --exp 3
    uv run python examples/inference/hierarchical_routing_experiments.py --all --n-pcs 3
"""

from __future__ import annotations

import argparse
import json
import math
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
# Fact bank — same as hspace_routing_n12.py
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
    "Q1_zarkov": "What city was Zarkov Industries founded in?",
    "Q2_nexaris": "What city was Nexaris Corporation founded in?",
    "Q11_namath": "What did Joe Namath agree to do?",
    "Q12_marchand": "What did Sylvia Marchand agree to do?",
}

GROUND_TRUTH = {
    "Q1_zarkov": "F1_zarkov",
    "Q2_nexaris": "F2_nexaris",
    "Q11_namath": "F11_namath",
    "Q12_marchand": "F12_marchand",
}

CITY_KEYS = [k for k in FACT_DOCS if k.startswith(("F1_", "F2_", "F3_", "F4_", "F5_", "F6_", "F7_", "F8_"))]
VERB_KEYS = [k for k in FACT_DOCS if k.startswith(("F11_", "F12_", "F13_", "F14_"))]
FACT_KEYS = list(FACT_DOCS.keys())
QUERY_KEYS = list(BARE_QUERIES.keys())
CITY_QUERIES = ["Q1_zarkov", "Q2_nexaris"]
VERB_QUERIES = ["Q11_namath", "Q12_marchand"]

LAYER = 29

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model_id, local_files_only=True,
                                      allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"]))
    except Exception:
        pass
    return Path(snapshot_download(model_id,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"]))


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
    sanitized = {k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
                 for k, v in std.sanitize(raw).items()}
    std.update(tree_unflatten(list(sanitized.items())))
    mx.eval(std.parameters())
    std.eval()

    rs = GemmaResidualStreamForCausalLM(config)
    rs.update(std.parameters())
    mx.eval(rs.parameters())
    rs.eval()

    kv_gen = KVDirectGenerator.from_gemma_rs(rs, config)
    return kv_gen, config, rs


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def build_fact_prompt(text: str) -> str:
    return "<bos><start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n"


def build_query_prompt(question: str) -> str:
    return "<bos><start_of_turn>user\n" + question + "<end_of_turn>\n<start_of_turn>model\n"


def tokenize(tokenizer, text: str) -> mx.array:
    return mx.array(tokenizer.encode(text, add_special_tokens=False), dtype=mx.int32)[None]


def extract_hspace(kv_gen, ids: mx.array) -> np.ndarray:
    """H-space at L29 last position — shape (hidden_size,)."""
    h = kv_gen.prefill_to_layer(ids, target_layer=LAYER)
    h_last = h[0, -1, :].astype(mx.float32)
    mx.eval(h_last)
    return np.array(h_last, dtype=np.float32)


def extract_kspace(kv_gen, rs, ids: mx.array, config) -> np.ndarray:
    """K-vector at L29 H4 last position — shape (head_dim,)."""
    from chuk_lazarus.inference.context.adapters.gemma_adapter import GemmaLayerAdapter

    B, S = ids.shape
    h = kv_gen.prefill_to_layer(ids, target_layer=LAYER)

    block = rs.model.layers[LAYER]
    adapter = GemmaLayerAdapter(block)
    x = adapter.pre_attn_norm(h)
    _, k, _ = adapter.project_qkv(x, B, S, offset=0)
    # k: (1, nkv, S, head_dim) — H4 is kv head 4 but nkv=4 for 4B, use H4%nkv=0
    # For 4B: 8 query heads, 4 KV heads → head_4 maps to kv_head 4//2 = 2
    nkv = config.num_key_value_heads
    nq = config.num_attention_heads
    query_head = 4
    kv_head = query_head * nkv // nq  # GQA mapping
    k_h4 = k[0, kv_head, -1, :].astype(mx.float32)
    mx.eval(k_h4)
    return np.array(k_h4, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def angle_deg(sim: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, sim))))


def routing_table(query_vecs: dict, fact_vecs: dict, ground_truth: dict, label: str) -> dict[str, dict]:
    print(f"\n{BOLD}{label}{RESET}")
    print(f"{'─'*80}")
    print(f"{'Query':<20} {'Correct':<20} {'Sim':>10} {'Angle':>8} {'Best wrong':>10} {'Ratio':>8}  Status")
    print(f"{'─'*80}")
    results = {}
    correct_total = 0
    for qk, qv in query_vecs.items():
        if qk not in ground_truth:
            continue
        cfk = ground_truth[qk]
        sims = {fk: cosine_sim(qv, fv) for fk, fv in fact_vecs.items()}
        correct_sim = sims[cfk]
        wrong = {fk: s for fk, s in sims.items() if fk != cfk}
        best_wrong_fk = max(wrong, key=lambda k: wrong[k])
        best_wrong_sim = wrong[best_wrong_fk]
        ratio = correct_sim / best_wrong_sim if best_wrong_sim > 0 else float("inf")
        hit = correct_sim > best_wrong_sim
        if hit:
            correct_total += 1
        status = f"{GREEN}✓{RESET}" if hit else f"{RED}✗ → {best_wrong_fk}{RESET}"
        print(f"{qk:<20} {cfk:<20} {correct_sim:>10.4f} {angle_deg(correct_sim):>7.2f}° "
              f"{best_wrong_sim:>10.4f} {ratio:>8.3f}×  {status}")
        results[qk] = {"correct_sim": correct_sim, "best_wrong_sim": best_wrong_sim,
                       "ratio": ratio, "hit": hit}
    print(f"{'─'*80}")
    color = GREEN if correct_total == len(results) else (YELLOW if correct_total > 0 else RED)
    print(f"Accuracy: {color}{correct_total}/{len(results)}{RESET}")
    return results


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------


def fit_pca(vectors: np.ndarray, n_components: int):
    """Fit PCA on (N, D) matrix. Returns (components, mean, explained_variance_ratio)."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = (s ** 2).sum()
    evr = (s ** 2) / total_var
    return Vt[:n_components], mean, evr[:n_components]


def project_out(vectors: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Remove top-k PCA components from vectors. Returns projected residuals."""
    centered = vectors - mean
    # Project onto components, then subtract
    coords = centered @ components.T  # (N, k)
    reconstruction = coords @ components  # (N, D)
    return centered - reconstruction  # (N, D) — template component removed


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


# ---------------------------------------------------------------------------
# Experiment 1 — K-space cluster separation
# ---------------------------------------------------------------------------


def experiment_1(kv_gen, rs, config, tokenizer) -> None:
    print(f"\n{'='*80}")
    print(f"{BOLD}Experiment 1 — K-space cluster separation (L29 H4){RESET}")
    print(f"{'='*80}")
    print("Hypothesis: W_K amplifies template structure. City/verb clusters")
    print("should be separable in 256D K-space even if crowded in 2560D H-space.\n")

    k_vecs: dict[str, np.ndarray] = {}
    for key, doc in FACT_DOCS.items():
        ids = tokenize(tokenizer, build_fact_prompt(doc))
        k_vecs[key] = extract_kspace(kv_gen, rs, ids, config)

    # Within-cluster and cross-cluster cosine stats
    def cluster_stats(keys_a: list[str], keys_b: list[str], label: str) -> tuple[float, float]:
        sims = []
        for ka in keys_a:
            for kb in keys_b:
                if ka != kb:
                    sims.append(cosine_sim(k_vecs[ka], k_vecs[kb]))
        if not sims:
            return 0.0, 0.0
        mean_s = np.mean(sims)
        print(f"  {label:<40} n={len(sims):>3}  mean cos={mean_s:.4f} ({angle_deg(mean_s):.2f}°)"
              f"  min={min(sims):.4f}  max={max(sims):.4f}")
        return mean_s, np.std(sims)

    print(f"{CYAN}K-space pairwise cosines:{RESET}")
    city_mean, city_std = cluster_stats(CITY_KEYS, CITY_KEYS, "Within city cluster")
    verb_mean, verb_std = cluster_stats(VERB_KEYS, VERB_KEYS, "Within verb cluster")
    cross_mean, cross_std = cluster_stats(CITY_KEYS, VERB_KEYS, "City vs verb (cross)")

    # Separation ratio
    if cross_mean > 0:
        city_sep = city_mean / cross_mean
        verb_sep = verb_mean / cross_mean
        print(f"\n  City within / cross ratio : {city_sep:.3f}×")
        print(f"  Verb within / cross ratio : {verb_sep:.3f}×")
        if city_mean > cross_mean and verb_mean > cross_mean:
            print(f"\n  {GREEN}Clusters are TIGHTER within than across → K-space separates templates.{RESET}")
            print(f"  K-space coarse assignment is viable for stage 1 of hierarchical routing.")
        else:
            print(f"\n  {RED}Within-cluster ≤ cross-cluster → K-space does NOT separate templates.{RESET}")
            print(f"  Stage 1 K-space cluster assignment is not viable.")

    # Bare query K-vectors against fact clusters
    print(f"\n{CYAN}Bare query K-vectors vs cluster centroids:{RESET}")
    city_centroid = l2_norm(np.stack([k_vecs[k] for k in CITY_KEYS]).mean(axis=0)[None])[0]
    verb_centroid = l2_norm(np.stack([k_vecs[k] for k in VERB_KEYS]).mean(axis=0)[None])[0]

    print(f"  {'Query':<20} {'City sim':>10} {'Verb sim':>10}  Assignment  Correct?")
    for qk, q in BARE_QUERIES.items():
        ids = tokenize(tokenizer, build_query_prompt(q))
        kq = extract_kspace(kv_gen, rs, ids, config)
        cs = cosine_sim(kq, city_centroid)
        vs = cosine_sim(kq, verb_centroid)
        assigned = "city" if cs > vs else "verb"
        expected = "city" if qk in CITY_QUERIES else "verb"
        ok = assigned == expected
        flag = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {qk:<20} {cs:>10.4f} {vs:>10.4f}  {assigned:<8} {flag}")


# ---------------------------------------------------------------------------
# Experiment 2 — Per-template PCA (full-doc queries)
# ---------------------------------------------------------------------------


def experiment_2(kv_gen, tokenizer, n_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (city_components, city_mean) for use in Experiment 3."""
    print(f"\n{'='*80}")
    print(f"{BOLD}Experiment 2 — Per-template PCA (full-doc queries, {n_pcs} PCs removed){RESET}")
    print(f"{'='*80}")
    print("Hypothesis: The top PCs of the city cluster capture shared template variance.")
    print("Removing them exposes entity-discriminating dimensions that raw cosine misses.\n")

    # Extract H-space for all facts (full-doc)
    print(f"{CYAN}Extracting H-space vectors (full-doc)…{RESET}")
    h_fact: dict[str, np.ndarray] = {}
    for key, doc in FACT_DOCS.items():
        ids = tokenize(tokenizer, build_fact_prompt(doc))
        h_fact[key] = extract_hspace(kv_gen, ids)

    # Full-doc city queries (fact prompt used as query — 0° gap baseline)
    print(f"{CYAN}Extracting H-space vectors (full-doc city queries as queries)…{RESET}")
    h_city_query: dict[str, np.ndarray] = {}
    city_q_map = {"Q1_zarkov": "F1_zarkov", "Q2_nexaris": "F2_nexaris"}
    for qk, fk in city_q_map.items():
        # Use the same full-doc prompt as the fact (0° gap — clean baseline)
        ids = tokenize(tokenizer, build_fact_prompt(FACT_DOCS[fk]))
        h_city_query[qk] = extract_hspace(kv_gen, ids)

    # Baseline: raw cosine, city queries vs all city facts
    print()
    routing_table(
        {qk: h_city_query[qk] for qk in city_q_map},
        {fk: h_fact[fk] for fk in CITY_KEYS},
        GROUND_TRUTH,
        "Baseline: raw H-space cosine, full-doc queries vs city facts only"
    )

    # Fit PCA on city fact cluster
    city_matrix = np.stack([h_fact[k] for k in CITY_KEYS])  # (8, 2560)
    components, mean, evr = fit_pca(city_matrix, n_components=min(n_pcs + 4, 8))

    print(f"\n{CYAN}PCA on city cluster (N=8 full-doc facts):{RESET}")
    cum = 0.0
    for i, ev in enumerate(evr):
        cum += ev
        marker = " ← removing" if i < n_pcs else ""
        print(f"  PC{i+1}: {ev*100:.2f}% variance  (cumulative {cum*100:.2f}%){marker}")

    # Project out top n_pcs from facts and queries
    all_fact_h = np.stack([h_fact[k] for k in CITY_KEYS])  # (8, 2560)
    all_query_h = np.stack([h_city_query[qk] for qk in city_q_map.values()])  # (2, 2560)

    fact_residuals = project_out(all_fact_h, components[:n_pcs], mean)
    query_residuals = project_out(all_query_h, components[:n_pcs], mean)

    # Normalise residuals
    fact_residuals_n = l2_norm(fact_residuals)
    query_residuals_n = l2_norm(query_residuals)

    fact_res_dict = {CITY_KEYS[i]: fact_residuals_n[i] for i in range(len(CITY_KEYS))}
    query_res_dict = {list(city_q_map.keys())[i]: query_residuals_n[i] for i in range(len(city_q_map))}

    routing_table(
        query_res_dict,
        fact_res_dict,
        GROUND_TRUTH,
        f"After PCA projection ({n_pcs} PCs removed): full-doc queries vs city facts"
    )

    return components[:n_pcs], mean


# ---------------------------------------------------------------------------
# Experiment 3 — Per-template PCA format gap survival (bare queries)
# ---------------------------------------------------------------------------


def experiment_3(kv_gen, tokenizer, city_components: np.ndarray, city_mean: np.ndarray,
                 n_pcs: int) -> None:
    print(f"\n{'='*80}")
    print(f"{BOLD}Experiment 3 — Per-template PCA format gap survival (bare queries){RESET}")
    print(f"{'='*80}")
    print("Key question: is the 7-12° format gap in the template-shared PCs or in the")
    print("entity-discriminating residual?\n")
    print("If the format gap lives in the top PCs → PCA removes it → bare-query routing works.")
    print("If the format gap lives in the residual → PCA cannot help → per-template PCA fails.\n")

    # Re-extract full-doc fact H-vectors
    print(f"{CYAN}Extracting H-space vectors (full-doc facts)…{RESET}")
    h_fact: dict[str, np.ndarray] = {}
    for key, doc in FACT_DOCS.items():
        ids = tokenize(tokenizer, build_fact_prompt(doc))
        h_fact[key] = extract_hspace(kv_gen, ids)

    # Extract bare query H-vectors
    print(f"{CYAN}Extracting H-space vectors (bare queries)…{RESET}")
    h_query: dict[str, np.ndarray] = {}
    for qk, q in BARE_QUERIES.items():
        ids = tokenize(tokenizer, build_query_prompt(q))
        h_query[qk] = extract_hspace(kv_gen, ids)

    # Diagnose: where does the format gap land?
    print(f"\n{CYAN}Format gap diagnosis — city queries only:{RESET}")
    print("  Measuring angle between bare query and correct full-doc fact,")
    print("  before and after projecting out city cluster PCs.\n")

    city_fact_h = np.stack([h_fact[k] for k in CITY_KEYS])
    city_fact_res = project_out(city_fact_h, city_components, city_mean)
    city_fact_res_n = l2_norm(city_fact_res)
    city_fact_res_dict = {CITY_KEYS[i]: city_fact_res_n[i] for i in range(len(CITY_KEYS))}

    city_bare_qs = {qk: h_query[qk] for qk in CITY_QUERIES}
    city_bare_res = project_out(np.stack(list(city_bare_qs.values())),
                                city_components, city_mean)
    city_bare_res_n = l2_norm(city_bare_res)
    city_bare_res_dict = {CITY_QUERIES[i]: city_bare_res_n[i] for i in range(len(CITY_QUERIES))}

    # Show angle before/after for correct pair
    print(f"  {'Pair':<35} {'Raw angle':>10} {'PCA-res angle':>14}  Change")
    for qk, fk in [("Q1_zarkov", "F1_zarkov"), ("Q2_nexaris", "F2_nexaris")]:
        raw_sim = cosine_sim(h_query[qk], h_fact[fk])
        res_sim = cosine_sim(city_bare_res_dict[qk], city_fact_res_dict[fk])
        raw_ang = angle_deg(raw_sim)
        res_ang = angle_deg(res_sim)
        delta = res_ang - raw_ang
        direction = f"{GREEN}↓ closer{RESET}" if delta < 0 else f"{RED}↑ further{RESET}"
        print(f"  {qk} ↔ {fk:<20} {raw_ang:>9.2f}°  {res_ang:>13.2f}°  {direction} ({delta:+.2f}°)")

    # Full routing test — bare city queries vs all city facts, after PCA
    print()
    routing_table(
        city_bare_res_dict,
        city_fact_res_dict,
        GROUND_TRUTH,
        f"After PCA projection ({n_pcs} PCs removed): bare queries vs city facts"
    )

    # Verb cluster for comparison — PCA doesn't apply here, but check raw still works
    print()
    verb_fact_h = {k: h_fact[k] for k in VERB_KEYS}
    verb_bare_q = {qk: h_query[qk] for qk in VERB_QUERIES}
    routing_table(
        verb_bare_q,
        verb_fact_h,
        GROUND_TRUTH,
        "Verb cluster: raw H-space cosine (no PCA), bare queries vs verb facts"
    )

    # Summary interpretation
    print(f"\n{BOLD}Interpretation:{RESET}")
    city_hits = sum(1 for qk in CITY_QUERIES
                    if cosine_sim(city_bare_res_dict[qk],
                                  city_fact_res_dict[GROUND_TRUTH[qk]]) >
                    max(cosine_sim(city_bare_res_dict[qk], city_fact_res_dict[fk])
                        for fk in CITY_KEYS if fk != GROUND_TRUTH[qk]))
    if city_hits == len(CITY_QUERIES):
        print(f"  {GREEN}City routing RECOVERED after per-template PCA.{RESET}")
        print(f"  The format gap was in the template-shared PCs. PCA removes the gap.")
        print(f"  → Hierarchical routing (K-space cluster + per-template PCA) is viable.")
    elif city_hits > 0:
        print(f"  {YELLOW}Partial recovery ({city_hits}/{len(CITY_QUERIES)} city queries correct).{RESET}")
        print(f"  Format gap is partially in shared PCs. More PCs may help.")
    else:
        print(f"  {RED}City routing NOT recovered. Format gap survives per-template PCA.{RESET}")
        print(f"  The format gap is in the entity-discriminating residual, not the template PCs.")
        print(f"  Per-template PCA cannot bridge bare-query routing. Dead end.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical routing experiments 1-3")
    p.add_argument("--model", default="mlx-community/gemma-3-4b-it-bf16")
    p.add_argument("--exp", type=int, choices=[1, 2, 3], help="Run a single experiment")
    p.add_argument("--all", action="store_true", help="Run all three experiments")
    p.add_argument("--n-pcs", type=int, default=3, help="PCs to remove in Experiments 2 & 3 (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    run_all = args.all or args.exp is None

    print(f"\n{BOLD}Hierarchical Routing — Experiments 1, 2, 3{RESET}")
    print(f"Model : {args.model}")
    print(f"Layer : {LAYER}  |  PCs to remove: {args.n_pcs}\n")

    from transformers import AutoTokenizer

    kv_gen, config, rs = load_model(args.model)
    model_path = _download(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    city_components, city_mean = None, None

    if run_all or args.exp == 1:
        experiment_1(kv_gen, rs, config, tokenizer)

    if run_all or args.exp == 2:
        city_components, city_mean = experiment_2(kv_gen, tokenizer, args.n_pcs)

    if run_all or args.exp == 3:
        if city_components is None:
            # Need to refit PCA — extract facts and fit
            print(f"\n{DIM}Fitting city PCA for Experiment 3…{RESET}")
            h_facts = {}
            for key, doc in FACT_DOCS.items():
                ids = tokenize(tokenizer, build_fact_prompt(doc))
                h_facts[key] = extract_hspace(kv_gen, ids)
            city_matrix = np.stack([h_facts[k] for k in CITY_KEYS])
            city_components, city_mean, _ = fit_pca(city_matrix, n_components=args.n_pcs)
            city_components = city_components[:args.n_pcs]
        experiment_3(kv_gen, tokenizer, city_components, city_mean, args.n_pcs)

    print()


if __name__ == "__main__":
    main()
