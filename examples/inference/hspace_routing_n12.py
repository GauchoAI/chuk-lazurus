#!/usr/bin/env python3
"""
H-space routing experiment — N=12 margin survival test.

Experiment 8a42e949 follow-on: tests whether 1.005× H-space cosine margins
hold when the candidate set grows from 4 to 12 facts.

Setup
-----
  - 8 city-founding facts  (F1–F8)
  - 4 "agreed to do" facts (F11–F14)
  - 4 bare queries with entity name in question: Q1, Q2, Q11, Q12

Each fact has a full-document context prompt (multi-sentence passage + question).
Each query is a bare question (no document context).

Both are encoded at L29 last-token position — the routing vector is
the hidden state at the "model is about to answer" position.

H4 at L29 is a copy head: it attends to entity name tokens and copies
that signal into the last position residual, regardless of preceding context.
This is why bare-query (entity in question) ≈ full-doc (entity in doc) at L29.

Usage
-----
    uv run python examples/inference/hspace_routing_n12.py
    uv run python examples/inference/hspace_routing_n12.py --model mlx-community/gemma-3-4b-it-bf16
    uv run python examples/inference/hspace_routing_n12.py --layer 28
    uv run python examples/inference/hspace_routing_n12.py --show-matrix
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
# Fact bank — 8 city facts + 4 "agreed to do" facts
# ---------------------------------------------------------------------------

# Full-document fact prompts.
# Format: multi-sentence passage about the entity, then the question.
# The model's last-position residual at L29 encodes the entity from
# both the passage and the question.
FACT_DOCS = {
    # City founding facts
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
    # "Agreed to do" facts
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

# Bare queries — entity name is explicitly in the question.
# ~17-19 tokens in Gemma's vocabulary.
BARE_QUERIES = {
    "Q1_zarkov": "What city was Zarkov Industries founded in?",
    "Q2_nexaris": "What city was Nexaris Corporation founded in?",
    "Q11_namath": "What did Joe Namath agree to do?",
    "Q12_marchand": "What did Sylvia Marchand agree to do?",
}

# Maps each bare query to its correct fact key
GROUND_TRUTH = {
    "Q1_zarkov": "F1_zarkov",
    "Q2_nexaris": "F2_nexaris",
    "Q11_namath": "F11_namath",
    "Q12_marchand": "F12_marchand",
}

# Answer tokens for reference (used in generation check, not routing)
ANSWERS = {
    "F1_zarkov": "Voltara",
    "F2_nexaris": "Cerulion",
    "F3_helion": "Dravenport",
    "F4_keltara": "Solmere",
    "F5_vexon": "Brindor",
    "F6_pyraxis": "Tornaval",
    "F7_stratex": "Fenwick",
    "F8_oberon": "Selcourt",
    "F11_namath": "endorse",
    "F12_marchand": "sell",
    "F13_webb": "testify",
    "F14_frost": "join",
}

FACT_KEYS = list(FACT_DOCS.keys())
QUERY_KEYS = list(BARE_QUERIES.keys())

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
# Tokenisation helpers (Gemma instruct template)
# ---------------------------------------------------------------------------


def build_fact_prompt(text: str) -> str:
    """Wrap a fact document + question in Gemma's instruct template."""
    return (
        "<bos><start_of_turn>user\n"
        f"{text}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def build_query_prompt(question: str) -> str:
    """Bare question in Gemma's instruct template (no document context)."""
    return (
        "<bos><start_of_turn>user\n"
        f"{question}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def tokenize(tokenizer, text: str) -> mx.array:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return mx.array(ids, dtype=mx.int32)[None]  # (1, S)


# ---------------------------------------------------------------------------
# H-space extraction
# ---------------------------------------------------------------------------


def extract_hspace(kv_gen, input_ids: mx.array, layer: int) -> np.ndarray:
    """Extract hidden state at last token position after `layer`.

    Returns shape (hidden_size,) as float32 numpy array.
    """
    h = kv_gen.prefill_to_layer(input_ids, target_layer=layer)
    mx.eval(h)
    # h: (1, S, hidden_size) — take last position
    h_last = h[0, -1, :].astype(mx.float32)  # cast bfloat16 → float32 before numpy
    mx.eval(h_last)
    return np.array(h_last, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def angle_deg(sim: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, sim))))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(model_id: str, layer: int, show_matrix: bool) -> None:
    print(f"\n{BOLD}H-space Routing — N=12 Margin Survival Test{RESET}")
    print(f"Model : {model_id}")
    print(f"Layer : {layer}")
    print(f"Facts : {len(FACT_KEYS)} ({sum(1 for k in FACT_KEYS if 'zarkov' in k or 'nexaris' in k or 'helion' in k or 'keltara' in k or 'vexon' in k or 'pyraxis' in k or 'stratex' in k or 'oberon' in k)} city, "
          f"{sum(1 for k in FACT_KEYS if k.startswith('F1'))} entity-type facts)")
    print(f"Queries: {len(QUERY_KEYS)} bare entity-explicit\n")

    from transformers import AutoTokenizer

    kv_gen, config = load_model(model_id)
    hidden_size = config.hidden_size
    print(f"  hidden_size = {hidden_size}, num_layers = {config.num_hidden_layers}")

    model_path = _download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # --- Extract fact H-space vectors ---
    print(f"\n{CYAN}Extracting fact H-space vectors at L{layer}…{RESET}")
    fact_vecs: dict[str, np.ndarray] = {}
    for key, doc in FACT_DOCS.items():
        prompt = build_fact_prompt(doc)
        ids = tokenize(tokenizer, prompt)
        h = extract_hspace(kv_gen, ids, layer)
        fact_vecs[key] = h
        n_tok = ids.shape[1]
        print(f"  {key:<20} {n_tok:>4} tokens  ‖h‖={np.linalg.norm(h):.2f}")

    # --- Extract bare query H-space vectors ---
    print(f"\n{CYAN}Extracting bare query H-space vectors at L{layer}…{RESET}")
    query_vecs: dict[str, np.ndarray] = {}
    for key, q in BARE_QUERIES.items():
        prompt = build_query_prompt(q)
        ids = tokenize(tokenizer, prompt)
        h = extract_hspace(kv_gen, ids, layer)
        query_vecs[key] = h
        n_tok = ids.shape[1]
        print(f"  {key:<20} {n_tok:>4} tokens  ‖h‖={np.linalg.norm(h):.2f}")

    # --- Compute full similarity matrix ---
    print(f"\n{CYAN}Computing cosine similarities (2560D H-space)…{RESET}")
    sim_matrix: dict[str, dict[str, float]] = {}
    for qk, qv in query_vecs.items():
        sim_matrix[qk] = {}
        for fk, fv in fact_vecs.items():
            sim_matrix[qk][fk] = cosine_sim(qv, fv)

    # --- Routing results table ---
    print(f"\n{BOLD}{'─'*80}{RESET}")
    print(f"{BOLD}Routing Results: bare query vs all {len(FACT_KEYS)} fact candidates{RESET}")
    print(f"{'─'*80}")
    print(f"{'Query':<20} {'Correct fact':<20} {'Correct sim':>12} {'Angle':>8} {'Best wrong':>12} {'Ratio':>8}  Status")
    print(f"{'─'*80}")

    correct = 0
    results = []
    for qk in QUERY_KEYS:
        correct_fk = GROUND_TRUTH[qk]
        sims = sim_matrix[qk]
        correct_sim = sims[correct_fk]
        correct_angle = angle_deg(correct_sim)

        # Best wrong candidate
        wrong_sims = {fk: s for fk, s in sims.items() if fk != correct_fk}
        best_wrong_fk = max(wrong_sims, key=lambda k: wrong_sims[k])
        best_wrong_sim = wrong_sims[best_wrong_fk]

        ratio = correct_sim / best_wrong_sim if best_wrong_sim > 0 else float("inf")
        hit = correct_sim > best_wrong_sim
        if hit:
            correct += 1
            status = f"{GREEN}✓ CORRECT{RESET}"
        else:
            status = f"{RED}✗ WRONG → {best_wrong_fk}{RESET}"

        print(
            f"{qk:<20} {correct_fk:<20} {correct_sim:>12.4f} {correct_angle:>7.2f}°"
            f" {best_wrong_sim:>12.4f} {ratio:>8.3f}×  {status}"
        )
        results.append({
            "query": qk,
            "correct": correct_fk,
            "correct_sim": correct_sim,
            "angle_deg": correct_angle,
            "best_wrong_fk": best_wrong_fk,
            "best_wrong_sim": best_wrong_sim,
            "ratio": ratio,
            "hit": hit,
        })

    print(f"{'─'*80}")
    acc_str = f"{correct}/{len(QUERY_KEYS)}"
    color = GREEN if correct == len(QUERY_KEYS) else (YELLOW if correct >= len(QUERY_KEYS) // 2 else RED)
    print(f"{BOLD}Accuracy: {color}{acc_str}{RESET}")

    # --- Margin summary ---
    ratios = [r["ratio"] for r in results]
    print(f"\n{BOLD}Margin summary:{RESET}")
    print(f"  Min ratio : {min(ratios):.4f}×")
    print(f"  Max ratio : {max(ratios):.4f}×")
    print(f"  Mean ratio: {sum(ratios)/len(ratios):.4f}×")

    # Compare to N=4 baseline from experiment 8a42e949
    print(f"\n{DIM}N=4 baseline (experiment 8a42e949): 4/4, margins 1.005–1.006×{RESET}")
    margin_held = all(r["ratio"] >= 1.001 for r in results)
    if margin_held and correct == len(QUERY_KEYS):
        print(f"{GREEN}{BOLD}→ Margins held at N=12. H-space stage-2 routing is viable.{RESET}")
    elif correct == len(QUERY_KEYS):
        print(f"{YELLOW}{BOLD}→ All correct but margins degraded. Per-cluster PCA may be needed at larger N.{RESET}")
    else:
        print(f"{RED}{BOLD}→ Routing failures at N=12. Per-cluster PCA required to recover margins.{RESET}")

    # --- Optional full similarity matrix ---
    if show_matrix:
        print(f"\n{BOLD}Full similarity matrix (query × fact):{RESET}")
        header = f"{'':20}" + "".join(f"{k[-8:]:>12}" for k in FACT_KEYS)
        print(DIM + header + RESET)
        for qk in QUERY_KEYS:
            correct_fk = GROUND_TRUTH[qk]
            row = f"{qk:<20}"
            for fk in FACT_KEYS:
                s = sim_matrix[qk][fk]
                cell = f"{s:>12.4f}"
                if fk == correct_fk:
                    row += BOLD + GREEN + cell + RESET
                elif s > sim_matrix[qk][correct_fk]:
                    row += BOLD + RED + cell + RESET
                else:
                    row += cell
            print(row)

    # --- Per-template analysis ---
    print(f"\n{BOLD}Template-cluster separation:{RESET}")
    city_keys = [k for k in FACT_KEYS if k.startswith(("F1_", "F2_", "F3_", "F4_", "F5_", "F6_", "F7_", "F8_"))]
    verb_keys = [k for k in FACT_KEYS if k.startswith(("F11_", "F12_", "F13_", "F14_"))]
    for qk in QUERY_KEYS:
        correct_fk = GROUND_TRUTH[qk]
        sims = sim_matrix[qk]
        template = "city" if correct_fk in city_keys else "verb"
        same_cluster = city_keys if template == "city" else verb_keys
        cross_cluster = verb_keys if template == "city" else city_keys

        same_wrong = [sims[fk] for fk in same_cluster if fk != correct_fk]
        cross = [sims[fk] for fk in cross_cluster]
        best_same_wrong = max(same_wrong) if same_wrong else 0.0
        best_cross = max(cross) if cross else 0.0

        print(
            f"  {qk:<20}  within-cluster best wrong: {best_same_wrong:.4f} ({angle_deg(best_same_wrong):.2f}°)"
            f"  cross-cluster best: {best_cross:.4f} ({angle_deg(best_cross):.2f}°)"
        )

    print()


def parse_args():
    p = argparse.ArgumentParser(description="H-space routing N=12 test")
    p.add_argument("--model", default="mlx-community/gemma-3-4b-it-bf16")
    p.add_argument("--layer", type=int, default=29, help="Layer to extract hidden state from (default: 29)")
    p.add_argument("--show-matrix", action="store_true", help="Print full NxM similarity matrix")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.model, args.layer, args.show_matrix)
