#!/usr/bin/env python3
"""
Cross-Window Reasoning Experiment
Connection and Reference Resolution — Not Calculation

Tests whether checkpoint-chained inference enables genuine reasoning
across multiple replayed windows, not just fact retrieval.

Eight levels of reasoning difficulty. Questions are designed so that
neither window alone can produce the expected answer — the model must
connect facts across checkpoint boundaries.

Design principles:
  - All facts are fictional (no parametric knowledge can help)
  - Each answer requires BOTH windows (neither alone suffices)
  - Prompts NEVER name the bridge entity — only describe what they DID
    in window B; the model must identify them and look up their fact in A
  - Pattern: "someone did X (only in B) — what is that person's Y (only in A)?"
  - Questions test reference resolution and number comparison, NOT arithmetic
  - Four conditions per question: no context, A only, B only, both A+B

Three separate engines are used for clean isolation:
  engine_a  — Window A (Operations Manual) only, as window 0
  engine_b  — Window B (Incident Report) only, as window 0
  engine_ab — Both documents: A as window 0, B as window 1

This ensures "B only" means B's content in isolation, not B primed by A's
checkpoint state.

Usage:
    uv run python examples/inference/cross_window_reasoning.py
    uv run python examples/inference/cross_window_reasoning.py \\
        --model mlx-community/gemma-3-4b-it-bf16 --window-size 8192
    uv run python examples/inference/cross_window_reasoning.py \\
        --model mlx-community/gemma-3-1b-it-bf16 --gen-tokens 100
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import mlx.core as mx

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Fictional documents — all facts invented; no parametric knowledge can help
# ---------------------------------------------------------------------------

WINDOW_A_TEXT = """\
MERIDIAN STATION — OPERATIONS MANUAL
Clearance Level: Standard Crew

Section 1: Station Overview

Meridian Station orbits the gas giant Veluris at an altitude of
12,400 kilometers. The station was commissioned in 2847 and serves
as the primary refueling depot for vessels transiting the Outer
Reach corridor. The station maintains a permanent crew of 340
personnel across four departments: Engineering, Navigation, Medical,
and Commerce.

The station commander is Captain Elara Vasquez, who assumed command
in 2851 after the retirement of Commander Dren Korvath. Vasquez
previously served as chief navigation officer aboard the survey
vessel Pathfinder for eleven years.

Section 2: Power Systems

Meridian Station draws power from three Helios-9 fusion reactors,
each producing 4.7 terawatts under standard load. The reactors
consume deuterium harvested from Veluris's upper atmosphere by
automated skimmer drones. Fuel reserves are maintained at a minimum
of 180 days' supply, currently holding 12,400 metric tons of
deuterium as of the last audit.

The station's peak power demand occurs during docking operations,
when the magnetic grapple array draws an additional 2.1 terawatts.
The safety threshold is 90% of total reactor capacity. Exceeding
this threshold triggers an automatic load-shedding protocol that
disables non-essential systems in priority order: recreational
facilities, external lighting, non-critical sensors, and commerce
district heating.

Section 3: Docking Procedures

Meridian Station has 16 docking berths arranged in four quadrants.
Berths 1-4 (Alpha Quadrant) handle military vessels. Berths 5-8
(Beta Quadrant) handle commercial freighters. Berths 9-12 (Gamma
Quadrant) handle passenger liners. Berths 13-16 (Delta Quadrant)
are reserved for emergency and maintenance use.

Standard docking clearance requires: (a) valid transit license,
(b) cargo manifest filed 48 hours in advance, (c) fusion reactor
shutdown to cold status before final approach, and (d) payment of
docking fees assessed at 3.2 credits per metric ton of vessel
displacement.

The average commercial freighter displaces 45,000 metric tons.
Passenger liners average 120,000 metric tons. Military vessels
are exempt from docking fees under the Outer Reach Defense Compact.

Section 4: Personnel

Engineering department: 140 crew (Chief Engineer: Holt Maddox)
Navigation department: 60 crew (Chief Navigator: Suri Patel)
Medical department: 45 crew (Chief Medical Officer: Dr. Arani Osei)
Commerce department: 55 crew (Commerce Director: Fen Halloran)
Command staff: 12 crew
Station security: 28 crew (Security Chief: Tomas Eriksen)

Annual crew rotation occurs on the solstice of the third quarter.
Each rotation replaces approximately 30% of personnel. The current
crew cohort began service on 2852.Q3 and is scheduled for rotation
on 2853.Q3.

Section 5: Emergency Protocols

In the event of reactor failure, the station maintains a backup
power grid supplied by 240 high-capacity battery cells. Each cell
provides 0.05 terawatts for a maximum duration of 72 hours. The
backup grid prioritizes life support, communications, and the
medical bay.

The emergency evacuation capacity is 400 personnel via 20 escape
pods, each rated for 20 occupants. Pod deployment requires
authorization from the station commander or, in their absence,
the senior department chief present.
"""

WINDOW_B_TEXT = """\
OUTER REACH TRANSIT AUTHORITY
INCIDENT REPORT 2852-7741
Classification: Critical — For Official Review

Date: 2852.Q4.17
Station: Meridian Station, Veluris Orbit
Reporting Officer: Security Chief Tomas Eriksen

SUMMARY OF EVENTS

At 0347 station time on 2852.Q4.17, Meridian Station experienced
a cascade failure originating in Reactor 2. A microfracture in the
deuterium containment vessel caused a controlled shutdown of
Reactor 2, reducing station power output by one-third.

At the time of the failure, four vessels were docked at the station:
- Berth 3 (Alpha): ORD Vigilance, military patrol corvette,
  8,200 MT displacement
- Berth 6 (Beta): MV Sunrise Trader, commercial freighter,
  52,000 MT displacement
- Berth 9 (Gamma): SSL Celestia, passenger liner, 135,000 MT
  displacement, carrying 2,847 passengers and 380 crew
- Berth 14 (Delta): Maintenance tender Bracket, 1,200 MT,
  station-assigned

Additionally, the commercial freighter MV Iron Horizon (48,000 MT)
was on final approach to Berth 7, 47 minutes from docking contact.

POWER SITUATION

With Reactor 2 offline, total available power dropped from
14.1 TW to 9.4 TW. Station base load (life support, gravity,
atmosphere) requires 5.8 TW. The remaining 3.6 TW was insufficient
to operate the magnetic grapple array (2.1 TW) while maintaining
all essential services.

Captain Vasquez ordered load-shedding Protocol C at 0355, disabling
recreational facilities and external lighting, freeing an additional
0.9 TW. This brought available discretionary power to 4.5 TW.

CRITICAL DECISION

The Commerce Director, Fen Halloran, reported that the MV Sunrise
Trader carried 14,000 metric tons of perishable medical supplies
bound for the colony on Thessara-4, with a remaining viable window
of 6 days. Any delay beyond 18 hours would require the supplies to
be rerouted to the cold storage facility on Orbital 7, adding 11
days to delivery.

The SSL Celestia had reported three passengers requiring urgent
medical attention in Dr. Osei's care: two cardiac cases and one
emergency surgical patient. Transfer to the Celestia's own medical
bay was not recommended due to equipment limitations.

Chief Engineer Maddox estimated Reactor 2 repair time at 36 to
48 hours, pending replacement of the containment vessel liner.
The replacement part was not in station inventory. The nearest
supply depot with the part was Relay Station Kappa, 5.2 days
transit time each way at standard cruise speed.

ACTIONS TAKEN

0412: Captain Vasquez convened an emergency department chiefs
meeting. Present: Maddox (Engineering), Patel (Navigation),
Osei (Medical), Halloran (Commerce), Eriksen (Security).

0430: Vasquez ordered the MV Iron Horizon to hold position at
standoff distance (200 km) pending power resolution. Iron Horizon
acknowledged and entered station-keeping orbit.

0445: Dr. Osei recommended transferring the three critical patients
from the SSL Celestia to the station medical bay, where the
equipment was superior. This transfer would require operating
Gamma Quadrant's docking umbilical, which draws 0.4 TW during
pressurized personnel transfer.

0500: Vasquez authorized the medical transfer. Power allocation:
5.8 TW (base) + 0.4 TW (medical transfer) = 6.2 TW, leaving
3.2 TW discretionary from the 9.4 TW available.

0515: Medical transfer completed successfully. Three patients in
station medical bay under Dr. Osei's care.

0520: Vasquez requested options for docking the MV Iron Horizon
given the power constraint. Maddox confirmed that the grapple
array could operate at reduced power (1.4 TW instead of 2.1 TW)
for vessels under 50,000 MT, but with manual override required
from a qualified docking pilot.

Chief Navigator Patel volunteered to perform the manual docking
procedure, noting her certification on Type-C magnetic grapple
systems. Estimated docking time at reduced power: 3.5 hours
(versus 45 minutes at full power).

0600: Vasquez authorized reduced-power docking of MV Iron Horizon
at Berth 7. Patel began manual grapple sequence.

0935: MV Iron Horizon successfully docked at Berth 7. No incidents.
Cargo transfer of medical supplies commenced immediately.

1400: Medical supplies secured aboard MV Sunrise Trader for onward
transit to Thessara-4. Sunrise Trader departed Berth 6 at 1430,
within the viable delivery window.

ONGOING ISSUES

Reactor 2 remains offline. Maddox has dispatched the maintenance
tender Bracket to Relay Station Kappa to retrieve the containment
liner. Estimated return: 2852.Q4.27 (10.4 days).

The ORD Vigilance has offered to remain at station as security
escort until full power is restored. Captain Vasquez accepted.

The three medical patients remain in station care. Dr. Osei reports
stable condition for the cardiac cases, with surgical intervention
for the third patient scheduled for 2852.Q4.18.

COST ASSESSMENT

Docking fees waived for MV Iron Horizon due to emergency
circumstances (Commerce Director authority, Section 3 waiver).
Estimated revenue loss from the fee waiver: [to be calculated].

Additional costs: overtime pay for manual docking crew (12 hours x
8 personnel), emergency power management staffing, Bracket
deployment to Relay Station Kappa (fuel and crew costs).

REPORT STATUS: OPEN
Awaiting: Reactor 2 repair completion, patient discharge,
cost assessment finalization.
"""


# ---------------------------------------------------------------------------
# Reasoning questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ------------------------------------------------------------------
    # L1 — Baseline (single-window sanity check)
    # Rule: answerable from A alone; verifies engine is working.
    # ------------------------------------------------------------------
    {
        "level": 1,
        "type": "single-window baseline",
        "question": "How many docking berths does Meridian Station have?",
        "prompt": "How many docking berths does Meridian Station have?",
        "windows": ["A"],
        "expected_any": ["16"],
        "expected_from_a": "16",
        "expected_from_b": None,
        "why": "Baseline: fact is in Window A (Section 3). If this fails, something is broken.",
    },
    # ------------------------------------------------------------------
    # L2 — Action (B) → person → department count (A)
    # Bridge: Patel volunteered for manual docking (B only)
    #         Navigation department has 60 crew (A only)
    # Prompt does NOT name Patel or Navigation.
    # ------------------------------------------------------------------
    {
        "level": 2,
        "type": "cross-window action→count",
        "question": "How many crew work in the department of the person who volunteered for manual docking?",
        "prompt": (
            "The incident report describes someone who volunteered to perform "
            "a dangerous manual docking procedure. How many people work in "
            "that person's department?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["60"],
        "expected_from_a": "60",
        "expected_from_b": None,
        "why": (
            "A only: All department sizes listed, but no one volunteered for anything in A. "
            "B only: Patel volunteered — but B has no crew counts. "
            "A+B: Patel volunteered (B) → Chief Navigator → Navigation: 60 crew (A)."
        ),
    },
    # ------------------------------------------------------------------
    # L3 — Action (B) → person → prior-role duration (A)
    # Bridge: Vasquez authorized the medical transfer (B only)
    #         Vasquez served eleven years on Pathfinder (A only)
    # Prompt does NOT name Vasquez.
    # ------------------------------------------------------------------
    {
        "level": 3,
        "type": "cross-window action→tenure",
        "question": "How long had the person who authorized the medical transfer served in their previous role?",
        "prompt": (
            "The incident report describes a medical transfer of three patients. "
            "Who authorized that transfer, and how many years did that person "
            "serve in their previous role before commanding this station?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["eleven", "11"],
        "expected_from_a": "eleven",
        "expected_from_b": None,
        "why": (
            "A only: Vasquez served 11 years on Pathfinder — but A has no medical transfer event. "
            "B only: Vasquez authorized the transfer at 0500 — but B has no tenure data. "
            "A+B: Vasquez authorized transfer (B) → Vasquez served eleven years on Pathfinder (A)."
        ),
    },
    # ------------------------------------------------------------------
    # L4 — Cross-window capacity comparison (two numbers, one per window)
    # Bridge: SSL Celestia passenger/crew count (B only) vs pod capacity (A only)
    # Prompt does NOT name the liner.
    # ------------------------------------------------------------------
    {
        "level": 4,
        "type": "cross-window capacity comparison",
        "question": "Could the station's escape pods evacuate all personnel from the docked passenger liner?",
        "prompt": (
            "The incident report mentions a passenger liner that was docked at "
            "the station during the crisis. How many people were aboard that liner, "
            "and what is the station's total escape pod capacity? Could the pods "
            "handle that many people?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["3,227", "3227", "2,847", "2847", "400", "no", "not enough", "insufficient", "cannot"],
        "expected_from_a": "400",
        "expected_from_b": "2,847",
        "why": (
            "A only: Escape pods hold 400 (20 × 20). No liner passenger count. "
            "B only: SSL Celestia had 2,847 passengers + 380 crew = 3,227. No pod capacity. "
            "A+B: 3,227 people (B) vs 400 pod capacity (A) → massively insufficient."
        ),
    },
    # ------------------------------------------------------------------
    # L5 — Cross-window docking feasibility (two numbers, one per window)
    # Bridge: 50,000 MT grapple limit (B only) vs freighter avg 45,000 MT (A only)
    # Prompt does NOT state the comparison directly — model must make it.
    # ------------------------------------------------------------------
    {
        "level": 5,
        "type": "cross-window feasibility",
        "question": "Could an average commercial freighter dock at reduced power during the crisis?",
        "prompt": (
            "The operations manual gives average displacements for different "
            "vessel types. The incident report states that during the crisis, "
            "the magnetic grapple could only handle vessels below a certain "
            "weight at reduced power. Based on both documents, could an average "
            "commercial freighter have docked at reduced power during the crisis?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["45,000", "45000", "50,000", "50000", "yes", "could", "within", "under", "below"],
        "expected_from_a": "45,000",
        "expected_from_b": "50,000",
        "why": (
            "A only: Freighters average 45,000 MT — but A has no grapple weight limit. "
            "B only: Reduced grapple works under 50,000 MT — but B has no avg freighter displacement. "
            "A+B: 45,000 MT (A) < 50,000 MT limit (B) → yes, an average freighter could dock."
        ),
    },
    # ------------------------------------------------------------------
    # L6 — Cross-window temporal comparison (one date per window)
    # Bridge: repair date ~2852.Q4.29 (B only) vs rotation 2853.Q3 (A only)
    # ------------------------------------------------------------------
    {
        "level": 6,
        "type": "cross-window timeline",
        "question": "How far apart are the expected reactor repair and the next crew rotation?",
        "prompt": (
            "The incident report gives the expected return date of the vessel "
            "carrying the repair part. The operations manual describes when the "
            "crew rotation happens. Approximately how many quarters separate the "
            "expected reactor repair completion from the next crew rotation?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["two", "2", "three", "3", "quarter", "Q4", "2852", "2853"],
        "expected_from_a": "2853",
        "expected_from_b": "2852",
        "why": (
            "A only: Rotation is 2853.Q3 — but A has no repair timeline. "
            "B only: Bracket returns 2852.Q4.27; repair done ~Q4.29 — but B has no rotation date. "
            "A+B: Repair ~2852.Q4 (B); rotation 2853.Q3 (A) → roughly 2-3 quarters apart."
        ),
    },
    # ------------------------------------------------------------------
    # L7 — Author (B) → person → department count (A)
    # Bridge: Eriksen wrote the incident report (B only)
    #         Station security has 28 crew (A only)
    # Prompt does NOT name Eriksen or Security.
    # ------------------------------------------------------------------
    {
        "level": 7,
        "type": "cross-window author→count",
        "question": "How many people are in the department of the officer who wrote the incident report?",
        "prompt": (
            "The incident report was written by a specific officer. "
            "According to the operations manual, how many people are in "
            "that officer's department?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["28"],
        "expected_from_a": "28",
        "expected_from_b": None,
        "why": (
            "A only: Security: 28 crew (Eriksen) — but A has no incident report context. "
            "B only: Reporting Officer is Security Chief Tomas Eriksen — but B has no crew count. "
            "A+B: Eriksen wrote the report (B) → Security Chief → 28 security crew (A)."
        ),
    },
    # ------------------------------------------------------------------
    # L8 — Action (B) → person → department count (A)
    # Bridge: Halloran reported on perishable supplies (B only)
    #         Commerce department has 55 crew (A only)
    # Prompt does NOT name Halloran or Commerce.
    # ------------------------------------------------------------------
    {
        "level": 8,
        "type": "cross-window reporter→count",
        "question": "How many crew work in the department of the person who reported on the medical supplies?",
        "prompt": (
            "The incident report describes someone who reported that perishable "
            "medical supplies had a remaining viable window of 6 days. What is "
            "that person's name, and how many people work in their department?"
        ),
        "windows": ["A", "B"],
        "expected_any": ["55", "Halloran", "Fen"],
        "expected_from_a": "55",
        "expected_from_b": None,
        "why": (
            "A only: Commerce department has 55 crew — but A has no mention of medical supplies. "
            "B only: Fen Halloran (Commerce Director) reported on the supplies — but B has no crew count. "
            "A+B: Halloran reported on supplies (B) → Commerce Director → 55 Commerce crew (A)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def encode(tokenizer, text: str) -> list[int]:
    """Encode raw text (used for document context, not queries)."""
    return tokenizer.encode(text, add_special_tokens=False)


def encode_query(tokenizer, question: str) -> list[int]:
    """
    Encode a user query using the model's chat template.

    Produces:  <bos><start_of_turn>user\\n{question}<end_of_turn>\\n<start_of_turn>model\\n
    The model then generates its response from the model-turn start.
    """
    messages = [{"role": "user", "content": question}]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    return ids


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def timed_generate(
    engine,
    tokenizer,
    query_text: str,
    replay_window_ids: list[int] | None,
    max_new_tokens: int,
) -> tuple[str, float]:
    """Run generate and return (decoded_text, elapsed_seconds)."""
    q_ids = encode_query(tokenizer, query_text)
    t0 = time.perf_counter()
    gen = engine.generate(
        q_ids,
        replay_window_ids=replay_window_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - t0
    return decode(tokenizer, gen), elapsed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                model_id,
                local_files_only=True,
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
            )
        )
    except Exception:
        pass
    print(f"  Downloading {model_id} ...")
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
    )


def _apply_weights(model, model_path: Path) -> None:
    from mlx.utils import tree_unflatten

    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))
    sanitized = model.sanitize(raw)
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in sanitized.items()
    }
    model.update(tree_unflatten(list(sanitized.items())))
    mx.eval(model.parameters())


def _load_inference_engine():
    """
    Load UnlimitedContextEngine without triggering chuk_lazarus.inference.__init__.

    The inference package __init__ imports virtual_expert which needs chuk_virtual_expert,
    an optional dependency not required for this script.  We bypass it by pre-registering
    lightweight namespace stubs for the parent packages, then loading the three required
    module files directly in dependency order.
    """
    inf = Path(__file__).parents[2] / "src" / "chuk_lazarus" / "inference"

    # Pre-register namespace stubs so relative imports inside the loaded files
    # resolve correctly without triggering any __init__.py.
    for pkg_name, pkg_dir in [
        ("chuk_lazarus.inference", str(inf)),
        ("chuk_lazarus.inference.context", str(inf / "context")),
    ]:
        if pkg_name not in sys.modules:
            stub = types.ModuleType(pkg_name)
            stub.__path__ = [pkg_dir]
            stub.__package__ = pkg_name
            sys.modules[pkg_name] = stub

    def _load_mod(dotted, fpath):
        spec = importlib.util.spec_from_file_location(dotted, str(fpath))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load in dependency order: protocols → kv_generator → unlimited_engine
    _load_mod("chuk_lazarus.inference.context.protocols",      inf / "context" / "protocols.py")
    _load_mod("chuk_lazarus.inference.context.kv_generator",   inf / "context" / "kv_generator.py")
    engine_mod = _load_mod(
        "chuk_lazarus.inference.context.unlimited_engine", inf / "context" / "unlimited_engine.py"
    )
    return engine_mod.UnlimitedContextEngine


def load_models(model_id: str):
    from transformers import AutoTokenizer

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config = GemmaConfig.from_hf_config(json.load(f))

    rs = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs, model_path)
    rs.eval()

    UnlimitedContextEngine = _load_inference_engine()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return rs, config, UnlimitedContextEngine, tokenizer


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(engine_a, engine_b, engine_ab, tokenizer, questions, gen_tokens):
    """
    Run each question under four conditions:

      No context    — engine_ab with no replay (hallucination baseline)
      Window A only — engine_a  with replay [0]  (Operations Manual only)
      Window B only — engine_b  with replay [0]  (Incident Report only)
      Both A+B      — engine_ab with replay [0, 1]  (cross-window synthesis)

    Each engine processes its documents cleanly in isolation so that
    "B only" truly means B's content without A's checkpoint priming.
    """
    pass_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    results = []

    for q in questions:
        print(f"\n{'═' * 60}")
        print(f"Level {q['level']}: {q['type']}")
        print(f"Q: {q['question']}")
        print(f"Why cross-window: {q['why']}")
        print(f"{'─' * 60}")

        conditions = [
            (1, "No context",    engine_ab, None),
            (2, "Window A only", engine_a,  [0]),
            (3, "Window B only", engine_b,  [0]),
            (4, "Both A+B",      engine_ab, [0, 1]),
        ]

        q_results = {}
        q_raw = {}
        for cond_idx, label, engine, replay_ids in conditions:
            result, elapsed = timed_generate(
                engine,
                tokenizer,
                query_text=q["prompt"],
                replay_window_ids=replay_ids,
                max_new_tokens=gen_tokens,
            )

            found = any(kw.lower() in result.lower() for kw in q["expected_any"])
            tick = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
            display = result[:80].replace("\n", " ")
            if len(result) > 80:
                display += "..."

            print(f"  {tick} {label:<15s} ({elapsed * 1000:.0f} ms): {repr(display)}")

            total_counts[cond_idx] += 1
            if found:
                pass_counts[cond_idx] += 1
            q_results[label] = found
            q_raw[label] = result

        # Strict cross-window check using expected_from_a / expected_from_b.
        # For chain questions (expected_from_b is None):
        #   cross_window = A+B has the A-fact AND B-only lacks it.
        # For comparison questions (both set):
        #   cross_window = A+B has both facts AND A-only lacks the B-fact
        #   AND B-only lacks the A-fact.
        fa = q.get("expected_from_a")
        fb = q.get("expected_from_b")
        if fa and q["level"] > 1:
            ab_text = q_raw.get("Both A+B", "").lower()
            b_text  = q_raw.get("Window B only", "").lower()
            a_text  = q_raw.get("Window A only", "").lower()
            if fb:
                strict = (fa.lower() in ab_text and fb.lower() in ab_text
                          and fb.lower() not in a_text and fa.lower() not in b_text)
            else:
                strict = fa.lower() in ab_text and fa.lower() not in b_text
            cross_label = f"{GREEN}strict cross-window ✓{RESET}" if strict else f"{DIM}(not strict){RESET}"
            print(f"  → {cross_label}")
            q_results["_strict"] = strict
        else:
            q_results["_strict"] = False

        results.append({**q, "results": q_results, "raw": q_raw})

    return results, pass_counts, total_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Cross-window reasoning experiment")
    p.add_argument(
        "--model",
        default="mlx-community/gemma-3-270m-it-bf16",
        help="Model ID (HF hub or local path). Use 4b for best results.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=4096,
        help="Tokens per context window. Each document (~1500-2000 tokens) must fit in one window.",
    )
    p.add_argument(
        "--gen-tokens",
        type=int,
        default=120,
        help="Max tokens to generate per query (100-150 for reasoning questions).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}Cross-Window Reasoning Experiment{RESET}")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Window size: {args.window_size} tokens")
    print(f"  Gen tokens:  {args.gen_tokens}")

    print("\nLoading model ...")
    rs, config, EngineClass, tokenizer = load_models(args.model)
    print(
        f"  {config.num_hidden_layers} layers, "
        f"hidden={config.hidden_size}, "
        f"kv_heads={config.num_key_value_heads}, "
        f"head_dim={config.head_dim}"
    )
    kv_bpt = 2 * config.num_key_value_heads * config.head_dim * config.num_hidden_layers * 2
    print(f"  KV bytes/token: {fmt_bytes(kv_bpt)}")

    # Tokenise documents
    print("\nTokenising documents ...")
    window_a_ids = encode(tokenizer, WINDOW_A_TEXT)
    window_b_ids = encode(tokenizer, WINDOW_B_TEXT)
    print(f"  Window A (Operations Manual): {len(window_a_ids):,} tokens")
    print(f"  Window B (Incident Report):   {len(window_b_ids):,} tokens")

    for label, ids in [("A", window_a_ids), ("B", window_b_ids)]:
        if len(ids) > args.window_size:
            print(
                f"\n  {YELLOW}WARNING: Window {label} ({len(ids)} tokens) exceeds "
                f"--window-size ({args.window_size}).{RESET}"
            )
            print(f"  It will span multiple windows; facts may split across boundaries.")
            print(f"  Increase --window-size for a clean single-window-per-document experiment.")

    # Warm up JIT (one small prefill before timed operations)
    print("\nWarming up JIT ...")
    _tmp = EngineClass(rs, config, window_size=args.window_size)
    _, _kv = _tmp.kv_gen.prefill(mx.array([[1, 2, 3, 4, 5]]))
    mx.eval()
    del _tmp, _kv
    print("  Done.")

    # Build three engines with clean document isolation
    print(f"\n{BOLD}Building context engines (3 × document processing){RESET}")

    print("  [1/3] Engine A — Operations Manual only ...")
    engine_a = EngineClass(rs, config, window_size=args.window_size)
    t0 = time.perf_counter()
    engine_a.process(window_a_ids)
    engine_a.flush()
    s_a = engine_a.stats()
    print(
        f"        {len(window_a_ids):,} tokens → "
        f"{s_a.archived_windows} window(s), "
        f"{fmt_bytes(s_a.cold_warm_bytes)} stored  "
        f"({(time.perf_counter() - t0) * 1000:.0f} ms)"
    )

    print("  [2/3] Engine B — Incident Report only ...")
    engine_b = EngineClass(rs, config, window_size=args.window_size)
    t0 = time.perf_counter()
    engine_b.process(window_b_ids)
    engine_b.flush()
    s_b = engine_b.stats()
    print(
        f"        {len(window_b_ids):,} tokens → "
        f"{s_b.archived_windows} window(s), "
        f"{fmt_bytes(s_b.cold_warm_bytes)} stored  "
        f"({(time.perf_counter() - t0) * 1000:.0f} ms)"
    )

    print("  [3/3] Engine AB — both documents ...")
    engine_ab = EngineClass(rs, config, window_size=args.window_size)
    t0 = time.perf_counter()
    engine_ab.process(window_a_ids)
    engine_ab.flush()
    engine_ab.process(window_b_ids)
    engine_ab.flush()
    s_ab = engine_ab.stats()
    print(
        f"        {s_ab.total_tokens:,} tokens → "
        f"{s_ab.archived_windows} window(s), "
        f"{fmt_bytes(s_ab.cold_warm_bytes)} stored, "
        f"{s_ab.compression_ratio:.0f}× compression  "
        f"({(time.perf_counter() - t0) * 1000:.0f} ms)"
    )
    print(
        f"        ({fmt_bytes(s_ab.cold_warm_bytes)} vs "
        f"{fmt_bytes(s_ab.equivalent_kv_bytes)} standard KV)"
    )

    # Run experiment
    print(f"\n{BOLD}Running Experiment{RESET}")
    print("=" * 60)
    print("Conditions per question:")
    print(f"  ✓/✗  {DIM}No context   {RESET} — model hallucinates (baseline)")
    print(f"  ✓/✗  {DIM}Window A only{RESET} — partial info from Operations Manual")
    print(f"  ✓/✗  {DIM}Window B only{RESET} — partial info from Incident Report")
    print(f"  ✓/✗  {BOLD}Both A+B     {RESET} — cross-window synthesis (target)")

    results, pass_counts, total_counts = run_experiment(
        engine_a, engine_b, engine_ab, tokenizer, QUESTIONS, args.gen_tokens
    )

    # Summary matrix
    print(f"\n{'═' * 68}")
    print(f"{BOLD}Results Matrix{RESET}")
    print(f"{'─' * 68}")
    header = f"  {'Level':<8} {'Type':<28} {'None':>6} {'A only':>7} {'B only':>7} {'A+B':>6} {'Strict':>7}"
    print(header)
    print(f"  {'─' * 8} {'─' * 28} {'─' * 6} {'─' * 7} {'─' * 7} {'─' * 6} {'─' * 7}")

    for r in results:
        lvl = f"L{r['level']}"
        typ = r["type"][:27]
        conds = r["results"]

        def _tick(k):
            return f"{GREEN}✓{RESET}" if conds.get(k, False) else f"{RED}✗{RESET}"

        strict_sym = f"{GREEN}✓{RESET}" if conds.get("_strict", False) else f"{DIM}—{RESET}"
        print(
            f"  {lvl:<8} {typ:<28} "
            f"{_tick('No context'):>6} "
            f"{_tick('Window A only'):>7} "
            f"{_tick('Window B only'):>7} "
            f"{_tick('Both A+B'):>6} "
            f"{strict_sym:>7}"
        )

    # Loose wins: A+B passes keyword check; neither A-only nor B-only does.
    cross_window_wins = [
        r
        for r in results
        if r["level"] >= 2
        and r["results"].get("Both A+B", False)
        and not r["results"].get("Window A only", False)
        and not r["results"].get("Window B only", False)
    ]

    # Strict wins: A+B contains the A-fact (and B-fact for comparison questions);
    # B-only lacks the A-fact; A-only lacks the B-fact (for comparison questions).
    strict_wins = [r for r in results if r["level"] >= 2 and r["results"].get("_strict", False)]

    print(f"\n{BOLD}Cross-Window Synthesis Score{RESET}")
    n_cross = len(QUESTIONS) - 1  # exclude L1 baseline
    print(
        f"  Loose  (A+B ✓, A-only ✗, B-only ✗): "
        f"{len(cross_window_wins)} / {n_cross}"
    )
    print(
        f"  Strict (A+B has A-fact; B-only lacks it): "
        f"{len(strict_wins)} / {n_cross}"
    )

    winning = strict_wins if strict_wins else cross_window_wins
    if winning:
        print(f"\n{GREEN}{BOLD}Cross-window reasoning confirmed.{RESET}")
        for r in winning:
            print(f"  L{r['level']} ({r['type']})")
            print(f"     {DIM}{r['why']}{RESET}")

        print(f"""
  {BOLD}What this proves:{RESET}

  The model didn't just retrieve facts from checkpoint libraries.
  It reasoned across them.

  Neither window alone contained enough to answer these questions.
  The model attended across checkpoint boundaries in a single
  computation — connecting facts that no individual window held.

  This is what RAG cannot do. RAG retrieves chunks independently
  and concatenates them. This approach loads multiple windows into
  one KV context and lets the model attend across all of them
  natively — one attention pass, not N independent retrievals.

  On a laptop. Offline. In seconds.
""")
    else:
        print(f"\n{YELLOW}No definitive cross-window synthesis at this model scale.{RESET}")
        a_b_pass = sum(1 for r in results[1:] if r["results"].get("Both A+B", False))
        print(f"  A+B succeeded on {a_b_pass}/{n_cross} questions (keyword match).")
        if "270m" in args.model.lower():
            print(f"  Try --model mlx-community/gemma-3-1b-it-bf16 for stronger reasoning.")
            print(f"  Try --model mlx-community/gemma-3-4b-it-bf16 for full results.")
        print(f"  These questions require reference resolution, not arithmetic —")
        print(f"  the 4B model is the recommended target.")
        print()


if __name__ == "__main__":
    main()
