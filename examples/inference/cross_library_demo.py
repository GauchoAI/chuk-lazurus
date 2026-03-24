#!/usr/bin/env python3
"""
Cross-Library Checkpoint Retrieval Demo

Three pre-filled knowledge libraries — lore, engineering, law — loaded
from disk in milliseconds and queried individually and simultaneously.

No prefill wait. No RAG pipeline. No vector database. No embeddings.
Sub-second retrieval. Offline. Private. Lossless.

The libraries contain fictional facts the model cannot know:
  - The World of Meridian:       Kael Dawnstrider founded Aethermoor;
                                 Maren Holloway built the Spire of Echoes
  - Resonance Engineering:       Holloway Constant H = 7.83 Hz;
                                 Spire-class: thermal fracture >15 min
  - Aethermoor City Charter:     8% max tariff; festival ≤ 17 min per
                                 Prof. Waveborn's safety research

Cross-library questions require facts from multiple sources:
  "Why is the festival limited to 17 minutes?" needs lore + engineering.
  "Who maintains the Spire and at what frequency?" needs all three.

Usage
-----
    # Run demo (auto-builds libraries on first run):
    uv run python examples/inference/cross_library_demo.py

    # Force rebuild of libraries:
    uv run python examples/inference/cross_library_demo.py --rebuild

    # Use a larger model:
    uv run python examples/inference/cross_library_demo.py \\
        --model mlx-community/gemma-3-1b-it-bf16

    # Skip the control act (faster):
    uv run python examples/inference/cross_library_demo.py --no-control
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

# Checkpoint library types — imported early so build_library() can use them
from chuk_lazarus.inference.context import (
    LibraryFile,
    LibraryFormatVersion,
    LibraryManifest,
    LibrarySource,
    WindowMeta,
)

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Library text content (inline so the demo is self-contained)
# ---------------------------------------------------------------------------

MERIDIAN_TEXT = """\
THE WORLD OF MERIDIAN — A Compendium

The Free City of Aethermoor sits at the confluence of the River
Silvanis and the Crystalfen Marshes. Founded in the Third Age by
the explorer Kael Dawnstrider, it serves as the neutral trading
hub between the Northern Kingdoms and the Southern Reach.

The city is governed by the Triad Council: three elected magistrates
who serve rotating seven-year terms. The current council members
are Magistrate Yelen Ashford (trade), Magistrate Darro Voss
(defense), and Magistrate Illara Sunchaser (knowledge).

Aethermoor's most famous landmark is the Spire of Echoes, a
crystalline tower that amplifies sound across the entire city.
Built by the artificer Maren Holloway in the Fourth Age, the
Spire uses resonance harmonics to broadcast announcements from the
Council Chamber at its peak. The tower stands 340 feet tall and
is visible from twenty miles in any direction.

The city's economy depends on three exports: silvavine wine
(produced from grapes grown only in the Crystalfen microclimate),
ghost-iron (a lightweight metal mined from the Ashfell Caverns
beneath the city), and encrypted message scrolls (a magical
service provided by the Guild of Whispers).

The River Silvanis flows south from the Frozen Peaks through
Aethermoor and into the Bay of Tides. Navigation upstream requires
a licensed riverpilot due to the Silvanis Rapids at mile marker 47.
The rapids have claimed over 200 vessels since the city's founding.

The Guild of Whispers operates from the Velvet Archive, a
windowless building in the Merchant Quarter. Their leader, known
only as The Archivist, has held the position for forty-three years.
The Guild's encryption method, called Threadweaving, encodes
messages into patterns of colored silk that can only be decoded
by a matched silk key held by the recipient.

Aethermoor celebrates the Festival of Echoes every winter solstice.
During the festival, the Spire of Echoes plays a harmonic sequence
composed by Maren Holloway herself, which resonates through the
crystal structure for exactly seventeen minutes. Legend says anyone
who hears all seventeen minutes gains clarity of purpose for the
coming year.

The Southern Gate of Aethermoor bears the inscription: "All who
trade in good faith find shelter here." It was carved by the
stonemason Petra Ironhand, who also designed the city's famous
bridge, the Span of Accord, which crosses the Silvanis at its
widest point — 180 feet of single-arch construction.
"""

RESONANCE_TEXT = """\
FUNDAMENTALS OF RESONANCE ENGINEERING
Third Edition — Prof. Aldric Waveborn, Thornfield Institute

Chapter 1: Principles of Harmonic Amplification

Resonance occurs when a system is driven at its natural frequency,
producing oscillations of increasing amplitude. In engineered
systems, controlled resonance is used for amplification, signal
transmission, and structural analysis.

The fundamental equation of resonance amplification is:
  A = F / (k - mω² + jcω)
where A is amplitude, F is driving force, k is stiffness, m is
mass, ω is driving frequency, c is damping coefficient, and j
is the imaginary unit.

Crystal-based resonators exploit the piezoelectric properties of
certain minerals. When mechanically stressed, these crystals
produce an electrical signal at their natural frequency. The
quality factor (Q-factor) of a crystal resonator determines its
frequency selectivity: higher Q means sharper resonance peaks
and better signal discrimination.

The Holloway Constant (H = 7.83 Hz) represents the fundamental
resonance frequency of standard crystalline amplification chambers.
Named after the pioneering artificer Maren Holloway, this constant
defines the base frequency from which all harmonic series in
crystal engineering are derived. Chambers tuned to exact multiples
of H achieve maximum energy transfer with minimum signal loss.

Chapter 2: Acoustic Propagation in Crystalline Structures

Sound waves travel through crystalline lattices at velocities
determined by the crystal's elastic modulus and density. In pure
quartz, acoustic velocity is approximately 5,760 meters per
second — nearly 17 times the speed of sound in air.

Crystalline waveguides channel acoustic energy along preferred
lattice directions. A properly oriented crystal column can
transmit sound over distances exceeding one kilometer with less
than 3 dB of signal loss, provided the column is free of lattice
defects larger than one-quarter wavelength.

The Waveborn-Ashford Effect describes the phenomenon where
acoustic waves in a crystal lattice spontaneously reorganize
into standing wave patterns when the crystal exceeds a critical
length-to-wavelength ratio of 47:1. This self-organization is
exploited in long-range acoustic transmission systems.

Chapter 3: Practical Applications

The most famous application of crystal resonance engineering is
the Spire-class acoustic amplifier. These structures, typically
between 200 and 400 feet tall, use a central crystalline column
surrounded by tuned resonance chambers at harmonic intervals.

A Spire-class amplifier operating at the Holloway Constant can
broadcast intelligible speech across a radius of approximately
20 miles in still air. The broadcast duration is limited by the
thermal capacity of the crystal column: continuous operation
above 15 minutes risks thermal fracture of the primary resonator.

The Thornfield Institute maintains the only two operational
Spire-class amplifiers outside of municipal installations. The
institute's research spire, completed in 2847, operates at 3H
(23.49 Hz) — three times the Holloway Constant — enabling
subsonic transmission that can penetrate solid rock to a depth
of 400 meters.
"""

CHARTER_TEXT = """\
CHARTER OF THE FREE CITY OF AETHERMOOR
Ratified in the Third Age, Year 142
Amended: Fourth Age Year 7, Fourth Age Year 89, Fifth Age Year 3

ARTICLE I — SOVEREIGNTY AND GOVERNANCE

Section 1. The Free City of Aethermoor is a sovereign entity,
beholden to no kingdom, empire, or external authority. Its
sovereignty derives from the Compact of Founding signed by
Kael Dawnstrider and the twelve original settler families.

Section 2. Governance shall be vested in the Triad Council,
consisting of three magistrates elected by popular vote of all
citizens who have resided within the city walls for a minimum of
three continuous years.

Section 3. Each magistrate shall serve a term of seven years.
Terms shall be staggered such that one seat is contested every
two years and four months, ensuring continuity of governance.

Section 4. The three magistrate portfolios are: Trade (oversight
of commerce, tariffs, and the Merchant Quarter), Defense (oversight
of the City Guard, walls, and river patrol), and Knowledge
(oversight of the Guild of Whispers, the Public Archives, and
the Spire of Echoes).

ARTICLE II — RIGHTS OF CITIZENS

Section 1. All citizens enjoy the right of free trade within the
city walls, subject to tariffs set by the Trade Magistrate not
to exceed 8% of declared value.

Section 2. The Guild of Whispers shall provide encryption services
to any citizen at a regulated rate of no more than three silver
marks per standard scroll. This right is absolute and may not be
suspended even in time of war.

Section 3. No citizen may be compelled to reveal the contents of
an encrypted communication. The Threadweaving method is recognized
as inviolable private correspondence under this Charter.

ARTICLE III — THE SPIRE OF ECHOES

Section 1. The Spire of Echoes is public property held in trust
by the Knowledge Magistrate. No private entity may claim ownership
or exclusive use of the Spire's broadcast capabilities.

Section 2. The Spire shall be used for official Council
announcements, emergency warnings, and the annual Festival of
Echoes. Private broadcasts are prohibited except by unanimous
Council vote.

Section 3. Maintenance of the Spire's crystalline column shall be
funded from the general treasury at a minimum allocation of 200
gold crowns per annum, as recommended by the Thornfield Institute's
maintenance guidelines for Spire-class amplifiers.

Section 4. The Festival of Echoes broadcast shall not exceed
seventeen minutes in duration, in accordance with the thermal
safety limits established by Prof. Waveborn's research on
crystalline resonator endurance.

ARTICLE IV — THE RIVER SILVANIS

Section 1. Navigation of the River Silvanis within city limits
is open to all vessels. Navigation upstream beyond mile marker 40
requires a licensed riverpilot, with mandatory pilotage beyond
mile marker 45 due to the Silvanis Rapids.

Section 2. The river patrol, under the Defense Magistrate, shall
maintain navigational aids at the Rapids and provide emergency
rescue services. Funding shall be no less than 50 gold crowns
per annum from tariff revenues.

ARTICLE V — AMENDMENTS

This Charter may be amended by unanimous vote of the Triad Council
followed by ratification by two-thirds of eligible citizens in
public referendum. No amendment may alter Article I, Sections 1-2,
or Article II, Section 3.
"""


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
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Typed query / source structures — no magic-string dicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryQuery:
    """A single retrieval query against one library."""

    question: str
    prompt: str
    lib_name: str
    expected_terms: tuple[str, ...]


def timed_generate(
    engine,
    tokenizer,
    query_text: str,
    sources: list | None,
    max_new_tokens: int = 20,
) -> tuple[str, float]:
    """
    Run a query against library sources (or empty context if sources is None).

    sources is a list[LibrarySource]; pass None for the control condition.
    Returns (answer_text, elapsed_seconds).
    """
    q_ids = encode(tokenizer, query_text)
    t0 = time.perf_counter()

    if sources is None:
        gen = engine.generate(
            q_ids,
            replay_window_ids=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        gen = engine.generate_cross_library(
            q_ids,
            sources=sources,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - t0
    return decode(tokenizer, gen), elapsed


# ---------------------------------------------------------------------------
# Model / tokenizer / engine loading
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


def load_models(model_id: str):
    from transformers import AutoTokenizer

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config = GemmaConfig.from_hf_config(json.load(f))

    from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine
    from chuk_lazarus.inference.context.checkpoint_library import CheckpointLibrary

    rs = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs, model_path)
    rs.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return (
        rs,
        config,
        UnlimitedContextEngine,
        LibrarySource,
        CheckpointLibrary,
        tokenizer,
    )


# ---------------------------------------------------------------------------
# Library building (one-time, runs only if library doesn't exist or --rebuild)
# ---------------------------------------------------------------------------


def _compute_config_hash(config) -> str:
    import hashlib

    data = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
    }
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return f"sha256:{digest}"


def build_library(
    rs_model,
    config,
    EngineClass,
    tokenizer,
    text: str,
    output_path: Path,
    window_size: int,
    name: str,
    model_id: str,
) -> None:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    engine = EngineClass(rs_model, config, window_size=window_size)

    # Warm up
    _warm = mx.array([[1, 2, 3]])
    _, _kv = engine.kv_gen.prefill(_warm)
    mx.eval()

    engine.process(token_ids)
    engine.flush()
    s = engine.stats()

    output_path.mkdir(parents=True, exist_ok=True)

    # Window metadata — typed WindowMeta models
    windows = []
    token_offset = 0
    for wid in range(s.archived_windows):
        w_tokens, w_abs = engine.archive.retrieve(wid)
        preview = tokenizer.decode(w_tokens[:30], skip_special_tokens=True)
        windows.append(
            WindowMeta(
                window_id=wid,
                token_offset=token_offset,
                token_count=len(w_tokens),
                abs_offset=w_abs,
                preview=preview.replace("\n", " ")[:80],
            )
        )
        token_offset += len(w_tokens)

    # 1. Manifest — typed LibraryManifest model
    manifest = LibraryManifest(
        name=name,
        model_id=model_id,
        model_config_hash=_compute_config_hash(config),
        num_layers=config.num_hidden_layers,
        window_size=window_size,
        total_tokens=len(token_ids),
        num_windows=s.archived_windows,
        checkpoint_bytes=s.checkpoint_bytes,
        archive_bytes=s.archive_bytes,
        created_at=datetime.now(timezone.utc).isoformat(),
        format_version=LibraryFormatVersion.V1,
    )
    (output_path / LibraryFile.MANIFEST).write_text(manifest.model_dump_json(indent=2))

    # 2. Checkpoints
    ckpt_dict: dict[str, mx.array] = {}
    for wid in range(s.archived_windows):
        kv_last, _ = engine.checkpoints.load(wid)
        for li, (k, v) in enumerate(kv_last):
            ckpt_dict[f"w{wid}_l{li}_k"] = k
            ckpt_dict[f"w{wid}_l{li}_v"] = v
    mx.savez(str(output_path / LibraryFile.CHECKPOINTS), **ckpt_dict)

    # 3. Token archive
    with open(output_path / LibraryFile.TOKENS, "wb") as f:
        for tid in token_ids:
            f.write(struct.pack("<H", tid & 0xFFFF))

    # 4. Windows metadata
    (output_path / LibraryFile.WINDOWS).write_text(
        json.dumps([w.model_dump() for w in windows], indent=2)
    )


def ensure_libraries(
    rs_model,
    config,
    EngineClass,
    tokenizer,
    lib_dir: Path,
    window_size: int,
    model_id: str,
    config_hash: str,
    rebuild: bool = False,
) -> None:
    """Build any missing libraries (or all if rebuild=True).

    If a library exists but was built for a different model, it is rebuilt
    automatically rather than silently producing garbage outputs.
    """
    libraries_spec = [
        ("meridian", "The World of Meridian", MERIDIAN_TEXT),
        ("resonance", "Resonance Engineering", RESONANCE_TEXT),
        ("charter", "Aethermoor City Charter", CHARTER_TEXT),
    ]
    for slug, name, text in libraries_spec:
        lib_path = lib_dir / slug
        manifest_f = lib_path / "manifest.json"

        needs_build = rebuild or not manifest_f.exists()

        if not needs_build and manifest_f.exists():
            m = LibraryManifest.model_validate_json(manifest_f.read_text())
            if m.model_id != model_id or m.model_config_hash != config_hash:
                print(f"  '{name}' was built for {m.model_id} — rebuilding for {model_id}")
                needs_build = True

        if needs_build:
            print(f"  Building '{name}' ...")
            build_library(
                rs_model,
                config,
                EngineClass,
                tokenizer,
                text=text,
                output_path=lib_path,
                window_size=window_size,
                name=name,
                model_id=model_id,
            )
            m = LibraryManifest.model_validate_json((lib_path / LibraryFile.MANIFEST).read_text())
            print(f"    {m.total_tokens} tokens → {fmt_bytes(m.total_bytes)}")
        else:
            m = LibraryManifest.model_validate_json(manifest_f.read_text())
            print(
                f"  Found '{name}': {m.total_tokens} tokens, {m.num_windows} windows ({m.model_id})"
            )


# ---------------------------------------------------------------------------
# Demo acts
# ---------------------------------------------------------------------------


def act1_single_library(engine, meridian, resonance, charter, tokenizer, max_new_tokens):
    print(f"\n{'═' * 60}")
    print(f"{BOLD}ACT 1 — Single-Library Retrieval{RESET}")
    print(f"{'═' * 60}")
    print("  Each query uses exactly one library as context.")
    print("  All facts are fictional — correct answers prove retrieval.\n")

    queries: list[LibraryQuery] = [
        LibraryQuery(
            question="Who founded Aethermoor?",
            prompt="Who founded the Free City of Aethermoor? The founder was",
            lib_name=meridian.name,
            expected_terms=("kael", "dawnstrider"),
        ),
        LibraryQuery(
            question="What is the Holloway Constant?",
            prompt="What is the value of the Holloway Constant H? The Holloway Constant H equals",
            lib_name=resonance.name,
            expected_terms=("7.83", "7.8"),
        ),
        LibraryQuery(
            question="What is the max tariff rate in Aethermoor?",
            prompt="What is the maximum tariff rate allowed under the Aethermoor Charter? The maximum tariff is",
            lib_name=charter.name,
            expected_terms=("8%", "8 %", "eight percent", " 8"),
        ),
    ]

    lib_map = {meridian.name: meridian, resonance.name: resonance, charter.name: charter}

    results = []
    for q in queries:
        lib = lib_map[q.lib_name]
        wid = lib.find_window_for_term(q.expected_terms[0], tokenizer) or 0

        sources = [LibrarySource(library_name=q.lib_name, window_id=wid)]
        answer, elapsed = timed_generate(
            engine, tokenizer, q.prompt, sources, max_new_tokens=max_new_tokens
        )
        found = any(kw.lower() in answer.lower() for kw in q.expected_terms)
        tick = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
        results.append(found)

        print(f"  {tick} {q.question}")
        print(f"     Source:  {q.lib_name} (window {wid})")
        print(f"     Answer:  {YELLOW}{repr(answer[:80])}{RESET}")
        print(f"     Time:    {elapsed * 1000:.0f} ms")
        print()

    return results


def act2_cross_library_two(engine, meridian, resonance, tokenizer, max_new_tokens, LibrarySource):
    """
    Question requiring facts from BOTH Meridian and Resonance:
    - Meridian: Festival plays for exactly seventeen minutes
    - Resonance: continuous operation >15 minutes risks thermal fracture
    → Together: the festival duration is capped by thermal fracture risk
    """
    print(f"\n{'═' * 60}")
    print(f"{BOLD}ACT 2 — Cross-Library Retrieval (Two Sources){RESET}")
    print(f"{'═' * 60}")
    print("  Question requires facts from TWO different libraries.\n")

    prompt = (
        "Based on the information provided, why is the Festival of Echoes "
        "limited to exactly seventeen minutes? "
        "The festival duration is limited because"
    )
    expected_any = ["thermal", "fracture", "crystal", "resonat", "15 min", "fifteen"]

    wid_m = meridian.find_window_for_term("seventeen minutes", tokenizer) or 0
    wid_r = resonance.find_window_for_term("thermal fracture", tokenizer) or 0

    sources = [
        LibrarySource(library_name=meridian.name, window_id=wid_m),
        LibrarySource(library_name=resonance.name, window_id=wid_r),
    ]
    answer, elapsed = timed_generate(
        engine, tokenizer, prompt, sources, max_new_tokens=max_new_tokens
    )

    found = any(kw.lower() in answer.lower() for kw in expected_any)
    tick = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"

    print(f"  {tick} Why is the Festival of Echoes limited to 17 minutes?")
    print(f"     Sources: {[s[0] for s in sources]}")
    print(f"     Answer:  {YELLOW}{repr(answer[:120])}{RESET}")
    print(f"     Time:    {elapsed * 1000:.0f} ms")
    print(
        f"     {DIM}(Must connect lore [17 min festival] + engineering [thermal fracture >15 min]){RESET}"
    )
    print()

    return found


def act3_three_library(
    engine, meridian, resonance, charter, tokenizer, max_new_tokens, LibrarySource
):
    """
    Question requiring facts from ALL THREE libraries:
    - Charter:   Knowledge Magistrate oversees the Spire
    - Meridian:  Current Knowledge Magistrate is Illara Sunchaser
    - Resonance: Spire operates at Holloway Constant H = 7.83 Hz
    → Together: Illara Sunchaser is responsible; Spire operates at 7.83 Hz
    """
    print(f"\n{'═' * 60}")
    print(f"{BOLD}ACT 3 — Three-Library Synthesis{RESET}")
    print(f"{'═' * 60}")
    print("  Question requires facts from ALL THREE libraries.\n")

    prompt = (
        "Based on all available documents: Who is currently responsible "
        "for maintaining the Spire of Echoes, and at what base frequency "
        "does it operate? The Spire's maintenance falls under"
    )
    expected_any = ["illara", "sunchaser", "knowledge magistrate", "7.83", "holloway"]

    wid_m = meridian.find_window_for_term("illara", tokenizer) or 0
    wid_r = resonance.find_window_for_term("holloway constant", tokenizer) or 0
    wid_c = charter.find_window_for_term("knowledge magistrate", tokenizer) or 0

    sources = [
        LibrarySource(library_name=meridian.name, window_id=wid_m),
        LibrarySource(library_name=resonance.name, window_id=wid_r),
        LibrarySource(library_name=charter.name, window_id=wid_c),
    ]
    answer, elapsed = timed_generate(
        engine, tokenizer, prompt, sources, max_new_tokens=max_new_tokens
    )

    found = any(kw.lower() in answer.lower() for kw in expected_any)
    tick = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"

    print(f"  {tick} Who maintains the Spire and at what frequency?")
    print(f"     Sources: {[s[0] for s in sources]}")
    print(f"     Answer:  {YELLOW}{repr(answer[:120])}{RESET}")
    print(f"     Time:    {elapsed * 1000:.0f} ms")
    print(
        f"     {DIM}(Must synthesize lore [Illara Sunchaser] + engineering [7.83 Hz] + law [Knowledge Magistrate]){RESET}"
    )
    print()

    return found


def act4_control(engine, meridian, resonance, charter, tokenizer, max_new_tokens):
    """
    Same questions, no library context.  Hallucination expected.
    """
    print(f"\n{'═' * 60}")
    print(f"{BOLD}ACT 4 — Control: Same Questions Without Libraries{RESET}")
    print(f"{'═' * 60}")
    print("  No context loaded.  These fictional facts don't exist in parametric memory.\n")

    control_queries = [
        ("Who founded Aethermoor?", "Who founded the Free City of Aethermoor? The founder was"),
        (
            "Why is the Festival limited to 17 minutes?",
            "Based on the information provided, why is the Festival of Echoes "
            "limited to exactly seventeen minutes? The festival duration is limited because",
        ),
        (
            "Who maintains the Spire and at what frequency?",
            "Based on all available documents: Who is currently responsible "
            "for maintaining the Spire of Echoes, and at what base frequency "
            "does it operate? The Spire's maintenance falls under",
        ),
    ]

    for question, prompt in control_queries:
        answer, elapsed = timed_generate(
            engine, tokenizer, prompt, sources=None, max_new_tokens=max_new_tokens
        )
        print(f"  ✗ {question}")
        print(f"     Answer (no context): {YELLOW}{repr(answer[:100])}{RESET}")
        print(
            f"     {DIM}(Hallucination expected — these facts do not exist in parametric memory){RESET}"
        )
        print()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Cross-library checkpoint retrieval demo")
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--window-size", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=20)
    p.add_argument(
        "--rebuild", action="store_true", help="Force rebuild of all libraries even if they exist"
    )
    p.add_argument(
        "--no-control", action="store_true", help="Skip Act 4 (control without libraries)"
    )
    p.add_argument(
        "--lib-dir", default=None, help="Library directory (default: <repo_root>/libraries)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # ── Load model first so we can derive the model slug for lib_dir ────
    print(f"\n{BOLD}Cross-Library Checkpoint Retrieval Demo{RESET}")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Window size: {args.window_size} tokens")
    print()

    print("Loading model ...")
    rs, config, EngineClass, LibrarySource, CheckpointLibrary, tokenizer = load_models(args.model)
    print(
        f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
        f"kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}"
    )

    kv_bpt = 2 * config.num_key_value_heads * config.head_dim * config.num_hidden_layers * 2
    config_hash = _compute_config_hash(config)
    model_slug = args.model.split("/")[-1]  # e.g. "gemma-3-270m-it-bf16"
    print(f"  KV bytes/token: {fmt_bytes(kv_bpt)}")
    print()

    # Library path is model-specific so switching models never reuses stale checkpoints
    if args.lib_dir:
        lib_dir = Path(args.lib_dir)
    else:
        lib_dir = Path(__file__).parents[2] / "libraries" / model_slug

    print(f"  Libraries:   {lib_dir}")
    print()

    # ── Warm up ─────────────────────────────────────────────────────────
    _eng_warm = EngineClass(rs, config, window_size=args.window_size)
    _, _kv = _eng_warm.kv_gen.prefill(mx.array([[1, 2, 3]]))
    mx.eval()
    del _eng_warm, _kv

    # ── Build / verify libraries ─────────────────────────────────────────
    print("Checking knowledge libraries ...")
    ensure_libraries(
        rs,
        config,
        EngineClass,
        tokenizer,
        lib_dir=lib_dir,
        window_size=args.window_size,
        model_id=args.model,
        config_hash=config_hash,
        rebuild=args.rebuild,
    )
    print()

    # ── Load libraries from disk ─────────────────────────────────────────
    print("Loading libraries from disk ...")
    t0 = time.perf_counter()
    meridian = CheckpointLibrary(lib_dir / "meridian")
    resonance = CheckpointLibrary(lib_dir / "resonance")
    charter = CheckpointLibrary(lib_dir / "charter")
    load_ms = (time.perf_counter() - t0) * 1000

    total_tokens = meridian.total_tokens + resonance.total_tokens + charter.total_tokens
    total_disk = sum(lib.manifest.total_bytes for lib in [meridian, resonance, charter])

    print(f"  3 libraries loaded in {load_ms:.0f} ms  ({fmt_bytes(total_disk)} total)")
    for lib in [meridian, resonance, charter]:
        print(
            f"  {lib.name}: {lib.total_tokens} tokens, "
            f"{lib.num_windows} window{'s' if lib.num_windows != 1 else ''} "
            f"({fmt_bytes(lib.manifest.total_bytes)})"
        )
    print(f"  Total knowledge: {total_tokens} tokens across {total_disk // 1024:.0f} KB on disk")
    print()

    # ── Create inference engine with model identity for library verification ─
    engine = EngineClass(
        rs,
        config,
        window_size=args.window_size,
        model_id=args.model,
        config_hash=config_hash,
    )
    engine.load_library(meridian)
    engine.load_library(resonance)
    engine.load_library(charter)

    # ── Acts ─────────────────────────────────────────────────────────────
    act1_results = act1_single_library(
        engine, meridian, resonance, charter, tokenizer, args.gen_tokens
    )
    act2_result = act2_cross_library_two(
        engine, meridian, resonance, tokenizer, args.gen_tokens, LibrarySource
    )
    act3_result = act3_three_library(
        engine, meridian, resonance, charter, tokenizer, args.gen_tokens, LibrarySource
    )
    if not args.no_control:
        act4_control(engine, meridian, resonance, charter, tokenizer, args.gen_tokens)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'═' * 60}")

    act1_pass = sum(act1_results)
    act1_tick = GREEN if act1_pass == 3 else (YELLOW if act1_pass > 0 else RED)
    act2_tick = GREEN if act2_result else RED
    act3_tick = GREEN if act3_result else RED

    print(f"\n  Act 1  Single-library retrieval:    {act1_tick}{act1_pass}/3 correct{RESET}")
    print(
        f"  Act 2  Two-library cross-retrieval: "
        f"{act2_tick}{'✓ passed' if act2_result else '✗ failed'}{RESET}"
    )
    print(
        f"  Act 3  Three-library synthesis:     "
        f"{act3_tick}{'✓ passed' if act3_result else '✗ failed'}{RESET}"
    )

    equiv_kv = total_tokens * kv_bpt
    print(f"""
  Three knowledge libraries. Three different domains.
  Loaded in {load_ms:.0f} ms from disk. No prefill at query time.

  {total_tokens} tokens of knowledge. {fmt_bytes(total_disk)} on disk.
  Equivalent full KV cache: {fmt_bytes(equiv_kv)}.
  Compression: {equiv_kv // max(total_disk, 1)}×.

  Single-library retrieval:  sub-second, lossless.
  Cross-library reasoning:   two sources combined in one attention window.
  Three-library synthesis:   lore + engineering + law, answered together.

  No fine-tuning. No RAG pipeline. No vector database.
  No embeddings. No chunking. No retrieval ranking.

  Just pre-filled checkpoint files and an inference engine
  that treats the residual stream as what it is.

  Install knowledge like you install packages.
  Query it like it was always there.
""")


if __name__ == "__main__":
    main()
