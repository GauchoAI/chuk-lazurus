#!/usr/bin/env python3
"""
Needle-in-a-Haystack Benchmark for Sparse Semantic Index
Builds keyword indexes from the Apollo 11 transcript windows
and outputs test prompts for evaluation.
"""
import json
import struct
import re
from pathlib import Path
from collections import Counter

LIBRARY_DIR = Path("/Users/christopherhay/chris-source/apollo-demo/apollo11_lean")
TRANSCRIPT_PATH = Path("/Users/christopherhay/chris-source/apollo-demo/docs/apollo11_clean.txt")

# The needle
NEEDLE = (
    "Professor Alaric Thornfield discovered the high-frequency "
    "resonance pattern of crystallized helium at the Voss "
    "Institute in Greenland in 2019."
)

NEEDLE_KEYWORDS = "Thornfield: Voss Institute, Greenland, crystallized helium resonance, 2019"

# Additional needles for multi-needle experiments
MULTI_NEEDLES = [
    {
        "sentence": "Professor Alaric Thornfield discovered the high-frequency resonance pattern of crystallized helium at the Voss Institute in Greenland in 2019.",
        "keywords": "Thornfield: Voss Institute, Greenland, crystallized helium resonance, 2019",
        "query": "Where did Professor Thornfield discover the resonance pattern?",
        "expected": "Voss Institute in Greenland",
    },
    {
        "sentence": "Dr. Helena Marchetti of the Zurich Quantum Lab published groundbreaking results on topological qubit arrays in 2023.",
        "keywords": "Marchetti: Zurich Quantum Lab, topological qubit arrays, 2023",
        "query": "Where did Dr. Marchetti publish results on qubit arrays?",
        "expected": "Zurich Quantum Lab",
    },
    {
        "sentence": "Engineer Kazuki Yamamoto developed the phase-locked plasma containment system at the Osaka Research Center in 2021.",
        "keywords": "Yamamoto: Osaka Research Center, phase-locked plasma containment, 2021",
        "query": "Where did Yamamoto develop the plasma containment system?",
        "expected": "Osaka Research Center",
    },
    {
        "sentence": "Dr. Fatima Al-Rashid proved the existence of tachyonic neutrino oscillations at CERN Laboratory B7 in 2020.",
        "keywords": "Al-Rashid: CERN Laboratory B7, tachyonic neutrino oscillations, 2020",
        "query": "Where did Dr. Al-Rashid prove tachyonic neutrino oscillations?",
        "expected": "CERN Laboratory B7",
    },
    {
        "sentence": "Professor Mikhail Petrov synthesized the first stable sample of metallic hydrogen at the Novosibirsk Institute in 2018.",
        "keywords": "Petrov: Novosibirsk Institute, metallic hydrogen, 2018",
        "query": "Where did Professor Petrov synthesize metallic hydrogen?",
        "expected": "Novosibirsk Institute",
    },
    {
        "sentence": "Dr. Amara Osei-Bonsu detected gravitational wave echoes from binary magnetar collisions at the Accra Deep Space Array in 2022.",
        "keywords": "Osei-Bonsu: Accra Deep Space Array, gravitational wave echoes, binary magnetar, 2022",
        "query": "Where did Dr. Osei-Bonsu detect gravitational wave echoes?",
        "expected": "Accra Deep Space Array",
    },
    {
        "sentence": "Engineer Sofia Lindqvist designed the zero-point energy harvester prototype at the Uppsala Advanced Physics Center in 2024.",
        "keywords": "Lindqvist: Uppsala Advanced Physics Center, zero-point energy harvester, 2024",
        "query": "Where did Lindqvist design the zero-point energy harvester?",
        "expected": "Uppsala Advanced Physics Center",
    },
    {
        "sentence": "Dr. Chen Wei-Lin measured the anomalous magnetic moment of the tau lepton at Fermilab Building 42 in 2017.",
        "keywords": "Chen Wei-Lin: Fermilab Building 42, tau lepton anomalous magnetic moment, 2017",
        "query": "Where did Dr. Chen Wei-Lin measure the tau lepton moment?",
        "expected": "Fermilab Building 42",
    },
    {
        "sentence": "Professor Rodrigo Espinoza calculated the theoretical mass of the gravitino particle at the Santiago Astrophysics Observatory in 2016.",
        "keywords": "Espinoza: Santiago Astrophysics Observatory, gravitino particle mass, 2016",
        "query": "Where did Espinoza calculate the gravitino mass?",
        "expected": "Santiago Astrophysics Observatory",
    },
    {
        "sentence": "Dr. Ingrid Bjornsdottir isolated the dark photon interaction signature at the Reykjavik Particle Collider in 2025.",
        "keywords": "Bjornsdottir: Reykjavik Particle Collider, dark photon interaction, 2025",
        "query": "Where did Dr. Bjornsdottir isolate the dark photon signature?",
        "expected": "Reykjavik Particle Collider",
    },
]

# Needle types for Experiment 3
NEEDLE_TYPES = {
    "entity_rich": {
        "sentence": "Dr. Helena Marchetti of the Zurich Quantum Lab published groundbreaking results on topological qubit arrays in 2023.",
        "keywords": "Marchetti: Zurich Quantum Lab, topological qubit arrays, 2023",
        "query": "Where did Dr. Marchetti publish results on qubit arrays?",
        "expected": "Zurich Quantum Lab",
    },
    "numeric": {
        "sentence": "The reactor core temperature reached 4721 degrees at precisely 14:23:07 on the third day of the experiment.",
        "keywords": "reactor: 4721 degrees, 14:23:07, third day",
        "query": "What temperature did the reactor core reach?",
        "expected": "4721",
    },
    "dialogue": {
        "sentence": 'The mission controller said to the crew: we need you to hold position for another forty minutes, there is an anomaly in the telemetry data that we cannot explain.',
        "keywords": "controller: hold position, forty minutes, telemetry anomaly",
        "query": "How long did the mission controller ask the crew to hold position?",
        "expected": "forty minutes",
    },
    "implicit": {
        "sentence": "The crew fell silent for nearly three minutes after the alarm sounded. When communication resumed, the commander's voice was noticeably different.",
        "keywords": "crew: silent three minutes, alarm, commander voice changed",
        "query": "How long was the crew silent after the alarm?",
        "expected": "three minutes",
    },
}


def load_transcript():
    """Load the full cleaned transcript."""
    return TRANSCRIPT_PATH.read_text(encoding="utf-8")


def load_windows_metadata():
    """Load window metadata from the pre-built library."""
    with open(LIBRARY_DIR / "windows.json") as f:
        return json.load(f)


def load_manifest():
    """Load manifest."""
    with open(LIBRARY_DIR / "manifest.json") as f:
        return json.load(f)


def split_transcript_into_lines(text, num_windows=725):
    """Split transcript into roughly equal chunks (line-based approximation).
    Since we don't have the tokenizer, we split by character count
    to approximate 512-token windows (~2048 chars each).
    """
    lines = text.split("\n")
    total_chars = len(text)
    chars_per_window = total_chars / num_windows

    windows = []
    current_window = []
    current_chars = 0

    for line in lines:
        current_window.append(line)
        current_chars += len(line) + 1  # +1 for newline

        if current_chars >= chars_per_window and len(windows) < num_windows - 1:
            windows.append("\n".join(current_window))
            current_window = []
            current_chars = 0

    # Last window gets remaining text
    if current_window:
        windows.append("\n".join(current_window))

    return windows


def extract_keywords(window_text, window_id):
    """Extract keywords from a window using simple heuristics.

    Rules:
    1. Capitalized words (proper nouns, acronyms)
    2. Numbers (timestamps, measurements)
    3. Technical terms (multi-word capitalized phrases)
    4. Skip common transcript markers (CDR, CC, LMP, CMP, Roger, Over)
    """
    # Common transcript words to skip (they appear in most windows)
    STOP_WORDS = {
        "CDR", "CC", "CMP", "LMP", "SC", "MS", "Roger", "Over",
        "Apollo", "Houston", "The", "And", "That", "This", "For",
        "We", "You", "It", "Go", "Are", "Is", "Was", "Were",
        "A", "An", "In", "On", "At", "To", "Of", "By", "Or",
        "OK", "Okay", "Copy", "Affirmative", "Negative",
    }

    keywords = set()

    # Find capitalized words (potential proper nouns)
    caps = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b', window_text)
    for w in caps:
        if w not in STOP_WORDS and len(w) > 2:
            keywords.add(w)

    # Find ALL-CAPS words (acronyms, technical terms)
    acronyms = re.findall(r'\b([A-Z]{2,}(?:[-/][A-Z]{2,})*)\b', window_text)
    for a in acronyms:
        if a not in STOP_WORDS and len(a) > 1:
            keywords.add(a)

    # Find numbers that look like measurements or coordinates
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', window_text)
    significant_numbers = [n for n in numbers if len(n) >= 3 or "." in n]
    for n in significant_numbers[:3]:  # max 3 numbers per window
        keywords.add(n)

    # Limit to top 8 keywords per window
    return list(keywords)[:8]


def build_sparse_index(windows, needle_positions=None, needles=None):
    """Build a sparse keyword index from window texts.

    Args:
        windows: list of window text strings
        needle_positions: dict of {window_id: needle_index} for needle placement
        needles: list of needle dicts (from MULTI_NEEDLES)

    Returns:
        index_text: the sparse index as a string
        index_entries: list of (window_id, keywords) tuples
    """
    if needle_positions is None:
        needle_positions = {}
    if needles is None:
        needles = [MULTI_NEEDLES[0]]

    entries = []

    for i, window_text in enumerate(windows):
        # Check if this window has a needle
        if i in needle_positions:
            needle_idx = needle_positions[i]
            needle = needles[needle_idx] if needle_idx < len(needles) else needles[0]
            keywords = needle["keywords"]
        else:
            kw_list = extract_keywords(window_text, i)
            if not kw_list:
                continue
            keywords = ", ".join(kw_list)

        entries.append((i, keywords))

    # Build index text
    lines = []
    for wid, kw in entries:
        lines.append(f"W{wid}: {kw}")

    return "\n".join(lines), entries


def build_query_prompt(index_text, query):
    """Build a complete prompt with index + query."""
    return (
        f"<start_of_turn>user\n"
        f"The following is a keyword index extracted from a long document. "
        f"Use it to answer the question.\n\n"
        f"Index:\n{index_text}\n\n"
        f"Question: {query}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def position_sweep(windows, positions_pct):
    """Generate test cases for Phase 1a: position sweep at fixed doc length."""
    num_windows = len(windows)
    results = []

    for pct in positions_pct:
        window_id = max(0, min(int(pct / 100.0 * num_windows), num_windows - 1))

        # Build index with needle at this position
        needle_positions = {window_id: 0}
        index_text, entries = build_sparse_index(windows, needle_positions)

        prompt = build_query_prompt(
            index_text,
            "Where did Professor Thornfield discover the resonance pattern?"
        )

        results.append({
            "position_pct": pct,
            "window_id": window_id,
            "num_entries": len(entries),
            "index_tokens_approx": len(index_text.split()),
            "prompt": prompt,
            "prompt_chars": len(prompt),
        })

    return results


def document_length_sweep(windows, lengths_tokens, needle_pct=50):
    """Generate test cases for Phase 1b: document length sweep at fixed position."""
    results = []
    tokens_per_window = 512

    for length in lengths_tokens:
        num_windows = max(1, length // tokens_per_window)
        subset = windows[:num_windows]

        # Plant needle at middle
        needle_window = max(0, int(needle_pct / 100.0 * num_windows))
        needle_positions = {needle_window: 0}
        index_text, entries = build_sparse_index(subset, needle_positions)

        prompt = build_query_prompt(
            index_text,
            "Where did Professor Thornfield discover the resonance pattern?"
        )

        results.append({
            "doc_length_tokens": length,
            "num_windows": num_windows,
            "needle_window": needle_window,
            "num_entries": len(entries),
            "index_tokens_approx": len(index_text.split()),
            "prompt": prompt,
            "prompt_chars": len(prompt),
        })

    return results


def multi_needle_test(windows, needle_counts):
    """Generate test cases for Phase 2a: increasing needle count."""
    num_windows = len(windows)
    results = []

    for count in needle_counts:
        if count > len(MULTI_NEEDLES):
            count = len(MULTI_NEEDLES)

        # Distribute needles evenly
        spacing = num_windows // (count + 1)
        needle_positions = {}
        for i in range(count):
            wid = spacing * (i + 1)
            needle_positions[wid] = i

        index_text, entries = build_sparse_index(
            windows, needle_positions, MULTI_NEEDLES[:count]
        )

        # Build queries for each needle
        queries = []
        for i in range(count):
            n = MULTI_NEEDLES[i]
            prompt = build_query_prompt(index_text, n["query"])
            queries.append({
                "needle_idx": i,
                "query": n["query"],
                "expected": n["expected"],
                "prompt": prompt,
                "prompt_chars": len(prompt),
            })

        results.append({
            "needle_count": count,
            "num_entries": len(entries),
            "index_tokens_approx": len(index_text.split()),
            "queries": queries,
        })

    return results


if __name__ == "__main__":
    # Load data
    transcript = load_transcript()
    manifest = load_manifest()
    windows_meta = load_windows_metadata()

    print(f"Transcript: {len(transcript)} chars")
    print(f"Library: {manifest['num_windows']} windows, {manifest['total_tokens']} tokens")

    # Split transcript into windows
    windows = split_transcript_into_lines(transcript, manifest['num_windows'])
    print(f"Split into {len(windows)} windows")

    # Show sample extraction
    print("\n=== Sample keyword extraction ===")
    for i in [0, 100, 363, 500, 700]:
        if i < len(windows):
            kw = extract_keywords(windows[i], i)
            print(f"W{i}: {', '.join(kw[:8])}")

    # Phase 1a: Position sweep
    print("\n=== Phase 1a: Position Sweep ===")
    positions = [0, 10, 25, 50, 75, 90, 100]
    pos_tests = position_sweep(windows, positions)
    for t in pos_tests:
        print(f"  Position {t['position_pct']}% (W{t['window_id']}): "
              f"{t['num_entries']} entries, ~{t['index_tokens_approx']} index tokens, "
              f"{t['prompt_chars']} prompt chars")

    # Phase 1b: Document length sweep
    print("\n=== Phase 1b: Document Length Sweep ===")
    lengths = [5000, 25000, 50000, 100000, 200000, 370000]
    len_tests = document_length_sweep(windows, lengths)
    for t in len_tests:
        print(f"  {t['doc_length_tokens']}tok ({t['num_windows']}w): "
              f"{t['num_entries']} entries, ~{t['index_tokens_approx']} index tokens, "
              f"{t['prompt_chars']} prompt chars")

    # Phase 2a: Multi-needle
    print("\n=== Phase 2a: Multi-Needle ===")
    counts = [1, 3, 5, 10]
    multi_tests = multi_needle_test(windows, counts)
    for t in multi_tests:
        print(f"  {t['needle_count']} needles: "
              f"{t['num_entries']} entries, ~{t['index_tokens_approx']} index tokens")

    # Needle types
    print("\n=== Phase 3: Needle Types ===")
    for ntype, ndata in NEEDLE_TYPES.items():
        print(f"  {ntype}: keywords='{ndata['keywords']}'")

    print("\n=== Index size analysis ===")
    # Full index (all windows, no needle)
    full_index, full_entries = build_sparse_index(windows)
    print(f"Full index: {len(full_entries)} entries, "
          f"{len(full_index)} chars, ~{len(full_index.split())} tokens")

    # Print first 20 entries of full index
    print("\nFirst 20 entries:")
    for wid, kw in full_entries[:20]:
        print(f"  W{wid}: {kw}")
