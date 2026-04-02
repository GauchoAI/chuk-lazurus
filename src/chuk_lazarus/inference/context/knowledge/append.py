"""Dynamic append — add a single skill/document to an existing knowledge store.

Instead of rebuilding the entire store, this module:
1. Loads (or creates) a "base state" — the residual from a system prompt.
2. Processes a single new document from that base state (NOT chained from
   previous skills, so each skill is independent).
3. Appends the new window(s), boundary, entries, and keywords to the store.
4. Recomputes IDF globally and saves the updated store.

Usage:
    from .append import append_skill
    append_skill(kv_gen, tokenizer, store_path, new_doc_path, config)

Or via CLI:
    lazarus knowledge append -m <model> -s <store> -i <markdown_file>
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np

from .build import _select_targets
from .config import ArchitectureConfig
from .route import SparseKeywordIndex, TFIDFRouter, extract_window_keywords
from .store import (
    BOUNDARIES_DIR,
    BOUNDARY_RESIDUAL_FILE,
    ENTRIES_FILE,
    IDF_FILE,
    KEYWORDS_FILE,
    MANIFEST_FILE,
    RESIDUALS_DIR,
    STORE_VERSION,
    WINDOW_TOKEN_LISTS_FILE,
    WINDOW_TOKENS_FILE,
    InjectionEntry,
    KnowledgeStore,
    _entries_to_numpy,
    _numpy_to_entries,
)

# ── File constants for base state ────────────────────────────────────

BASE_STATE_FILE = "base_state.npy"


# ── Phase 1: Base State ─────────────────────────────────────────────


def build_base_state(
    kv_gen,
    tokenizer,
    system_prompt: str,
    config: ArchitectureConfig,
    store_path: Path,
) -> mx.array:
    """Process a system prompt and save the resulting residual as base_state.

    Returns the base state residual: (1, 1, hidden_dim) float32.
    """
    tokens = tokenizer.encode(system_prompt, add_special_tokens=True)
    token_mx = mx.array(tokens)[None]

    h = kv_gen.prefill_to_layer(
        token_mx,
        target_layer=config.crystal_layer,
        initial_residual=None,
    )

    # Take the final position as the base state
    base_state = h[:, -1:, :]  # (1, 1, hidden_dim)
    mx.eval(base_state)

    # Save to disk
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    base_np = np.array(base_state[0, 0, :].tolist(), dtype=np.float32)
    np.save(str(store_path / BASE_STATE_FILE), base_np)

    print(f"  Base state saved: {store_path / BASE_STATE_FILE} "
          f"({base_np.nbytes / 1024:.1f} KB)", file=sys.stderr)

    return base_state


def load_base_state(store_path: Path) -> mx.array:
    """Load the base state from disk.

    Returns (1, 1, hidden_dim) float32.
    """
    path = Path(store_path) / BASE_STATE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Base state not found at {path}. "
            "Run 'lazarus knowledge init' first to create it from a system prompt."
        )
    base_np = np.load(str(path))
    base = mx.array(base_np, dtype=mx.float32).reshape(1, 1, -1)
    mx.eval(base)
    return base


# ── Phase 2: Index helpers ───────────────────────────────────────────


def load_index(store_path: Path) -> dict:
    """Load the full index state from disk. Returns a dict with all components.

    Handles empty/non-existent stores gracefully.
    """
    store_path = Path(store_path)
    result = {
        "entries": [],
        "window_tokens": {},
        "window_token_lists": {},
        "idf": {},
        "keywords": {},
        "num_windows": 0,
        "num_tokens": 0,
        "config": None,
    }

    if not store_path.exists():
        return result

    # Manifest
    manifest_path = store_path / MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        result["config"] = ArchitectureConfig.from_dict(manifest.get("arch_config", {}))
        result["num_windows"] = manifest.get("num_windows", 0)
        result["num_tokens"] = manifest.get("num_tokens", 0)

    # Entries
    entries_path = store_path / ENTRIES_FILE
    if entries_path.exists():
        npz = np.load(str(entries_path), allow_pickle=False)
        if len(npz["entries"]) > 0:
            result["entries"] = _numpy_to_entries(npz["entries"])

    # Window tokens
    wt_path = store_path / WINDOW_TOKENS_FILE
    if wt_path.exists():
        wt_npz = np.load(str(wt_path), allow_pickle=False)
        for key in wt_npz.files:
            result["window_tokens"][int(key)] = {int(t) for t in wt_npz[key]}

    # Window token lists
    wtl_path = store_path / WINDOW_TOKEN_LISTS_FILE
    if wtl_path.exists():
        wtl_npz = np.load(str(wtl_path), allow_pickle=False)
        for key in wtl_npz.files:
            result["window_token_lists"][int(key)] = [int(t) for t in wtl_npz[key]]

    # IDF
    idf_path = store_path / IDF_FILE
    if idf_path.exists():
        idf_raw = json.loads(idf_path.read_text())
        result["idf"] = {int(k): float(v) for k, v in idf_raw.items()}

    # Keywords
    kw_path = store_path / KEYWORDS_FILE
    if kw_path.exists():
        kw_raw = json.loads(kw_path.read_text())
        result["keywords"] = {int(k): v for k, v in kw_raw.items()}

    return result


def save_index(store_path: Path, index: dict, config: ArchitectureConfig) -> None:
    """Save updated index components back to disk (atomic-ish)."""
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    # Entries
    entries = index["entries"]
    if entries:
        entry_arr = _entries_to_numpy(entries)
        np.savez(str(store_path / ENTRIES_FILE), entries=entry_arr)
    else:
        from .store import _ENTRY_DTYPE
        np.savez(str(store_path / ENTRIES_FILE), entries=np.array([], dtype=_ENTRY_DTYPE))

    # Window tokens
    wt_data = {}
    for wid, tokens in index["window_tokens"].items():
        wt_data[str(wid)] = np.array(sorted(tokens), dtype=np.uint32)
    np.savez(str(store_path / WINDOW_TOKENS_FILE), **wt_data)

    # Window token lists
    wtl_data = {}
    for wid, token_list in index["window_token_lists"].items():
        wtl_data[str(wid)] = np.array(token_list, dtype=np.uint32)
    np.savez(str(store_path / WINDOW_TOKEN_LISTS_FILE), **wtl_data)

    # IDF
    idf_serializable = {str(k): v for k, v in index["idf"].items()}
    (store_path / IDF_FILE).write_text(json.dumps(idf_serializable, indent=1) + "\n")

    # Keywords
    kw_serializable = {str(k): v for k, v in index["keywords"].items()}
    (store_path / KEYWORDS_FILE).write_text(json.dumps(kw_serializable, indent=1) + "\n")

    # Manifest
    has_residuals = (store_path / RESIDUALS_DIR).exists()
    manifest = {
        "version": STORE_VERSION,
        "num_entries": len(entries),
        "num_windows": index["num_windows"],
        "num_tokens": index["num_tokens"],
        "entries_per_window": config.entries_per_window,
        "crystal_layer": config.crystal_layer,
        "window_size": config.window_size,
        "arch_config": config.to_dict(),
        "has_residuals": has_residuals,
    }
    (store_path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")


# ── Phase 3: append_skill ────────────────────────────────────────────


def append_skill(
    kv_gen,
    tokenizer,
    store_path: Path | str,
    new_doc_path: Path | str,
    config: ArchitectureConfig,
    progress_fn: Callable[[str, float], None] | None = None,
) -> dict:
    """Append a single skill/document to an existing knowledge store.

    Parameters
    ----------
    kv_gen       : KVDirectGenerator instance (model already loaded).
    tokenizer    : Tokenizer for the model.
    store_path   : Path to the knowledge store directory.
    new_doc_path : Path to the new Markdown/text file.
    config       : ArchitectureConfig.
    progress_fn  : Optional callback(stage: str, progress: float 0-1).

    Returns
    -------
    dict with keys: skill_id, num_new_windows, num_new_tokens, elapsed_s
    """
    store_path = Path(store_path)
    new_doc_path = Path(new_doc_path)
    t0 = time.monotonic()

    def _progress(stage: str, pct: float):
        if progress_fn:
            progress_fn(stage, pct)

    # ── Step 1: Load base state ───────────────────────────────────────
    _progress("loading_base_state", 0.0)
    base_state = load_base_state(store_path)

    # ── Step 2: Load existing index ───────────────────────────────────
    _progress("loading_index", 0.05)
    index = load_index(store_path)

    old_num_windows = index["num_windows"]
    old_max_fact_id = max((e.fact_id for e in index["entries"]), default=-1)

    # ── Step 3: Tokenize new document ─────────────────────────────────
    _progress("tokenizing", 0.1)
    text = new_doc_path.read_text(encoding="utf-8")
    doc_tokens = tokenizer.encode(text, add_special_tokens=False)

    window_size = config.window_size
    num_new_windows = math.ceil(len(doc_tokens) / window_size) if doc_tokens else 0

    # Chunk into windows
    new_windows: list[tuple[int, list[int]]] = []
    for i in range(num_new_windows):
        wid = old_num_windows + i
        start = i * window_size
        end = min(start + window_size, len(doc_tokens))
        new_windows.append((wid, doc_tokens[start:end]))

    print(f"  New document: {len(doc_tokens)} tokens, {num_new_windows} windows "
          f"(IDs {old_num_windows}..{old_num_windows + num_new_windows - 1})",
          file=sys.stderr)

    # ── Step 4: Process windows from base state ───────────────────────
    # Each skill starts from the base state (NOT chained from prior skills)
    _progress("building_boundaries", 0.15)

    bnd_dir = store_path / BOUNDARIES_DIR
    bnd_dir.mkdir(exist_ok=True)
    res_dir = store_path / RESIDUALS_DIR
    res_dir.mkdir(exist_ok=True)

    boundary_residual = base_state  # Start from base state
    sparse_index = SparseKeywordIndex()

    for i, (wid, chunk_ids) in enumerate(new_windows):
        # Forward pass through crystal layer
        w_ids = mx.array(chunk_ids)[None]
        h = kv_gen.prefill_to_layer(
            w_ids,
            target_layer=config.crystal_layer,
            initial_residual=boundary_residual,
        )

        # Chain boundary within this skill's windows
        boundary_residual = h[:, -1:, :]
        mx.eval(boundary_residual)

        # Save boundary
        bnd_np = np.array(boundary_residual[0, 0, :].tolist(), dtype=np.float32)
        np.save(str(bnd_dir / f"window_{wid:03d}.npy"), bnd_np)

        # Save residual stream
        offset = 1 if i > 0 else 0
        stream = h[0, offset:, :]
        mx.eval(stream)
        stream_np = np.array(stream.tolist(), dtype=np.float16)
        np.save(str(res_dir / f"window_{wid:03d}.npy"), stream_np)

        del h, stream

        # Update window tokens and keywords
        index["window_tokens"][wid] = set(chunk_ids)
        index["window_token_lists"][wid] = chunk_ids

        kws = extract_window_keywords(chunk_ids, tokenizer)
        index["keywords"][wid] = kws
        sparse_index.add(wid, kws)

        pct = 0.15 + 0.5 * ((i + 1) / num_new_windows)
        _progress("building_boundaries", pct)

        print(f"\r  Window {i + 1}/{num_new_windows} (ID {wid})", end="", file=sys.stderr)

    print(file=sys.stderr)

    # ── Step 5: Keyword expansion (Pass 3) ────────────────────────────
    _progress("keyword_expansion", 0.65)

    from ._sampling import sample_token

    for i, (wid, chunk_ids) in enumerate(new_windows):
        window_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        topic_prompt = f"{window_text}\n\nTopics:"
        topic_ids = tokenizer.encode(topic_prompt, add_special_tokens=True)
        topic_mx = mx.array(topic_ids)[None]

        logits, kv_store = kv_gen.prefill(topic_mx)
        mx.eval(logits)
        seq_len = topic_mx.shape[1]

        for _ in range(5):
            token = sample_token(logits[0, -1], 0.0)
            index["window_tokens"][wid].add(token)
            kw_text = tokenizer.decode([token]).strip()
            if kw_text and len(kw_text) >= 2:
                kw_lower = kw_text.lower()
                index["keywords"][wid].append(kw_lower)
                sparse_index.add(wid, [kw_lower])
                for variant in [kw_lower, f" {kw_lower}", kw_text, f" {kw_text}"]:
                    var_ids = tokenizer.encode(variant, add_special_tokens=False)
                    for vid in var_ids:
                        index["window_tokens"][wid].add(vid)
            logits, kv_store = kv_gen.step_uncompiled(
                mx.array([[token]]), kv_store, seq_len=seq_len
            )
            seq_len += 1

        pct = 0.65 + 0.2 * ((i + 1) / num_new_windows)
        _progress("keyword_expansion", pct)

        print(f"\r  Keywords {i + 1}/{num_new_windows}", end="", file=sys.stderr)

    print(file=sys.stderr)

    # ── Step 6: Build injection entries ────────────────────────────────
    _progress("building_entries", 0.85)

    # Recompute IDF globally (all windows, old + new)
    # Use smoothed IDF: log((N+1) / df) to handle single-window case
    # where standard log(N/df) = log(1/1) = 0 for all tokens
    n_windows = len(index["window_tokens"])
    if n_windows > 0:
        token_df: dict[int, int] = {}
        for tokens in index["window_tokens"].values():
            for t in tokens:
                token_df[t] = token_df.get(t, 0) + 1
        index["idf"] = {t: math.log((n_windows + 1) / df) for t, df in token_df.items()}
    else:
        index["idf"] = {}

    # Select targets and create entries for new windows only
    fact_id_counter = old_max_fact_id + 1
    embed_matrix = kv_gen.backbone.embed_matrix
    new_entries: list[InjectionEntry] = []

    for wid, chunk_ids in new_windows:
        target_list = _select_targets(chunk_ids, index["idf"], min_k=config.entries_per_window)
        for token_id, pos_in_window in target_list:
            embed = embed_matrix[token_id]
            natural_coeff = float(mx.linalg.norm(embed).item())
            stored_coeff = config.inject_coefficient * natural_coeff
            new_entries.append(
                InjectionEntry(
                    token_id=token_id,
                    coefficient=stored_coeff,
                    window_id=wid,
                    position_in_window=pos_in_window,
                    fact_id=fact_id_counter,
                )
            )
            fact_id_counter += 1

    index["entries"].extend(new_entries)
    index["num_windows"] = old_num_windows + num_new_windows
    index["num_tokens"] += len(doc_tokens)

    # ── Step 7: Save updated boundary_residual ────────────────────────
    if boundary_residual is not None:
        br_np = np.array(boundary_residual[0, 0, :].tolist(), dtype=np.float32)
        np.save(str(store_path / BOUNDARY_RESIDUAL_FILE), br_np)

    # ── Step 8: Save updated index ────────────────────────────────────
    _progress("saving_index", 0.95)
    save_index(store_path, index, config)

    # ── Phase 4: Memory cleanup ───────────────────────────────────────
    del boundary_residual, base_state
    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass  # Not on Metal
    gc.collect()

    elapsed = time.monotonic() - t0
    _progress("done", 1.0)

    result = {
        "skill_id": new_doc_path.stem,
        "num_new_windows": num_new_windows,
        "num_new_tokens": len(doc_tokens),
        "num_new_entries": len(new_entries),
        "total_windows": index["num_windows"],
        "total_entries": len(index["entries"]),
        "elapsed_s": round(elapsed, 2),
    }

    print(f"  Appended '{new_doc_path.name}': {num_new_windows} windows, "
          f"{len(new_entries)} entries in {elapsed:.1f}s", file=sys.stderr)
    print(f"  Store total: {index['num_windows']} windows, "
          f"{len(index['entries'])} entries", file=sys.stderr)

    return result
