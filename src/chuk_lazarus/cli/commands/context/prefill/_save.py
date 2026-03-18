"""Library save orchestrator — coordinates writing all library files."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from .._types import KVectorMode, ResidualMode


def compute_config_hash(config) -> str:
    """Stable hash of the model config key fields."""
    data = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
    }
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return f"sha256:{digest}"


def save_library(
    engine,
    output_path: Path,
    all_token_ids: list[int],
    name: str,
    model_id: str,
    config,
    window_size: int,
    tokenizer,
    created_at: str,
    is_complete: bool = False,
    quick: bool = False,
    residual_mode: ResidualMode = ResidualMode.INTERVAL,
    frame_bank_data: dict | None = None,
    frame_bank_path: Path | None = None,
    store_pages: bool = False,
    append_from: int = 0,
    run_interval: bool = True,
    run_compass: bool = True,
    run_darkspace: bool = True,
    run_pages: bool = True,
    run_surprise: bool = True,
    run_sparse: bool = True,
    run_kvectors: bool = True,
    run_vec_inject: bool = False,
    run_mode7: bool = False,
    compass_layer: int | None = None,
    kvector_mode: KVectorMode = KVectorMode.SPARSE,
) -> None:
    """Write all library files from the engine's current archived state.

    quick=True writes only checkpoints/tokens/windows/manifest (for periodic saves).
    quick=False also runs extraction passes gated by the run_* flags.

    append_from: when > 0, only serialize windows [append_from, num_archived) into
    checkpoints/residuals by appending to existing zip files.  Windows metadata,
    tokens, and manifest are always fully rewritten (they're small).
    """
    from .....inference.context import (
        LibraryFile,
        LibraryFormatVersion,
        LibraryManifest,
        WindowMeta,
    )

    s = engine.stats()
    num_archived = s.archived_windows
    if num_archived == 0:
        return

    output_path.mkdir(parents=True, exist_ok=True)

    skip_checkpoints = residual_mode == ResidualMode.DARKSPACE

    # Collect window metadata for ALL windows (small — always rewritten)
    windows: list[WindowMeta] = []
    token_offset = 0
    for wid in range(num_archived):
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

    total_tokens_to_report = len(all_token_ids) if is_complete else token_offset

    # --- Checkpoints and residuals ---
    if not skip_checkpoints:
        from ._checkpoints import save_checkpoints, save_residuals

        save_checkpoints(engine, output_path, num_archived, append_from, quiet=quick)
        save_residuals(engine, output_path, num_archived, append_from)
    else:
        if not quick:
            print(
                f"  darkspace mode: skipping checkpoints (fresh prefill at generate time)",
                file=sys.stderr, flush=True,
            )

    # --- Post-prefill extraction passes (skipped during periodic quick saves) ---
    if not quick:
        if residual_mode == ResidualMode.DARKSPACE and run_darkspace:
            from ._darkspace import extract_darkspace

            extract_darkspace(
                engine, output_path, num_archived, config,
                frame_bank_data, frame_bank_path,
            )

        elif residual_mode in (ResidualMode.INTERVAL, ResidualMode.FULL):
            if run_interval:
                from ._interval import extract_interval_residuals

                extract_interval_residuals(
                    engine, output_path, num_archived, residual_mode,
                )

        # Compass runs for any residual mode when requested
        if run_compass:
            from ._compass import calibrate_compass

            compass_n_samples = None if residual_mode == ResidualMode.FULL else 8
            calibrate_compass(
                engine, output_path, num_archived, config,
                n_samples=compass_n_samples,
                compass_layer=compass_layer,
            )

        # Surprise: per-token perplexity scoring (anomaly detection)
        if run_surprise:
            from ._surprise import extract_surprise

            extract_surprise(engine, output_path, num_archived, config)

        # Sparse: keyword extraction for Mode 5 sparse semantic index
        if run_sparse:
            # Check if engine is SparseIndexEngine with inline extraction already done
            if hasattr(engine, 'sparse_index') and len(engine.sparse_index) > 0:
                # Inline extraction — index was built during _close_window()
                # Just save it. Zero additional compute.
                engine.sparse_index.save(output_path / "sparse_index.json")
                stats = engine.sparse_index.stats()
                size_kb = (output_path / "sparse_index.json").stat().st_size / 1024
                parametric_count = sum(
                    1 for e in engine.sparse_index.entries if not e.keywords
                )
                print(
                    f"  Sparse index (inline): {stats['non_empty']} novel, "
                    f"{parametric_count} parametric, "
                    f"{stats['total_keywords']} keywords "
                    f"({stats['avg_keywords']:.1f}/window), "
                    f"{size_kb:.0f} KB",
                    file=sys.stderr,
                )
            else:
                # Fallback: separate extraction pass (forward pass per window)
                from ._sparse import extract_sparse

                extract_sparse(engine, tokenizer, output_path, num_archived)

        # K-vector routing index: L29 H4 K vectors at fact positions
        if run_kvectors:
            from ._kv_route import extract_kv_route_index

            extract_kv_route_index(engine, output_path, num_archived, config, kvector_mode=kvector_mode)

        # Vec injection index: K vectors + coefficients c = dot(R_L30, embed(token))
        if run_vec_inject:
            from ._vec_inject import extract_vec_inject_index

            extract_vec_inject_index(
                engine, output_path, num_archived, config,
                tokenizer=tokenizer, kvector_mode=kvector_mode,
            )

        # Mode 7: calibrate query classifier + engagement/tension probes
        if run_mode7:
            from ._mode7_calibrate import calibrate_mode7_probes

            calibrate_mode7_probes(
                engine, output_path, num_archived, config,
                tokenizer, model_id, compass_layer,
            )

    # --- Pages ---
    if store_pages and run_pages:
        from ._pages import extract_pages

        extract_pages(engine, output_path, num_archived)

    # --- tokens.bin (uint32) ---
    with open(output_path / LibraryFile.TOKENS, "wb") as f:
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            for tid in w_tokens:
                f.write(struct.pack("<I", tid))

    # --- windows.json ---
    (output_path / LibraryFile.WINDOWS).write_text(
        json.dumps([w.model_dump() for w in windows], indent=2, ensure_ascii=False)
    )

    # --- manifest.json (written last — the "committed" marker) ---
    manifest = LibraryManifest(
        name=name,
        model_id=model_id,
        model_config_hash=compute_config_hash(config),
        num_layers=config.num_hidden_layers,
        window_size=window_size,
        total_tokens=total_tokens_to_report,
        num_windows=num_archived,
        checkpoint_bytes=s.checkpoint_bytes,
        archive_bytes=s.archive_bytes,
        created_at=created_at,
        format_version=LibraryFormatVersion.V1,
    )
    (output_path / LibraryFile.MANIFEST).write_text(manifest.model_dump_json(indent=2))
