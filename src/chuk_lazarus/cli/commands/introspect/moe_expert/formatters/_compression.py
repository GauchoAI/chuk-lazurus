"""Compression-related formatters for MoE expert CLI output."""

from __future__ import annotations

from ......introspection.moe import (
    OverlayRepresentation,
    ReconstructionVerification,
    StorageEstimate,
)
from ._base import format_header


def format_overlay_result(result: OverlayRepresentation) -> str:
    """Format overlay computation result for display.

    Args:
        result: Overlay representation result.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("OVERLAY REPRESENTATION"),
        f"Model:   {result.model_id}",
        f"Layer:   {result.layer_idx}",
        f"Experts: {result.num_experts}",
        "",
        "Projection Analysis:",
        f"  Gate:  rank={result.gate_rank:<4} shape={result.gate.shape}",
        f"         compression: {result.gate.compression_ratio:.1f}x",
        f"  Up:    rank={result.up_rank:<4} shape={result.up.shape}",
        f"         compression: {result.up.compression_ratio:.1f}x",
        f"  Down:  rank={result.down_rank:<4} shape={result.down.shape}",
        f"         compression: {result.down.compression_ratio:.1f}x",
        "",
        "Storage:",
        f"  Original:   {result.total_original_bytes / (1024 * 1024):>8.1f} MB",
        f"  Compressed: {result.total_compressed_bytes / (1024 * 1024):>8.1f} MB",
        f"  Ratio:      {result.compression_ratio:>8.1f}x",
        "=" * 70,
    ]
    return "\n".join(lines)


def format_verification_result(result: ReconstructionVerification) -> str:
    """Format reconstruction verification result for display.

    Args:
        result: Reconstruction verification result.

    Returns:
        Formatted output string.
    """
    status = "PASSED" if result.passed else "FAILED"
    status_marker = "✓" if result.passed else "✗"

    lines = [
        format_header("RECONSTRUCTION VERIFICATION"),
        f"Model:  {result.model_id}",
        f"Layer:  {result.layer_idx}",
        f"Status: {status_marker} {status}",
        "",
        f"Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}",
        "",
        "Weight Reconstruction Errors:",
        f"  Gate:  {result.gate.mean_relative_error:.6f} (max: {result.gate.max_relative_error:.6f})",
        f"  Up:    {result.up.mean_relative_error:.6f} (max: {result.up.max_relative_error:.6f})",
        f"  Down:  {result.down.mean_relative_error:.6f} (max: {result.down.max_relative_error:.6f})",
        "",
        "Output Reconstruction Errors:",
        f"  Mean:  {result.mean_output_error:.6f}",
        f"  Max:   {result.max_output_error:.6f}",
        "",
        f"Quality: {'<1% error - suitable for production' if result.passed else '>1% error - increase ranks'}",
        "=" * 70,
    ]
    return "\n".join(lines)


def format_storage_estimate(result: StorageEstimate) -> str:
    """Format storage estimate result for display.

    Args:
        result: Storage estimate result.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("STORAGE ESTIMATE"),
        f"Model:   {result.model_id}",
        f"Layers:  {result.num_layers} MoE layers",
        f"Experts: {result.num_experts} per layer",
        "",
        f"Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}",
        "",
        "Full Model Storage:",
        f"  Original:   {result.original_mb:>10.1f} MB",
        f"  Compressed: {result.compressed_mb:>10.1f} MB",
        f"  Savings:    {result.savings_mb:>10.1f} MB ({result.compression_ratio:.1f}x)",
        "",
        "Breakdown:",
        f"  Base experts (shared):     {result.compressed_mb / result.compression_ratio:>6.1f} MB",
        f"  Low-rank deltas:           {result.compressed_mb - result.compressed_mb / result.compression_ratio:>6.1f} MB",
        "=" * 70,
    ]
    return "\n".join(lines)
