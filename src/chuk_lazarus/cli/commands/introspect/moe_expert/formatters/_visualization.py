"""Visualization formatters for MoE expert CLI output."""

from __future__ import annotations

import math

import numpy as np

from ......introspection.moe import MoETypeAnalysis
from ._base import format_header


def _compute_2d_embedding(
    similarity_matrix: tuple[tuple[float, ...], ...],
    num_experts_to_show: int = 8,
) -> list[tuple[float, float, int]]:
    """Compute 2D embedding of experts from similarity matrix using classical MDS.

    Args:
        similarity_matrix: Pairwise cosine similarities (num_experts x num_experts)
        num_experts_to_show: Number of experts to embed (default 8 for clarity)

    Returns:
        List of (x, y, expert_idx) tuples for 2D positions
    """
    n = min(len(similarity_matrix), num_experts_to_show)

    # Convert similarity to distance (angle-based)
    # cos(theta) = similarity, so theta = arccos(similarity)
    # Use distance = 1 - similarity for simplicity (bounded [0, 2])
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim = similarity_matrix[i][j]
            # Clamp to valid range for arccos
            sim = max(-1.0, min(1.0, sim))
            # Use angle as distance
            dist_matrix[i, j] = math.acos(sim)

    # Classical MDS: convert distance matrix to coordinates
    # 1. Square the distances
    D_sq = dist_matrix**2

    # 2. Double-center the matrix
    n_pts = n
    H = np.eye(n_pts) - np.ones((n_pts, n_pts)) / n_pts
    B = -0.5 * H @ D_sq @ H

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 2 dimensions
    # Handle negative eigenvalues (set to small positive)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    coords_2d = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

    # Normalize to [-1, 1] range
    max_abs = np.max(np.abs(coords_2d)) + 1e-10
    coords_2d = coords_2d / max_abs

    return [(float(coords_2d[i, 0]), float(coords_2d[i, 1]), i) for i in range(n)]


def _draw_direction_diagram(
    coords: list[tuple[float, float, int]],
    width: int = 61,
    height: int = 25,
) -> list[str]:
    """Draw ASCII diagram with arrows showing expert directions.

    Args:
        coords: List of (x, y, expert_idx) in [-1, 1] range
        width: Diagram width in characters
        height: Diagram height in characters

    Returns:
        List of strings representing the ASCII diagram
    """
    # Create empty grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Center point
    cx, cy = width // 2, height // 2

    # Draw border
    for x in range(width):
        grid[0][x] = "─"
        grid[height - 1][x] = "─"
    for y in range(height):
        grid[y][0] = "│"
        grid[y][width - 1] = "│"
    grid[0][0] = "┌"
    grid[0][width - 1] = "┐"
    grid[height - 1][0] = "└"
    grid[height - 1][width - 1] = "┘"

    # Draw axes
    for x in range(2, width - 2):
        grid[cy][x] = "─"
    for y in range(2, height - 2):
        grid[y][cx] = "│"
    grid[cy][cx] = "┼"

    # Arrow characters based on direction (8 directions)
    def get_arrow_char(dx: float, dy: float) -> str:
        """Get arrow character based on direction."""
        angle = math.atan2(-dy, dx)  # Negative dy because y increases downward
        # Normalize to [0, 2pi)
        if angle < 0:
            angle += 2 * math.pi

        # 8 directions: →, ↗, ↑, ↖, ←, ↙, ↓, ↘
        arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
        idx = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        return arrows[idx]

    # Draw expert arrows
    for x, y, expert_idx in coords:
        # Convert [-1, 1] to grid coordinates
        # Leave margin for border and labels
        margin_x = 4
        margin_y = 2
        gx = int(cx + x * (width // 2 - margin_x))
        gy = int(cy - y * (height // 2 - margin_y))  # Flip y

        # Clamp to valid range
        gx = max(2, min(width - 3, gx))
        gy = max(2, min(height - 3, gy))

        # Draw arrow from center towards this point
        # Draw the expert label at the endpoint
        arrow = get_arrow_char(x, y)

        # Draw a line from center to the point
        steps = max(abs(gx - cx), abs(gy - cy))
        if steps > 0:
            for step in range(1, steps + 1):
                px = cx + int((gx - cx) * step / steps)
                py = cy + int((gy - cy) * step / steps)
                if 2 <= px < width - 2 and 2 <= py < height - 2:
                    if step == steps:
                        # Endpoint: show arrow and label
                        if px + 2 < width - 1:
                            label = f"{arrow}E{expert_idx}"
                            for i, c in enumerate(label):
                                if px + i < width - 1:
                                    grid[py][px + i] = c
                        else:
                            grid[py][px] = arrow
                    elif grid[py][px] in [" ", "─", "│"]:
                        # Path: show direction
                        if abs(gx - cx) > abs(gy - cy) * 2:
                            grid[py][px] = "─"
                        elif abs(gy - cy) > abs(gx - cx) * 2:
                            grid[py][px] = "│"
                        else:
                            grid[py][px] = "·"

    return ["".join(row) for row in grid]


def format_orthogonality_ascii(result: MoETypeAnalysis, *, max_display: int = 16) -> str:
    """Format ASCII visualization of expert orthogonality from actual similarity data.

    Creates a data-driven heatmap and directional diagram showing expert relationships.

    Args:
        result: MoE type analysis result with similarity_matrix.
        max_display: Maximum number of experts to display (default 16 for readability).

    Returns:
        ASCII art visualization string with heatmap and direction diagram.
    """
    sim = result.mean_cosine_similarity
    is_orthogonal = sim < 0.10
    is_clustered = sim > 0.25

    lines = [
        format_header("EXPERT ORTHOGONALITY VISUALIZATION"),
        f"Model:   {result.model_id}",
        f"Layer:   {result.layer_idx}",
        f"Experts: {result.num_experts}",
        f"Type:    {result.moe_type.value.upper()}",
        "",
    ]

    # Add directional diagram if we have the matrix
    if result.similarity_matrix:
        lines.append("Expert Direction Diagram (2D MDS projection):")
        lines.append("")

        # Compute 2D embedding and draw diagram
        num_to_show = min(8, len(result.similarity_matrix))
        coords = _compute_2d_embedding(result.similarity_matrix, num_to_show)
        diagram_lines = _draw_direction_diagram(coords)
        lines.extend(diagram_lines)

        # Add explanation
        lines.append("")
        if is_orthogonal:
            lines.append("  Arrows point in different directions → Experts are ORTHOGONAL")
        elif is_clustered:
            lines.append("  Arrows cluster together → Experts SHARE a common base")
        else:
            lines.append("  Mixed directions → Expert structure is ambiguous")

        lines.append("")

    # Add similarity heatmap if we have the matrix
    if result.similarity_matrix:
        lines.append("Expert Similarity Heatmap (cosine similarity):")
        lines.append("")

        # Determine how many experts to show
        num_experts = len(result.similarity_matrix)
        display_count = min(num_experts, max_display)

        # Heatmap characters from low to high similarity
        # Using intensity blocks: ░ ▒ ▓ █ for different similarity ranges
        def sim_to_char(s: float) -> str:
            """Convert similarity value to heatmap character."""
            if s >= 0.99:  # Diagonal (self-similarity)
                return "■"
            elif s >= 0.5:
                return "█"
            elif s >= 0.3:
                return "▓"
            elif s >= 0.15:
                return "▒"
            elif s >= 0.05:
                return "░"
            else:
                return "·"

        # Create header row with expert indices
        if display_count <= 10:
            header = "     " + " ".join(f"{i:2d}" for i in range(display_count))
        else:
            # Compact header for many experts
            header = "   " + "".join(f"{i % 10}" for i in range(display_count))

        lines.append(header)

        # Create heatmap rows
        for i in range(display_count):
            row_chars = []
            for j in range(display_count):
                similarity = result.similarity_matrix[i][j]
                row_chars.append(sim_to_char(similarity))

            if display_count <= 10:
                row = f" {i:2d}  " + "  ".join(row_chars)
            else:
                row = f"{i:2d} " + "".join(row_chars)

            # Add row summary
            row_sims = [result.similarity_matrix[i][j] for j in range(display_count) if i != j]
            if row_sims:
                avg_sim = sum(row_sims) / len(row_sims)
                row += f"  avg:{avg_sim:.2f}"

            lines.append(row)

        if num_experts > max_display:
            lines.append(f"     ... ({num_experts - max_display} more experts not shown)")

        # Add legend
        lines.append("")
        lines.append(
            "Legend: · (<0.05)  ░ (0.05-0.15)  ▒ (0.15-0.3)  ▓ (0.3-0.5)  █ (>0.5)  ■ (self)"
        )

        # Add similarity distribution summary
        lines.append("")
        lines.append("Similarity Distribution:")

        # Count similarities in each range
        all_sims = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                all_sims.append(result.similarity_matrix[i][j])

        if all_sims:
            ranges = [
                ("≈0 (orthogonal)", lambda s: s < 0.05),
                ("0.05-0.15", lambda s: 0.05 <= s < 0.15),
                ("0.15-0.30", lambda s: 0.15 <= s < 0.30),
                ("0.30-0.50", lambda s: 0.30 <= s < 0.50),
                (">0.50 (similar)", lambda s: s >= 0.50),
            ]

            for label, condition in ranges:
                count = sum(1 for s in all_sims if condition(s))
                pct = count / len(all_sims) * 100
                bar = "█" * int(pct / 5)
                lines.append(f"  {label:<18} {count:4d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("Summary Statistics:")
    lines.append(f"  Mean Similarity:   {result.mean_cosine_similarity:.3f}")
    lines.append(f"  Std Deviation:     {result.std_cosine_similarity:.3f}")
    lines.append(f"  Gate Rank Ratio:   {result.gate.rank_ratio * 100:.1f}%")
    lines.append("")

    # Add interpretation based on actual data
    if is_clustered:
        lines.extend(
            [
                "Interpretation: PSEUDO-MoE (Clustered Experts)",
                "  ╔═══════════════════════════════════════════════════════════════╗",
                "  ║  High similarity indicates experts share a common BASE.       ║",
                "  ║  Model was likely converted from dense → MoE (upcycling).     ║",
                "  ║                                                               ║",
                "  ║     Expert[i] = BASE + low_rank_delta[i]                      ║",
                "  ║                                                               ║",
                "  ║  ✓ COMPRESSIBLE via SVD overlay representation                ║",
                f"  ║  Estimated compression: {result.estimated_compression:.1f}x"
                + " " * (37 - len(f"{result.estimated_compression:.1f}"))
                + "║",
                "  ╚═══════════════════════════════════════════════════════════════╝",
            ]
        )
    elif is_orthogonal:
        lines.extend(
            [
                "Interpretation: NATIVE-MoE (Orthogonal Experts)",
                "  ╔═══════════════════════════════════════════════════════════════╗",
                "  ║  Low similarity indicates experts are genuinely different.    ║",
                "  ║  Model was trained natively as MoE from scratch.              ║",
                "  ║                                                               ║",
                "  ║     Expert[i] ⟂ Expert[j]  (orthogonal)                       ║",
                "  ║                                                               ║",
                "  ║  ✗ NOT compressible via SVD overlay                           ║",
                "  ║  Use quantization/pruning instead                             ║",
                "  ╚═══════════════════════════════════════════════════════════════╝",
            ]
        )
    else:
        lines.extend(
            [
                "Interpretation: UNKNOWN (Ambiguous Structure)",
                "  ╔═══════════════════════════════════════════════════════════════╗",
                "  ║  Mixed similarity pattern - neither fully clustered           ║",
                "  ║  nor fully orthogonal.                                        ║",
                "  ║                                                               ║",
                "  ║  Possible causes:                                             ║",
                "  ║  - Partial MoE training / fine-tuning                         ║",
                "  ║  - Hybrid architecture                                        ║",
                "  ║  - Model in transition state                                  ║",
                "  ║                                                               ║",
                "  ║  ? Compression potential unclear - test empirically           ║",
                "  ╚═══════════════════════════════════════════════════════════════╝",
            ]
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
