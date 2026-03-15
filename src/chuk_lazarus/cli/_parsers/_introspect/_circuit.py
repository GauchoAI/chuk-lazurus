"""Circuit introspect parsers."""

import argparse
import asyncio

from ...commands.introspect import (
    introspect_circuit_capture,
    introspect_circuit_compare,
    introspect_circuit_decode,
    introspect_circuit_export,
    introspect_circuit_invoke,
    introspect_circuit_test,
    introspect_circuit_view,
)


def register_circuit_parsers(introspect_subparsers):
    """Register circuit sub-subcommands."""
    circuit_parser = introspect_subparsers.add_parser(
        "circuit",
        help="Direct circuit capture, interpolation, and invocation",
        description="""Experimental: Capture and manipulate computation circuits directly.

Subcommands:
  capture   - Capture activations for a computation (e.g., "7 * 4 = 28")
  invoke    - Interpolate/combine captured circuits to compute new values
  decode    - Decode activations back to tokens/answers

Examples:
    # Capture multiplication examples
    lazarus introspect circuit capture -m model \\
        --prompts "7*4=28|6*8=48|9*3=27" --layer 19 --save mult_circuit.npz

    # Invoke circuit with new operands (interpolate)
    lazarus introspect circuit invoke -m model \\
        --circuit mult_circuit.npz --operands "5,6" --layer 19

    # Decode what answer the circuit produces
    lazarus introspect circuit decode -m model \\
        --activations circuit_state.npz --layer 19
        """,
    )
    circuit_subparsers = circuit_parser.add_subparsers(dest="circuit_command")

    # Circuit capture
    capture_parser = circuit_subparsers.add_parser(
        "capture",
        help="Capture circuit activations for known computations",
    )
    capture_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    capture_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Computation prompts (pipe-separated, e.g., '7*4=|6*8=' or '7*4=28|6*8=48')",
    )
    capture_parser.add_argument(
        "--results",
        "-r",
        help="Expected results (pipe-separated, e.g., '28|48') - use with prompts like '7*4=|6*8='",
    )
    capture_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        required=True,
        help="Layer to capture activations from",
    )
    capture_parser.add_argument(
        "--save",
        "-o",
        required=True,
        help="Save captured circuit to .npz file",
    )
    capture_parser.add_argument(
        "--extract-direction",
        action="store_true",
        help="Extract and save the direction that encodes the result value",
    )
    capture_parser.add_argument(
        "--position",
        choices=["last", "answer", "operator"],
        default="last",
        help="Position to capture: last token, answer position, or operator position",
    )
    capture_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_capture(args)))

    # Circuit invoke
    invoke_parser = circuit_subparsers.add_parser(
        "invoke",
        help="Invoke circuit with new operands via interpolation",
    )
    invoke_parser.add_argument(
        "--model",
        "-m",
        help="Model name or HuggingFace ID (required for 'steer' method)",
    )
    invoke_parser.add_argument(
        "--circuit",
        "-c",
        required=True,
        help="Captured circuit file (.npz)",
    )
    invoke_parser.add_argument(
        "--operands",
        help="New operands to compute (pipe-separated pairs, e.g., '5,6|8,9')",
    )
    invoke_parser.add_argument(
        "--prompts",
        "-p",
        dest="invoke_prompts",
        help="Prompts to run through circuit (for 'steer' method, e.g., '5*6=|8*9=')",
    )
    invoke_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer for circuit (default: from circuit file)",
    )
    invoke_parser.add_argument(
        "--method",
        choices=["steer", "interpolate", "extrapolate", "linear"],
        default="linear",
        help="How to invoke circuit: steer (uses direction), linear/interpolate/extrapolate (uses activations)",
    )
    invoke_parser.add_argument(
        "--output",
        "-o",
        help="Save result to file",
    )
    invoke_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_invoke(args)))

    # Circuit decode
    decode_parser = circuit_subparsers.add_parser(
        "decode",
        help="Decode circuit activations to see what answer they produce",
    )
    decode_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    decode_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Base prompt to inject activations into",
    )
    decode_parser.add_argument(
        "--inject",
        "-i",
        required=True,
        help="Activations to inject (.npz file)",
    )
    decode_parser.add_argument(
        "--inject-idx",
        type=int,
        default=0,
        help="Index of activation to inject (default: 0)",
    )
    decode_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to inject at",
    )
    decode_parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend factor (0=original, 1=full injection)",
    )
    decode_parser.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens to generate",
    )
    decode_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_decode(args)))

    # Circuit test - apply trained direction to new activations (proper OOD testing)
    test_parser = circuit_subparsers.add_parser(
        "test",
        help="Test if a circuit generalizes to new inputs",
    )
    test_parser.add_argument(
        "--circuit",
        "-c",
        required=True,
        help="Trained circuit file (.npz from 'circuit capture --extract-direction')",
    )
    # Option 1: Provide pre-captured activations
    test_parser.add_argument(
        "--test-activations",
        "-t",
        help="Pre-captured test activations (.npz file)",
    )
    # Option 2: Capture on the fly with model + prompts
    test_parser.add_argument(
        "--model",
        "-m",
        help="Model to use for capturing test activations",
    )
    test_parser.add_argument(
        "--prompts",
        "-p",
        help="Test prompts (e.g., '1*1=|11*11=|10*5=')",
    )
    test_parser.add_argument(
        "--results",
        "-r",
        help="Expected results (e.g., '1|121|50')",
    )
    test_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    test_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_test(args)))

    # Circuit compare - compare multiple circuits (directions)
    compare_circuit_parser = circuit_subparsers.add_parser(
        "compare",
        help="Compare multiple circuits (e.g., add vs mult vs div)",
        description="""Compare the directions/circuits extracted for different operations.

Shows:
- Cosine similarity between circuit directions
- Angle between circuits (orthogonal = independent computations)
- Top neurons for each circuit

Example:
    lazarus introspect circuit compare \\
        -c mult_circuit.npz add_circuit.npz sub_circuit.npz div_circuit.npz
        """,
    )
    compare_circuit_parser.add_argument(
        "--circuits",
        "-c",
        nargs="+",
        required=True,
        help="Circuit files to compare (.npz files from 'circuit capture --extract-direction')",
    )
    compare_circuit_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to show per circuit (default: 10)",
    )
    compare_circuit_parser.add_argument(
        "--output",
        "-o",
        help="Save comparison results to JSON file",
    )
    compare_circuit_parser.set_defaults(
        func=lambda args: asyncio.run(introspect_circuit_compare(args))
    )

    # Circuit view - display circuit contents
    view_parser = circuit_subparsers.add_parser(
        "view",
        help="View the contents of a captured circuit file",
        description="""Display circuit metadata, entries, and optionally as a formatted table.

Examples:
    # Basic view (shows first 20 entries)
    lazarus introspect circuit view -c mult_complete_table.npz

    # Show as multiplication/addition table grid
    lazarus introspect circuit view -c mult_complete_table.npz --table

    # Show with direction statistics and top neurons
    lazarus introspect circuit view -c mult_complete_table.npz --stats

    # Show all entries
    lazarus introspect circuit view -c mult_complete_table.npz --limit 0
        """,
    )
    view_parser.add_argument(
        "--circuit",
        "-c",
        required=True,
        help="Circuit file to view (.npz)",
    )
    view_parser.add_argument(
        "--table",
        "-t",
        action="store_true",
        help="Display as a formatted grid (for arithmetic circuits)",
    )
    view_parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show direction statistics and top neurons",
    )
    view_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Max entries to show in list view (0 for all, default: 20)",
    )
    view_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to show with --stats (default: 10)",
    )
    view_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_view(args)))

    # Circuit export - export circuit to various formats
    export_parser = circuit_subparsers.add_parser(
        "export",
        help="Export circuit graph to DOT, JSON, Mermaid, or HTML format",
        description="""Export ablation or direction results as a circuit graph.

Supports multiple output formats:
- DOT (Graphviz): For rendering with graphviz tools
- JSON: For programmatic processing
- Mermaid: For embedding in documentation
- HTML: Interactive visualization using vis.js

Examples:
    # Export ablation results to DOT
    lazarus introspect circuit export -i ablation_results.json -o circuit.dot --format dot

    # Export to interactive HTML
    lazarus introspect circuit export -i ablation_results.json -o circuit.html --format html

    # Export directions to Mermaid diagram
    lazarus introspect circuit export -i directions.json -o circuit.md --format mermaid
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file (ablation results JSON or directions JSON)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["dot", "json", "mermaid", "html"],
        default="json",
        help="Output format (default: json)",
    )
    export_parser.add_argument(
        "--type",
        choices=["ablation", "directions"],
        default="ablation",
        help="Input data type: ablation results or extracted directions (default: ablation)",
    )
    export_parser.add_argument(
        "--name",
        help="Circuit name (default: derived from input file)",
    )
    export_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum effect threshold for ablation circuits (default: 0.1)",
    )
    export_parser.add_argument(
        "--direction",
        choices=["TB", "LR", "BT", "RL"],
        default="TB",
        help="Graph direction: TB (top-bottom), LR (left-right), etc. (default: TB)",
    )
    export_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_export(args)))
