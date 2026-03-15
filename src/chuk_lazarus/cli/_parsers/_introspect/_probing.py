"""Probe, neurons, cluster introspect parsers."""

import asyncio

from ...commands.introspect import (
    introspect_activation_cluster,
    introspect_neurons,
    introspect_probe,
)


def register_probing_parsers(introspect_subparsers):
    """Register probe, neurons, and cluster subcommands."""
    # Probe command - train linear probe to find task classification layers
    probe_parser = introspect_subparsers.add_parser(
        "probe",
        help="Train linear probe to find task classification layers",
        description="""Train logistic regression probes at each layer to find where
the model classifies different types of prompts.

This reveals task classification in ACTIVATION SPACE (not logit space).

Example:
    lazarus introspect probe -m model \\
        --class-a "2+2=|45*45=|100-37=" --label-a math \\
        --class-b "Capital of France?|Write a poem" --label-b other
        """,
    )
    probe_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    probe_parser.add_argument(
        "--class-a",
        required=True,
        help="Class A prompts (pipe-separated or @file.txt)",
    )
    probe_parser.add_argument(
        "--class-b",
        required=True,
        help="Class B prompts (pipe-separated or @file.txt)",
    )
    probe_parser.add_argument(
        "--label-a",
        default="class_a",
        help="Label for class A (default: 'class_a')",
    )
    probe_parser.add_argument(
        "--label-b",
        default="class_b",
        help="Label for class B (default: 'class_b')",
    )
    probe_parser.add_argument(
        "--test",
        "-t",
        help="Test prompts to classify after training (pipe-separated or @file.txt)",
    )
    probe_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    probe_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Specific layer to probe (default: find best layer)",
    )
    probe_parser.add_argument(
        "--save-direction",
        help="Save extracted direction vector to .npz file",
    )
    probe_parser.add_argument(
        "--method",
        choices=["logistic", "difference"],
        default="logistic",
        help="Direction extraction method: 'logistic' (probe weights) or 'difference' (mean difference)",
    )
    probe_parser.set_defaults(func=lambda args: asyncio.run(introspect_probe(args)))

    # Neurons command - analyze individual neuron activations
    neurons_parser = introspect_subparsers.add_parser(
        "neurons",
        help="Analyze individual neuron activations across prompts",
        description="""Show how specific neurons fire across different prompts.

Useful for understanding what individual neurons encode after running a probe.

Examples:
    # Analyze top neurons from probe across prompts
    lazarus introspect neurons -m model -l 15 \\
        --prompts "2+2=|45*45=|47*47=|67*83=" \\
        --neurons 808,1190,1168,891

    # Load neurons from saved direction
    lazarus introspect neurons -m model -l 15 \\
        --prompts "2+2=|47*47=" \\
        --from-direction difficulty.npz --top-k 6

    # Track neuron across multiple layers
    lazarus introspect neurons -m model --layers 15,19,20,21 \\
        --prompts "10*10=|25*25=|100*100=|17*19=|23*29=|47*53=" \\
        --neurons 1930 \\
        --labels "trivial|easy|memorized|medium|hard|hardest"
        """,
    )
    neurons_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    neurons_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Single layer to analyze (use --layers for multiple)",
    )
    neurons_parser.add_argument(
        "--layers",
        help="Multiple layers to analyze (comma-separated, e.g., '15,19,20,21')",
    )
    neurons_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to analyze (pipe-separated or @file.txt)",
    )
    neurons_parser.add_argument(
        "--neurons",
        "-n",
        help="Neuron indices to analyze (comma-separated, e.g., '808,1190,1168')",
    )
    neurons_parser.add_argument(
        "--neuron-names",
        help="Names for neurons (pipe-separated, same order as --neurons, e.g., 'Confidence|Computation|Effort')",
    )
    neurons_parser.add_argument(
        "--from-direction",
        help="Load top neurons from saved direction .npz file",
    )
    neurons_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to show when using --from-direction or --auto-discover (default: 10)",
    )
    neurons_parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Automatically discover discriminative neurons by variance/separation across prompts",
    )
    neurons_parser.add_argument(
        "--labels",
        help="Labels for prompts (pipe-separated, same order as prompts). Required for --auto-discover",
    )
    neurons_parser.add_argument(
        "--steer",
        help="Apply steering during analysis. Either 'direction.npz:coefficient' or just 'direction.npz' (use --strength for coefficient)",
    )
    neurons_parser.add_argument(
        "--strength",
        type=float,
        help="Steering strength/coefficient when using --steer (default: 1.0)",
    )
    neurons_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    neurons_parser.set_defaults(func=introspect_neurons)

    # Cluster command - visualize activation clusters
    cluster_parser = introspect_subparsers.add_parser(
        "cluster",
        help="Visualize activation clusters using PCA",
        description="""Project hidden states to 2D to see if different prompt types
cluster separately in activation space.

Shows ASCII scatter plot and cluster statistics.

Supports two syntaxes:
1. Legacy two-class: --class-a "prompts" --class-b "prompts" --label-a X --label-b Y
2. Multi-class: --prompts "p1|p2|p3" --label L1 --prompts "p4|p5" --label L2 ...

Multi-class example:
    lazarus introspect cluster -m model \\
        --prompts "45*45=|25*25=|15*15=" --label mult \\
        --prompts "123+456=|100+37=|50+50=" --label add \\
        --prompts "The capital of France is|The opposite of hot is" --label language \\
        --layer 19 --save-plot L19_cluster.png
        """,
    )
    cluster_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    # Legacy two-class syntax
    cluster_parser.add_argument(
        "--class-a",
        help="Class A prompts (pipe-separated or @file.txt) [legacy syntax]",
    )
    cluster_parser.add_argument(
        "--class-b",
        help="Class B prompts (pipe-separated or @file.txt) [legacy syntax]",
    )
    cluster_parser.add_argument(
        "--label-a",
        default="class_a",
        help="Label for class A (default: 'class_a') [legacy syntax]",
    )
    cluster_parser.add_argument(
        "--label-b",
        default="class_b",
        help="Label for class B (default: 'class_b') [legacy syntax]",
    )
    # New multi-class syntax
    cluster_parser.add_argument(
        "--prompts",
        action="append",
        dest="prompt_groups",
        help="Prompts for a class (pipe-separated or @file.txt). Use multiple times with --label.",
    )
    cluster_parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Label for the preceding --prompts group. Must match number of --prompts.",
    )
    cluster_parser.add_argument(
        "--layer",
        "-l",
        help="Layer(s) to analyze - single int or comma-separated (e.g., '19' or '19,20,21'). Default: ~50%% of model depth",
    )
    cluster_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    cluster_parser.add_argument(
        "--save-plot",
        help="Save matplotlib scatter plot to file (e.g., cluster.png)",
    )
    cluster_parser.set_defaults(func=lambda args: asyncio.run(introspect_activation_cluster(args)))
