"""Introspect command parsers."""

from ._ablation import register_ablation_parsers
from ._analyze import register_analyze_parsers
from ._arithmetic import register_arithmetic_parsers
from ._circuit import register_circuit_parsers
from ._classifier import register_classifier_parsers
from ._directions import register_directions_parsers
from ._embedding import register_embedding_parsers
from ._generation import register_generation_parsers
from ._layer import register_layer_parsers
from ._memory import register_memory_parsers
from ._moe import register_moe_parsers
from ._patch import register_patch_parsers
from ._probing import register_probing_parsers
from ._steering import register_steering_parsers


def register_introspect_parsers(subparsers):
    """Register the introspect subcommand and all sub-subcommands."""
    introspect_parser = subparsers.add_parser(
        "introspect",
        help="Model introspection and logit lens analysis",
        description="Analyze model behavior using logit lens and attention visualization.",
    )
    introspect_subparsers = introspect_parser.add_subparsers(
        dest="introspect_command", help="Introspection commands"
    )

    register_analyze_parsers(introspect_subparsers)
    register_ablation_parsers(introspect_subparsers)
    register_layer_parsers(introspect_subparsers)
    register_generation_parsers(introspect_subparsers)
    register_steering_parsers(introspect_subparsers)
    register_arithmetic_parsers(introspect_subparsers)
    register_probing_parsers(introspect_subparsers)
    register_memory_parsers(introspect_subparsers)
    register_directions_parsers(introspect_subparsers)
    register_embedding_parsers(introspect_subparsers)
    register_patch_parsers(introspect_subparsers)
    register_circuit_parsers(introspect_subparsers)
    register_moe_parsers(introspect_subparsers)
    register_classifier_parsers(introspect_subparsers)
