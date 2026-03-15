"""Parser registration functions for CLI subcommands."""

from ._bench import register_bench_parser
from ._context import register_context_parsers
from ._data import register_data_parsers
from ._experiment import register_experiment_parsers
from ._gym import register_gym_parsers
from ._infer import register_infer_parser
from ._introspect import register_introspect_parsers
from ._tokenizer import register_tokenizer_parsers
from ._train import register_train_parsers

__all__ = [
    "register_train_parsers",
    "register_infer_parser",
    "register_context_parsers",
    "register_tokenizer_parsers",
    "register_data_parsers",
    "register_gym_parsers",
    "register_experiment_parsers",
    "register_bench_parser",
    "register_introspect_parsers",
]
