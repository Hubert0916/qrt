"""Run full benchmark covering tree and forest variants."""

from __future__ import annotations

from experiments import benchmark_core

MODEL_GROUP_ALL = list(benchmark_core.TREE_VARIANTS.keys()) \
    + list(benchmark_core.FOREST_VARIANTS.keys())

def main() -> None:
    parser = benchmark_core.build_arg_parser()
    args = parser.parse_args()
    benchmark_core.run_quantile_sweep(args, MODEL_GROUP_ALL)


if __name__ == "__main__":
    main()
