"""Run full benchmark covering tree and forest variants."""

from __future__ import annotations

from experiments import benchmark

MODEL_GROUP_ALL = list(benchmark.TREE_VARIANTS.keys()) \
    + list(benchmark.FOREST_VARIANTS.keys())

def main() -> None:
    parser = benchmark.build_arg_parser()
    args = parser.parse_args()
    benchmark.run_quantile_sweep(args, MODEL_GROUP_ALL)


if __name__ == "__main__":
    main()
