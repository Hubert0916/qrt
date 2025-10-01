"""Run full benchmark covering tree and forest variants."""

from __future__ import annotations

from experiments import benchmark_core

MODEL_GROUP_ALL = list(benchmark_core.TREE_VARIANTS.keys()) \
    + list(benchmark_core.FOREST_VARIANTS.keys())

def main() -> None:
    parser = benchmark_core.build_arg_parser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_GROUP_ALL,
        default=MODEL_GROUP_ALL,
        help="Specify one or more models to run. If not provided, all models are run.",
    )
    args = parser.parse_args()
    benchmark_core.run_quantile_sweep(args, args.models)


if __name__ == "__main__":
    main()
