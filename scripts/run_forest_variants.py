"""Benchmark only forest-based quantile models (standard/leaf/node)."""

from __future__ import annotations

from experiments import benchmark_core

MODEL_GROUP = list(benchmark_core.FOREST_VARIANTS.keys())


def main() -> None:
    parser = benchmark_core.build_arg_parser()
    args = parser.parse_args()
    args.quantile_pairs = "0.3:0.6,0.3:0.7,0.3:0.8"
    args.outdir = benchmark_core.ensure_subdir(args.outdir, "forest_variants")
    benchmark_core.run_quantile_sweep(args, MODEL_GROUP)


if __name__ == "__main__":
    main()
