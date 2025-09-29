"""Benchmark only the single-tree QRT variant."""

from __future__ import annotations

from experiments import benchmark_core

MODEL_GROUP = ["QRT"]


def main() -> None:
    parser = benchmark_core.build_arg_parser()
    args = parser.parse_args()
    benchmark_core.run_quantile_sweep(args, MODEL_GROUP)


if __name__ == "__main__":
    main()
