"""Benchmark baseline, leaf, and node quantile regression trees."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import benchmark

MODEL_GROUP = list(benchmark.TREE_VARIANTS.keys())


def main() -> None:
    parser = benchmark.build_arg_parser()
    args = parser.parse_args()
    args.quantile_pairs = "0.3:0.7"
    args.outdir = benchmark.ensure_subdir(args.outdir, "tree_variants")
    benchmark.run_quantile_sweep(args, MODEL_GROUP)


if __name__ == "__main__":
    main()
