"""Generate annualized return comparison charts for tree variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


COMPARISON_PAIRS: List[Tuple[str, str]] = [
    ("QRT", "QRT_leaf"),
    ("QRT", "QRT_node"),
]


def plot_ann_return_comparison(metrics_csv: Path) -> None:
    df = pd.read_csv(metrics_csv)
    if "annualized_return" not in df.columns or "model" not in df.columns:
        print(f"Skip {metrics_csv}: missing required columns")
        return

    ann_ret_by_model = df.groupby("model")["annualized_return"].mean()

    categories: List[str] = []
    base_values: List[float] = []
    leaf_values: List[float] = []

    for base, leaf in COMPARISON_PAIRS:
        if base in ann_ret_by_model and leaf in ann_ret_by_model:
            categories.append(base)
            base_values.append(float(ann_ret_by_model[base]))
            leaf_values.append(float(ann_ret_by_model[leaf]))

    if not categories:
        print(f"Skip {metrics_csv}: no matching model pairs found")
        return

    x = range(len(categories))
    width = 0.35
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar([xi - width / 2 for xi in x], base_values, width, label="Original")
    ax.bar([xi + width / 2 for xi in x], leaf_values, width, label="Leaf")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Annualized Return")
    ax.set_title("Original vs Leaf Annualized Return")

    max_val = max(base_values + leaf_values)
    min_val = min(base_values + leaf_values)
    ax.set_ylim(min_val * 0.95, max_val * 1.05 if max_val != 0 else 1.0)

    for xi, val in zip([xi - width / 2 for xi in x], base_values):
        ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, val in zip([xi + width / 2 for xi in x], leaf_values):
        ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()

    output_path = metrics_csv.with_name("05_ann_return_original_vs_leaf.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generated {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annualized return comparison chart for existing metrics.csv files."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Path to a metrics.csv file. If not provided, scan the default output directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/benchmark_split/tree_variants"),
        help="Root directory to scan for tree variant metrics.",
    )
    args = parser.parse_args()

    if args.metrics:
        plot_ann_return_comparison(args.metrics)
    else:
        for metrics_csv in args.root.rglob("metrics.csv"):
            plot_ann_return_comparison(metrics_csv)


if __name__ == "__main__":
    main()
