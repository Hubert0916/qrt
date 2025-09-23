"""Generate aggregated performance tables from benchmark metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def summarize_metrics(
    df: pd.DataFrame,
    coverage_target: float,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Aggregate per-window metrics according to the requested grouping."""
    group_cols = list(group_cols)
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            pinball=("pinball_sum", "mean"),
            coverage=("coverage", "mean"),
            cum_return=("cum_return", "mean"),
            n_windows=("window", "nunique"),
        )
    )
    grouped["calib_gap"] = (coverage_target - grouped["coverage"]).abs()

    ordered_cols = group_cols + [
        "pinball",
        "coverage",
        "calib_gap",
        "cum_return",
        "n_windows",
    ]
    return grouped[ordered_cols]


def _format_numeric(values: Iterable[float], digits: int = 4) -> list[str]:
    fmt = f"{{:.{digits}f}}"
    return [fmt.format(v) for v in values]


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame (already string-formatted) as GitHub-style Markdown."""
    headers = list(df.columns)
    alignments = []
    for col in headers:
        alignments.append(":---" if col in {"Model", "Split"} else "---:")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(alignments) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown_table(
    summary: pd.DataFrame,
    coverage_target: float,
    group_cols: Iterable[str],
) -> str:
    """Return a Markdown table with basic highlighting for best metrics."""
    df = summary.copy()
    group_cols = list(group_cols)
    numeric_cols = ["pinball", "coverage", "calib_gap", "cum_return"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["n_windows"] = df["n_windows"].astype(int)

    # Round values for display (allow column-wise precision)
    display_digits = {
        "pinball": 4,
        "coverage": 4,
        "calib_gap": 4,
        "cum_return": 2,
    }
    for col in numeric_cols:
        digits = display_digits.get(col, 4)
        df[col] = _format_numeric(df[col], digits=digits)

    # Identify extrema for highlighting
    best_pinball = df["pinball"].astype(float).min()
    best_gap = df["calib_gap"].astype(float).min()
    best_cumret = df["cum_return"].astype(float).max()

    def highlight(series: pd.Series, target: float) -> pd.Series:
        return series.apply(
            lambda v: f"**{v}**" if np.isclose(float(v), target) else str(v)
        )

    df["pinball"] = highlight(df["pinball"], best_pinball)
    df["calib_gap"] = highlight(df["calib_gap"], best_gap)
    df["cum_return"] = highlight(df["cum_return"], best_cumret)

    df.sort_values(group_cols, inplace=True)

    rename_map = {
        "model": "Model",
        "criterion": "Split",
        "pinball": "Pinball (\u2193)",
        "coverage": f"Coverage (\u2192{coverage_target:.2f})",
        "calib_gap": "Calib. Gap (\u2193)",
        "cum_return": "Mean CumRet (\u2191)",
        "n_windows": "nWindows",
    }
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=existing_map, inplace=True)
    return _dataframe_to_markdown(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark metrics into a Markdown table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/benchmark_split/metrics.csv"),
        help="Path to metrics CSV produced by main benchmark run.",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.4,
        help="Target coverage level used to compute calibration gap.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Optional path to save the rendered Markdown table.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the Markdown file instead of overwriting it.",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Optional section title inserted before the table.",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Optional descriptive text inserted before the table.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="model,criterion",
        help="Comma-separated columns to group by (e.g. 'model,criterion' or 'model').",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    group_cols = [col.strip() for col in args.group_by.split(",") if col.strip()]
    if not group_cols:
        raise ValueError("--group-by must include at least one column")

    summary = summarize_metrics(df, args.coverage_target, group_cols)

    markdown_table = render_markdown_table(summary, args.coverage_target, group_cols)

    sections: list[str] = []
    if args.title:
        sections.append(f"### {args.title}")
    if args.description:
        sections.append(args.description)
    sections.append(markdown_table)
    markdown = "\n\n".join(sections)

    print(markdown)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        has_content = args.output_md.exists() and args.output_md.stat().st_size > 0
        with args.output_md.open(mode, encoding="utf-8") as handle:
            if args.append and has_content:
                handle.write("\n\n")
            handle.write(markdown)
            handle.write("\n")


if __name__ == "__main__":
    main()
