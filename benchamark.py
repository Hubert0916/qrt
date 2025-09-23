# benchmark.py
"""
Benchmark: Original QRT vs Improved QRT
Emphasis: execution time and pinball loss across varying training sizes.

- Loads a single rolling window (first window by default) to keep the test set fixed.
- Subsamples the training set with multiple fractions (e.g., 0.1, 0.25, 0.5, 1.0).
- For each fraction and model:
    * Fit two single-quantile trees (ql, qh)
    * Predict on the full test set
    * Record (fit+predict) time and pinball loss sum (ql+qh)
- Outputs:
    * CSV results
    * ONE key plot: dual-axis line chart (Pinball Loss vs. Time by train size)
"""

from __future__ import annotations

import os
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.origin_quantile_regression_tree import QuantileRegressionTree as OldQRT
from models.quantile_regression_tree import QuantileRegressionTree as NewQRT
from utils.data_loader import load_data, rolling_time


# ---------------------- Metrics ---------------------- #

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Average pinball (quantile) loss for quantile q."""
    r = y_true - y_pred
    return float(np.mean(np.maximum(q * r, (q - 1.0) * r)))


def fit_predict_two_quantiles(
    model_cls,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    ql: float,
    qh: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit two single-quantile trees (ql and qh) and return predictions on X_test.
    Timing of (fit_l + fit_h + predict_l + predict_h) is measured outside.
    """
    model_l = model_cls(**kwargs)
    model_l.fit(X_train, y_train, ql)
    pred_l = model_l.predict(X_test)

    model_h = model_cls(**kwargs)
    model_h.fit(X_train, y_train, qh)
    pred_h = model_h.predict(X_test)

    return pred_l, pred_h


# ---------------------- Benchmark Core ---------------------- #

def run_benchmark(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Fix a single rolling window (first window) to hold test set constant
    n_windows = rolling_time(args.data, args.train_period, args.test_period)
    if n_windows < 1:
        raise RuntimeError(
            "No valid rolling window from the provided data/time setup.")

    print(f"Using window 1/{n_windows} for size sweep.")
    train_df, test_df = load_data(
        args.data, args.train_period, args.test_period, 3)

    # Features/labels
    x_cols = [c for c in train_df.columns if c.endswith("_TFIDF")]
    X_full, y_full = train_df[x_cols], train_df["報酬率"]
    X_test, y_test = test_df[x_cols], test_df["報酬率"]

    # 2) Define size fractions and repeatable sampling
    rng = np.random.default_rng(args.random_state)
    fractions = [float(x) for x in args.fractions.split(",")]
    fractions = [f for f in fractions if 0.0 < f <= 1.0]
    if not fractions:
        fractions = [0.1, 0.25, 0.5, 1.0]

    # 3) Sweep sizes
    rows: List[Dict] = []
    for frac in fractions:
        n_train = int(np.ceil(len(X_full) * frac))
        idx = rng.choice(len(X_full), size=n_train, replace=False)
        X_train = X_full.iloc[idx]
        y_train = y_full.iloc[idx]

        print(f"\n=== Train fraction {frac:.2f} → {n_train} samples ===")

        for name, cls in (("OldQRT", OldQRT), ("NewQRT", NewQRT)):
            # Measure (fit two trees + predict twice)
            t0 = time.perf_counter()
            pred_l, pred_h = fit_predict_two_quantiles(
                cls,
                X_train,
                y_train,
                X_test,
                args.ql,
                args.qh,
                split_criterion=args.split_criterion,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state,
                feature_names=x_cols,
            )
            t1 = time.perf_counter()

            pl_ql = pinball_loss(y_test.values, pred_l, args.ql)
            pl_qh = pinball_loss(y_test.values, pred_h, args.qh)
            pl_sum = pl_ql + pl_qh
            elapsed = t1 - t0

            rows.append(
                dict(
                    model=name,
                    fraction=frac,
                    train_size=n_train,
                    time_sec=elapsed,
                    pinball_ql=pl_ql,
                    pinball_qh=pl_qh,
                    pinball_sum=pl_sum,
                )
            )
            print(f"[{name}] time={elapsed:.3f}s  pinball_sum={pl_sum:.5f}")

    # 4) Save CSV
    df = pd.DataFrame(rows)
    df.sort_values(["train_size", "model"], inplace=True)
    csv_path = os.path.join(args.outdir, "qrt_size_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # 5) ONE key plot: Pinball Loss (left y) + Time (right y) vs. Train Size
    plt.figure(figsize=(10, 6))
    # Split by model
    df_old = df[df["model"] == "OldQRT"]
    df_new = df[df["model"] == "NewQRT"]

    # Left axis: pinball loss (lower is better)
    ax1 = plt.gca()
    l1, = ax1.plot(
        df_old["train_size"], df_old["pinball_sum"],
        marker="o", linewidth=2, label="OldQRT — Pinball"
    )
    l2, = ax1.plot(
        df_new["train_size"], df_new["pinball_sum"],
        marker="o", linewidth=2, label="NewQRT — Pinball"
    )
    ax1.set_xlabel("Training Size (samples)")
    ax1.set_ylabel("Pinball Loss (ql + qh)")
    ax1.grid(True, linewidth=0.3, alpha=0.6)

    # Right axis: time (lower is better)
    ax2 = ax1.twinx()
    l3, = ax2.plot(
        df_old["train_size"], df_old["time_sec"],
        marker="s", linestyle="--", linewidth=2, label="OldQRT — Time (s)", color="#c44e52"
    )
    l4, = ax2.plot(
        df_new["train_size"], df_new["time_sec"],
        marker="s", linestyle="--", linewidth=2, label="NewQRT — Time (s)", color="#55a868"
    )
    ax2.set_ylabel("Wall Time (s)")

    # Legend combining both axes
    lines = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    plt.legend(lines, labels, loc="best")

    plt.title(
        f"Old vs Improved QRT — Time & Pinball vs Training Size\n"
        f"(criterion={args.split_criterion}, ql={args.ql:.2f}, qh={args.qh:.2f})"
    )
    plt.tight_layout()
    fig_path = os.path.join(args.outdir, "qrt_size_sweep.png")
    plt.savefig(fig_path, dpi=220)
    plt.close()
    print(f"Saved plot → {fig_path}")


# ---------------------- CLI -------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Old vs New QRT focusing on time & accuracy over train sizes."
    )
    parser.add_argument(
        "--data", type=str,
        default="data/esg_tfidf_with_return_cleaned.csv",
        help="CSV path for dataset.",
    )
    parser.add_argument("--outdir", type=str,
                        default="output/benchmark_qrt_size")
    parser.add_argument("--train_period", type=int, default=5)
    parser.add_argument("--test_period", type=int, default=1)
    parser.add_argument("--ql", type=float, default=0.3)
    parser.add_argument("--qh", type=float, default=0.7)
    parser.add_argument(
        "--split_criterion", type=str, default="mse",
        choices=["loss", "mse", "r2"],
        help="Split criterion used for both models.",
    )
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--fractions", type=str, default="0.1,0.25,0.5,1.0",
        help="Comma-separated training set fractions to evaluate.",
    )
    return parser


run_benchmark(build_argparser().parse_args())
