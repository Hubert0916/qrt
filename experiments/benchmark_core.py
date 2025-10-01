"""Shared benchmarking utilities for QRT/QRF evaluations."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from models.quantile_regression_forest import QuantileRegressionForest
from models.quantile_regression_tree import QuantileRegressionTree
from utils.data_loader import load_data, rolling_time
from utils.trading_strategy import trading_rule

from models.qunatile_regression_model_leaf_tree import LeafQuantileRegressionTree
from models.qunatile_regression_model_node_tree import NodeQuantileRegressionTree

# ------------------------------ CLI -------------------------------- #


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Benchmark QRT variants under different split criteria and quantile bands.",
    )
    parser.add_argument("--train_period", type=int, default=3)
    parser.add_argument("--test_period", type=int, default=1)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--qh", type=float, default=0.7)
    parser.add_argument("--ql", type=float, default=0.3)
    parser.add_argument(
        "--data",
        type=str,
        default="data/esg_tfidf_with_return_cleaned.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output/benchmark_split",
    )
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--annualization-base",
        type=float,
        default=252.0,
        help="Number of periods per year for annualizing returns (e.g., 252 for daily data).",
    )
    parser.add_argument(
        "--quantile-pairs",
        type=str,
        default="0.3:0.7",
        help="Comma separated list of ql:qh pairs to evaluate.",
    )
    return parser


# --------------------------- Metrics --------------------------- #


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Average pinball loss for quantile q."""
    r = y_true - y_pred
    return np.mean(np.maximum(q * r, (q - 1) * r))


def coverage_rate(y_true: np.ndarray, ql_pred: np.ndarray, qh_pred: np.ndarray) -> float:
    """Fraction of targets lying within [ql, qh] interval."""
    return float(np.mean((y_true >= ql_pred) & (y_true <= qh_pred)))


SplitCriterion = ["loss", "mse", "r2"]

TREE_VARIANTS: Dict[str, type] = {
    "QRT": QuantileRegressionTree,
    "QRT_leaf": LeafQuantileRegressionTree,
    "QRT_node": NodeQuantileRegressionTree,
}

FOREST_VARIANTS: Dict[str, type] = {
    "QRF": QuantileRegressionTree,
    "QRF_leaf": LeafQuantileRegressionTree,
}


def quantile_dir_suffix(ql: float, qh: float) -> str:
    return f"ql_{round(ql*100):02d}_qh_{round(qh*100):02d}"


def parse_quantile_pairs(spec: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for raw in spec.split(','):
        raw = raw.strip()
        if not raw:
            continue
        try:
            ql_str, qh_str = raw.split(':')
            ql, qh = float(ql_str), float(qh_str)
        except ValueError as exc:
            raise ValueError(f"Invalid quantile pair '{raw}'. Use ql:qh format.") from exc
        if not (0.0 < ql < qh < 1.0):
            raise ValueError(f"Quantile pair '{raw}' must satisfy 0 < ql < qh < 1")
        pairs.append((ql, qh))
    if not pairs:
        raise ValueError("No valid quantile pairs supplied.")
    return pairs


def fit_predict_qrt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    criterion: str,
    ql: float,
    qh: float,
    max_depth: int,
    min_samples_leaf: int,
    feature_names: List[str],
    seed: int,
    tree_cls,
) -> Tuple[np.ndarray, np.ndarray]:
    model_l = tree_cls(
        split_criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        feature_names=feature_names,
        random_state=seed,
        max_threshold_candidates=128,
    )
    model_l.fit(X_train, y_train, ql)

    model_h = tree_cls(
        split_criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        feature_names=feature_names,
        random_state=seed,
        max_threshold_candidates=128,
    )
    model_h.fit(X_train, y_train, qh)

    y_pred_l = model_l.predict(X_test)
    y_pred_h = model_h.predict(X_test)
    return y_pred_l, y_pred_h


def fit_predict_qrf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    criterion: str,
    ql: float,
    qh: float,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
    tree_cls,
) -> Tuple[np.ndarray, np.ndarray]:
    model_h = QuantileRegressionForest(
        n_estimators=n_estimators,
        quantile=qh,
        split_criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        max_features="sqrt",
        max_threshold_candidates=128,
        random_thresholds=False,
        include_oob=True,
        min_leaf_agg=8,
        random_state=seed,
        tree_cls=tree_cls,
    )
    model_h.fit(X_train, y_train)

    model_l = QuantileRegressionForest(
        n_estimators=n_estimators,
        quantile=ql,
        split_criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        max_features="sqrt",
        max_threshold_candidates=128,
        random_thresholds=False,
        include_oob=True,
        min_leaf_agg=8,
        random_state=seed,
        tree_cls=tree_cls,
    )
    model_l.fit(X_train, y_train)

    y_pred_h = model_h.predict(X_test)
    y_pred_l = model_l.predict(X_test)
    return y_pred_l, y_pred_h


def run_benchmark(
    args: Namespace,
    model_kinds: Iterable[str],
    ql: float,
    qh: float,
    outdir: str,
) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)

    print("================================")
    print("Benchmark: model variants under split criteria [loss, mse, r2]")
    print(f"Train Period: {args.train_period}, Test Period: {args.test_period}")
    print(f"Quantiles: ql={round(ql*100):02d}, qh={round(qh*100):02d}")
    print("================================")

    n_windows = rolling_time(args.data, args.train_period, args.test_period)

    records: List[Dict] = []

    for win in range(n_windows):
        print(f"\n=== Rolling window {win + 1}/{n_windows} ===")
        train_df, test_df = load_data(args.data, args.train_period, args.test_period, win)
        x_cols = [c for c in train_df.columns if c.endswith("_TFIDF")]
        X_train, y_train = train_df[x_cols], train_df["報酬率"]
        X_test, y_test = test_df[x_cols], test_df["報酬率"]

        for model_kind in model_kinds:
            for crit in SplitCriterion:
                if model_kind in TREE_VARIANTS:
                    tree_cls = TREE_VARIANTS[model_kind]
                    y_pred_l, y_pred_h = fit_predict_qrt(
                        X_train,
                        y_train,
                        X_test,
                        crit,
                        ql,
                        qh,
                        args.max_depth,
                        args.min_samples_leaf,
                        x_cols,
                        args.random_state,
                        tree_cls,
                    )
                elif model_kind in FOREST_VARIANTS:
                    tree_cls = FOREST_VARIANTS[model_kind]
                    y_pred_l, y_pred_h = fit_predict_qrf(
                        X_train,
                        y_train,
                        X_test,
                        crit,
                        ql,
                        qh,
                        args.n_estimators,
                        args.max_depth,
                        args.min_samples_leaf,
                        args.random_state,
                        tree_cls,
                    )
                else:
                    raise ValueError(f"Unknown model kind: {model_kind}")

                pl_ql = pinball_loss(y_test.values, y_pred_l, ql)
                pl_qh = pinball_loss(y_test.values, y_pred_h, qh)
                cov = coverage_rate(y_test.values, y_pred_l, y_pred_h)

                df_pred = test_df.copy()
                df_pred[f"pred_q{ql}"] = y_pred_l
                df_pred[f"pred_q{qh}"] = y_pred_h
                df_traded = trading_rule(df_pred, qh, ql)
                n_periods = len(df_traded)
                cum_ret_final = (
                    float(df_traded["total_return"].iloc[-1]) if n_periods > 0 else 0.0
                )
                avg_return = float(df_traded["return"].mean()) if n_periods > 0 else 0.0

                annualized_return = 0.0
                base = float(args.annualization_base)
                if n_periods > 0 and cum_ret_final > -1.0 and base > 0:
                    annualized_return = (1.0 + cum_ret_final) ** (base / n_periods) - 1.0

                records.append(
                    dict(
                        window=win + 1,
                        ql=ql,
                        qh=qh,
                        model=model_kind,
                        criterion=crit,
                        pinball_ql=pl_ql,
                        pinball_qh=pl_qh,
                        pinball_sum=pl_ql + pl_qh,
                        coverage=cov,
                        cum_return=cum_ret_final,
                        avg_return=avg_return,
                        annualized_return=annualized_return,
                    )
                )

                print(
                    f"[{model_kind}/{crit}]  Pinball(ql_{round(ql*100):02d})={pl_ql:.4f}  "
                    f"Pinball(qh_{round(qh*100):02d})={pl_qh:.4f}  "
                    f"Coverage={cov:.3f}  CumRet={cum_ret_final:.4f}  "
                    f"AvgRet={avg_return:.4f}  AnnRet={annualized_return:.4f}"
                )

    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    _write_plots(metrics_df, ql, qh, outdir)
    return metrics_df


def _write_plots(metrics_df: pd.DataFrame, ql: float, qh: float, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    def add_bar_labels(ax, values, fmt="{:.4f}"):
        for rect, val in zip(ax.patches, values):
            height = rect.get_height()
            ax.annotate(
                fmt.format(val),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    agg = (
        metrics_df.groupby(["model", "criterion"])
        .agg(
            pinball_ql=("pinball_ql", "mean"),
            pinball_qh=("pinball_qh", "mean"),
            pinball_sum=("pinball_sum", "mean"),
            coverage=("coverage", "mean"),
            cum_return=("cum_return", "mean"),
            avg_return=("avg_return", "mean"),
            annualized_return=("annualized_return", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values("pinball_sum", ascending=True)

    labels = [f"{m}/{c}" for m, c in zip(agg["model"], agg["criterion"])]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(x, agg["pinball_sum"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Pinball Loss (ql+qh)")
    ax.set_title(
        f"Quantile Regression Accuracy (Pinball Loss) ql={round(ql*100):02d} qh={round(qh*100):02d}"
    )
    ax.set_ylim(agg["pinball_sum"].min() * 0.95, agg["pinball_sum"].max() * 1.05)
    add_bar_labels(ax, agg["pinball_sum"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_pinball_sum_bar.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    for (m, c), group in metrics_df.groupby(["model", "criterion"]):
        plt.plot(group["window"], group["pinball_sum"], marker="o", label=f"{m}/{c}")
    plt.xticks(range(1, metrics_df["window"].nunique() + 1))
    plt.xlabel("Rolling Window")
    plt.ylabel("Pinball Loss (ql + qh)")
    plt.title("Pinball Loss per Rolling Window")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01-1_pinball_sum_per_window.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(x, agg["coverage"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Coverage Rate")
    ax.set_title(
        f"Interval Coverage between q{round(ql*100):02d} and q{round(qh*100):02d}"
    )
    ax.set_ylim(agg["coverage"].min() * 0.95, min(1.0, agg["coverage"].max() * 1.05))
    add_bar_labels(ax, agg["coverage"], fmt="{:.3f}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_coverage_bar.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(x, agg["cum_return"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Final Cumulative Return")
    ax.set_title("Trading Strategy Performance (Cumulative Return)")
    ax.set_ylim(agg["cum_return"].min() * 0.95, agg["cum_return"].max() * 1.05)
    add_bar_labels(ax, agg["cum_return"], fmt="{:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_cum_return_bar.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(x, agg["avg_return"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Average Return")
    ax.set_title("Average Return per Trade")
    ax.set_ylim(agg["avg_return"].min() * 0.95, agg["avg_return"].max() * 1.05)
    add_bar_labels(ax, agg["avg_return"], fmt="{:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03-1_avg_return_bar.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(x, agg["annualized_return"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Annualized Return")
    ax.set_title("Annualized Trading Performance")
    ax.set_ylim(
        agg["annualized_return"].min() * 0.95,
        agg["annualized_return"].max() * 1.05,
    )
    add_bar_labels(ax, agg["annualized_return"], fmt="{:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_annualized_return_bar.png"), dpi=200)
    plt.close()

    # Compare baseline vs leaf annualized returns
    ann_ret_by_model = metrics_df.groupby("model")["annualized_return"].mean()
    comparison_pairs = [
        ("QRT", "QRT_leaf"),
        ("QRF", "QRF_leaf"),
    ]
    categories: List[str] = []
    base_values: List[float] = []
    leaf_values: List[float] = []
    for base, leaf in comparison_pairs:
        if base in ann_ret_by_model and leaf in ann_ret_by_model:
            categories.append(base)
            base_values.append(float(ann_ret_by_model[base]))
            leaf_values.append(float(ann_ret_by_model[leaf]))

    if categories:
        x = np.arange(len(categories))
        width = 0.35
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.bar(x - width / 2, base_values, width, label="Original")
        ax.bar(x + width / 2, leaf_values, width, label="Leaf")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Mean Annualized Return")
        ax.set_title("Original vs Leaf Annualized Return")
        max_val = max(base_values + leaf_values)
        min_val = min(base_values + leaf_values)
        ax.set_ylim(min_val * 0.95, max_val * 1.05 if max_val != 0 else 1.0)

        for xpos, val in zip(x - width / 2, base_values):
            ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        for xpos, val in zip(x + width / 2, leaf_values):
            ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "05_ann_return_original_vs_leaf.png"), dpi=200)
        plt.close()


def ensure_subdir(base: str, sub: str) -> str:
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


def run_quantile_sweep(args: Namespace, model_kinds: Iterable[str]) -> None:
    quantile_pairs = parse_quantile_pairs(args.quantile_pairs)
    for ql, qh in quantile_pairs:
        outdir_suffix = quantile_dir_suffix(ql, qh)
        target_outdir = os.path.join(args.outdir, outdir_suffix)
        run_benchmark(args, model_kinds, ql, qh, target_outdir)
