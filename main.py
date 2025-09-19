# main.py
import os
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.quantile_regression_forest import QuantileRegressionForest
from models.quantile_regression_tree import QuantileRegressionTree
from utils.data_loader import rolling_time, load_data

# ------------------------------ CLI -------------------------------- #

parser = argparse.ArgumentParser(
    description="Compare QRT vs QRF under different split criteria, "
                "evaluate quantile metrics and trading returns."
)
parser.add_argument("--train_period", type=int, default=5)
parser.add_argument("--test_period", type=int, default=1)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--min_samples_leaf", type=int, default=10)
parser.add_argument("--qh", type=float, default=0.9)
parser.add_argument("--ql", type=float, default=0.1)
parser.add_argument("--data", type=str, default="data/esg_tfidf_with_return_cleaned.csv")
parser.add_argument("--outdir", type=str, default="output/benchmark")
parser.add_argument("--n_estimators", type=int, default=50)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

TRAIN_PERIOD = args.train_period
TEST_PERIOD = args.test_period

# --------------------------- Metrics & Trading --------------------------- #

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Average pinball loss for quantile q."""
    r = y_true - y_pred
    return np.mean(np.maximum(q * r, (q - 1) * r))

def coverage_rate(y_true: np.ndarray, ql_pred: np.ndarray, qh_pred: np.ndarray) -> float:
    """Fraction of targets lying within [ql, qh] interval."""
    return float(np.mean((y_true >= ql_pred) & (y_true <= qh_pred)))

# TO DO: Implement trading strategy based on quantile predictions
def trading_rule(test_df: pd.DataFrame, qh: float, ql: float) -> pd.DataFrame:
    """
    Very simple long/short rule based on predicted quantiles (user-provided logic).
    Adjust freely to your use case.
    """
    df = test_df.copy()
    df["return"] = 0.0

    up_col = f"pred_q{qh}"
    low_col = f"pred_q{ql}"

    for idx, row in df.iterrows():
        up = row[up_col]
        low = row[low_col]
        # Example guards to avoid division by zero.
        if row.get("BIDLO_-4d", 0) != 0 and row.get("ASKHI_-4d", 0) != 0:
            up_gate = (row["OPENPRC_1d"] - row["BIDLO_-4d"]) / row["BIDLO_-4d"]
            low_gate = (row["OPENPRC_1d"] - row["ASKHI_-4d"]) / row["ASKHI_-4d"]
        else:
            up_gate, low_gate = 0.0, 0.0

        if up > 0 and up > up_gate:
            df.at[idx, "return"] = up
        elif low < 0 and low < low_gate:
            df.at[idx, "return"] = -low

    df["total_return"] = df["return"].cumsum()
    return df

# ----------------------------- Training ------------------------------- #

SplitCriterion = ["loss", "mse", "r2"]
ModelKinds = ["QRT", "QRF"]

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Train two single-quantile QRTs (ql, qh) and return predictions."""
    model_l = QuantileRegressionTree(
        split_criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        feature_names=feature_names,
        random_state=seed,
        random_features=False,           # deterministic per our comparison
        random_thresholds=False,
        max_threshold_candidates=128,
    )
    model_l.fit(X_train, y_train, ql)

    model_h = QuantileRegressionTree(
        split_criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        feature_names=feature_names,
        random_state=seed,
        random_features=False,
        random_thresholds=False,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a QRF and predict both quantiles (ql, qh)."""
    model = QuantileRegressionForest(
        n_estimators=n_estimators,
        quantile=qh,  # default; we override when predicting ql
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
    )
    model.fit(X_train, y_train)
    y_pred_h = model.predict(X_test, quantile=qh)
    y_pred_l = model.predict(X_test, quantile=ql)
    return y_pred_l, y_pred_h

# --------------------------- Orchestration ---------------------------- #

def run_benchmark() -> None:
    os.makedirs(args.outdir, exist_ok=True)

    print("================================")
    print("Benchmark: QRT vs QRF under split criteria [loss, mse, r2]")
    print(f"Train Period: {TRAIN_PERIOD}, Test Period: {TEST_PERIOD}")
    print("================================")

    n_windows = rolling_time(args.data, TRAIN_PERIOD, TEST_PERIOD)

    # Collect per-window & aggregated metrics
    records: List[Dict] = []
    equity_curves: Dict[Tuple[str, str], List[pd.DataFrame]] = {}  # (model, crit) -> list of test dfs

    for win in range(n_windows):
        print(f"\n=== Rolling window {win + 1}/{n_windows} ===")
        train_df, test_df = load_data(args.data, TRAIN_PERIOD, TEST_PERIOD, win)

        # Features/labels
        x_cols = [c for c in train_df.columns if c.endswith("_TFIDF")]
        X_train, y_train = train_df[x_cols], train_df["報酬率"]
        X_test, y_test = test_df[x_cols], test_df["報酬率"]

        for model_kind in ModelKinds:
            for crit in SplitCriterion:
                if model_kind == "QRT":
                    y_pred_l, y_pred_h = fit_predict_qrt(
                        X_train, y_train, X_test, crit, args.ql, args.qh,
                        args.max_depth, args.min_samples_leaf, x_cols, args.random_state
                    )
                else:  # QRF
                    y_pred_l, y_pred_h = fit_predict_qrf(
                        X_train, y_train, X_test, crit, args.ql, args.qh,
                        args.n_estimators, args.max_depth, args.min_samples_leaf, args.random_state
                    )

                # Metrics
                pl_ql = pinball_loss(y_test.values, y_pred_l, args.ql)
                pl_qh = pinball_loss(y_test.values, y_pred_h, args.qh)
                cov = coverage_rate(y_test.values, y_pred_l, y_pred_h)

                # Trading
                df_pred = test_df.copy()
                df_pred[f"pred_q{args.ql}"] = y_pred_l
                df_pred[f"pred_q{args.qh}"] = y_pred_h
                df_traded = trading_rule(df_pred, args.qh, args.ql)
                cum_ret_final = float(df_traded["total_return"].iloc[-1])

                # Save window record
                records.append(
                    dict(
                        window=win + 1,
                        model=model_kind,
                        criterion=crit,
                        pinball_ql=pl_ql,
                        pinball_qh=pl_qh,
                        pinball_sum=pl_ql + pl_qh,
                        coverage=cov,
                        cum_return=cum_ret_final,
                    )
                )

                equity_curves.setdefault((model_kind, crit), []).append(
                    df_traded[["日期", "total_return"]].assign(
                        model=model_kind, criterion=crit, window=win + 1
                    )
                )

                print(
                    f"[{model_kind}/{crit}]  Pinball(q{args.ql:.2f})={pl_ql:.4f}  "
                    f"Pinball(q{args.qh:.2f})={pl_qh:.4f}  "
                    f"Coverage={cov:.3f}  CumRet={cum_ret_final:.4f}"
                )

    # Aggregate to DataFrame
    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

    # ------------------------------- Plots ------------------------------- #

    # 1) Pinball loss bars (lower is better)
    agg = (
        metrics_df.groupby(["model", "criterion"])
        .agg(
            pinball_ql=("pinball_ql", "mean"),
            pinball_qh=("pinball_qh", "mean"),
            pinball_sum=("pinball_sum", "mean"),
            coverage=("coverage", "mean"),
            cum_return=("cum_return", "mean"),
        )
        .reset_index()
    )

    # Sort by combined pinball for nice ordering
    agg = agg.sort_values("pinball_sum", ascending=True)

    # Helper labels
    labels = [f"{m}/{c}" for m, c in zip(agg["model"], agg["criterion"])]
    x = np.arange(len(labels))

    # Plot: combined pinball loss
    plt.figure(figsize=(12, 6))
    plt.bar(x, agg["pinball_sum"])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean Pinball Loss (ql + qh)")
    plt.title("Quantile Accuracy (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "01_pinball_sum_bar.png"), dpi=200)
    plt.close()

    # Plot: coverage
    plt.figure(figsize=(12, 6))
    plt.bar(x, agg["coverage"])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean Coverage Rate")
    plt.title(f"Interval Coverage between q{args.ql:.2f} and q{args.qh:.2f} (higher is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "02_coverage_bar.png"), dpi=200)
    plt.close()

    # Plot: cumulative return
    plt.figure(figsize=(12, 6))
    plt.bar(x, agg["cum_return"])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean Final Cumulative Return")
    plt.title("Trading Strategy Outcome (higher is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "03_cum_return_bar.png"), dpi=200)
    plt.close()

    # 2) Equity curve for the best config (by lowest pinball_sum)
    best_row = agg.iloc[0]
    best_key = (best_row["model"], best_row["criterion"])
    best_curves = pd.concat(equity_curves[best_key], ignore_index=True)

    # Merge windows sequentially (sorted by date) for a single equity curve view.
    best_curves = best_curves.sort_values("日期").reset_index(drop=True)
    best_curves["merged_equity"] = best_curves["total_return"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(best_curves["日期"], best_curves["merged_equity"], linewidth=2)
    plt.title(f"Equity Curve — Best Config: {best_key[0]}/{best_key[1]}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (merged across windows)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "04_equity_best.png"), dpi=200)
    plt.close()

    # 3) Per-window equity curves (optional, looks cool)
    #    This overlays the equity curve of each window for the best config.
    plt.figure(figsize=(12, 6))
    for dfw in equity_curves[best_key]:
        plt.plot(dfw["日期"], dfw["total_return"], alpha=0.6)
    plt.title(f"Per-Window Equity Curves — {best_key[0]}/{best_key[1]}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (per window)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "05_equity_best_per_window.png"), dpi=200)
    plt.close()

    # Console summary
    print("\n================================")
    print("Aggregated performance (mean across windows):")
    print(agg.to_string(index=False))
    print("Saved metrics.csv and plots to:", args.outdir)


if __name__ == "__main__":
    run_benchmark()
