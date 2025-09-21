# main.py
import os
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--min_samples_leaf", type=int, default=5)
parser.add_argument("--qh", type=float, default=0.7)
parser.add_argument("--ql", type=float, default=0.3)
parser.add_argument("--data", type=str, default="data/esg_tfidf_with_return_cleaned.csv")
parser.add_argument("--outdir", type=str, default="output/benchmark")
parser.add_argument("--n_estimators", type=int, default=100)
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

def _valid_price(value) -> bool:
    try:
        return pd.notna(value) and float(value) > 0
    except (TypeError, ValueError):
        return False

# return the percentage
def _simulate_long(row: pd.Series, entry_price: float, previous_low: float, target_return: float) -> float:
    if target_return <= 0:
        return 0.0

    target_price = max(previous_low * (1.0 + target_return), 1e-6)
    stop_price = max(entry_price * 0.9, 1e-6)

    for day in (2, 3, 4):
        open_price = row.get(f"OPENPRC_{day}d")
        high_price = row.get(f"ASKHI_{day}d")
        low_price = row.get(f"BIDLO_{day}d")
        close_price = row.get(f"PRC_{day}d")

        if day > 1 and _valid_price(open_price):
            open_price = float(open_price)
            if open_price >= target_price:
                return open_price / entry_price - 1.0
            if open_price <= stop_price:
                return open_price / entry_price - 1.0

        if _valid_price(high_price) and float(high_price) >= target_price:
            return target_price / entry_price - 1.0
        if _valid_price(low_price) and float(low_price) <= stop_price:
            return stop_price / entry_price - 1.0

        if day == 4 and _valid_price(close_price):
            return float(close_price) / entry_price - 1.0

    return 0.0


def _simulate_short(row: pd.Series, entry_price: float, target_return: float) -> float:
    if target_return >= 0:
        return 0.0

    target_price = max(entry_price * (1.0 + target_return), 1e-6)
    stop_return = abs(target_return) / 10.0
    stop_price = max(entry_price * (1.0 + stop_return), 1e-6)

    for day in (1, 2, 3):
        open_price = row.get(f"OPENPRC_{day}d")
        high_price = row.get(f"ASKHI_{day}d")
        low_price = row.get(f"BIDLO_{day}d")
        close_price = row.get(f"PRC_{day}d")

        if day > 1 and _valid_price(open_price):
            open_price = float(open_price)
            if open_price <= target_price:
                return 1.0 - open_price / entry_price
            if open_price >= stop_price:
                return 1.0 - open_price / entry_price

        if _valid_price(low_price) and float(low_price) <= target_price:
            return 1.0 - target_price / entry_price
        if _valid_price(high_price) and float(high_price) >= stop_price:
            return 1.0 - stop_price / entry_price

        if day == 3 and _valid_price(close_price):
            return 1.0 - float(close_price) / entry_price

    return 0.0


# Trading strategy based on quantile predictions and 4-day management rules
def trading_rule(test_df: pd.DataFrame, qh: float, ql: float) -> pd.DataFrame:
    df = test_df.copy()
    df["return"] = 0.0

    up_col = f"pred_q{qh}"
    low_col = f"pred_q{ql}"

    for idx, row in df.iterrows():
        entry_price = row.get("OPENPRC_1d")
        if not _valid_price(entry_price):
            continue
        entry_price = float(entry_price)

        upper_pred = row.get(up_col)
        lower_pred = row.get(low_col)

        trade_return = 0.0

        prev_low = row.get("BIDLO_-4d")
        prev_high = row.get("ASKHI_-4d")

        long_gate = (
            (entry_price - prev_low) / prev_low
            if prev_low is not None else None
        )

        short_gate = (
            (entry_price - prev_high) / prev_high
            if prev_high is not None else None
        )

        if (
            upper_pred is not None
            and not pd.isna(upper_pred)
            and upper_pred > 0
            and long_gate is not None
            and long_gate > 0
            and long_gate <= float(upper_pred)
        ):
            trade_return = _simulate_long(row, entry_price, prev_low, float(upper_pred))

        # Version A (strict): require the realized drop to reach 30% of the predicted lower bound magnitude.
        # cond_short_a = (
        #     lower_pred is not None
        #     and not pd.isna(lower_pred)
        #     and lower_pred < 0
        #     and short_gate is not None
        #     and short_gate < 0
        #     and abs(short_gate) >= 0.3 * abs(float(lower_pred))
        # )

        # Version B (lenient): trigger once the realized drop exceeds 30% of the predicted lower bound (still negative).
        # cond_short_b = (
        #     lower_pred is not None
        #     and not pd.isna(lower_pred)
        #     and lower_pred < 0
        #     and short_gate is not None
        #     and short_gate < 0
        #     and short_gate >= 0.3 * float(lower_pred)
        # )

        # if cond_short_b:
        #     trade_return = _simulate_short(row, entry_price, float(lower_pred))
        df.at[idx, "return"] = trade_return

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
    )
    model_l.fit(X_train, y_train)

    y_pred_h = model_h.predict(X_test)
    y_pred_l = model_l.predict(X_test)
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

    def add_bar_labels(ax, values, fmt="{:.4f}"):
        """Add value labels on top of each bar."""
        for rect, val in zip(ax.patches, values):
            height = rect.get_height()
            ax.annotate(fmt.format(val),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    # 1) Aggregate metrics
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
    agg = agg.sort_values("pinball_sum", ascending=True)

    labels = [f"{m}/{c}" for m, c in zip(agg["model"], agg["criterion"])]
    x = np.arange(len(labels))

    # --- 1) Pinball Loss (Sum) ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    bars = ax.bar(x, agg["pinball_sum"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Pinball Loss (ql+qh)")
    ax.set_title("Quantile Regression Accuracy (Pinball Loss)")
    ax.set_ylim(agg["pinball_sum"].min() * 0.95, agg["pinball_sum"].max() * 1.05)
    add_bar_labels(ax, agg["pinball_sum"])
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "01_pinball_sum_bar.png"), dpi=200)
    plt.close()

    # --- 1.1) Pinball Loss for each window (optional) ---
    plt.figure(figsize=(12, 6))
    for (m, c), group in metrics_df.groupby(["model", "criterion"]):
        plt.plot(group["window"], group["pinball_sum"], marker='o', label=f"{m}/{c}")
    plt.xticks(range(1, n_windows + 1))
    plt.xlabel("Rolling Window")
    plt.ylabel("Pinball Loss (ql + qh)")
    plt.title("Pinball Loss per Rolling Window")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "01-1_pinball_sum_per_window.png"), dpi=200)
    plt.close()

    # --- 2) Coverage ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    bars = ax.bar(x, agg["coverage"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Coverage Rate")
    ax.set_title(f"Interval Coverage between q{args.ql:.2f} and q{args.qh:.2f}")
    ax.set_ylim(agg["coverage"].min() * 0.95, min(1.0, agg["coverage"].max() * 1.05))
    add_bar_labels(ax, agg["coverage"], fmt="{:.3f}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # optional: show as %
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "02_coverage_bar.png"), dpi=200)
    plt.close()

    # --- 3) Cumulative Return ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    bars = ax.bar(x, agg["cum_return"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Final Cumulative Return")
    ax.set_title("Trading Strategy Performance (Cumulative Return)")
    ax.set_ylim(agg["cum_return"].min() * 0.95, agg["cum_return"].max() * 1.05)
    add_bar_labels(ax, agg["cum_return"], fmt="{:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "03_cum_return_bar.png"), dpi=200)
    plt.close()

    # Console summary
    print("\n================================")
    print("Aggregated performance (mean across windows):")
    print(agg.to_string(index=False))
    print("Saved metrics.csv and plots to:", args.outdir)


if __name__ == "__main__":
    run_benchmark()
