# plot_advanced_metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="../output/benchmark/metrics.csv")
parser.add_argument("--outdir", type=str, default="../output/advanced")
parser.add_argument("--ql", type=float, default=0.3)
parser.add_argument("--qh", type=float, default=0.7)
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.outdir, exist_ok=True)

    # Read metrics
    metrics_df = pd.read_csv(args.input)
    n_windows = metrics_df["window"].nunique()

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

    # --- Pinball Loss for each window (optional) ---
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
    plt.savefig(os.path.join(args.outdir, "pinball_sum_per_window.png"), dpi=200)
    plt.close()

    # --- Signed Calibration Gap per Window (coverage − (qh−ql)) ---
    nominal = args.qh - args.ql
    tol = getattr(args, "coverage_tolerance", 0.05)  # 沒有 argparse 也會用 0.05

    work = metrics_df.copy()
    work["signed_gap"] = work["coverage"].astype(float) - nominal

    wmin = int(work["window"].min())
    wmax = int(work["window"].max())
    xs = np.arange(wmin, wmax + 1)

    plt.figure(figsize=(12, 6))
    for (m, c), g in work.groupby(["model", "criterion"]):
        gs = g.sort_values("window")
        plt.plot(gs["window"], gs["signed_gap"], marker="o", linewidth=1.6, label=f"{m}/{c}")

    plt.axhline(0.0, linewidth=1.0)
    plt.fill_between(xs, -tol, tol, alpha=0.12, label=f"±{tol:.02f} tolerance")

    plt.xticks(xs)
    plt.xlabel("Rolling Window")
    plt.ylabel("Signed Calibration Gap = coverage − (qh−ql)")
    plt.title(f"Signed Calibration Gap per Window (nominal = {nominal:.2f})")
    plt.legend(ncol=3, fontsize=9)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "04_signed_calibration_gap_per_window.png"), dpi=200)
    plt.close()


    # --- Culmulative Return for each window (optional) ---
    plt.figure(figsize=(12, 6))
    for (m, c), group in metrics_df.groupby(["model", "criterion"]):
        plt.plot(group["window"], group["cum_return"], marker='o', label=f"{m}/{c}")
    plt.xticks(range(1, n_windows + 1))
    plt.xlabel("Rolling Window")
    plt.ylabel("Final Cumulative Return")
    plt.title("Cumulative Return per Rolling Window")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cum_return_per_window.png"), dpi=200)
    plt.close()

    # --- Correlation between Pinball Loss and Cumulative Return for each window ---
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for i, ((m, c), group) in enumerate(metrics_df.groupby(["model", "criterion"])):
        ax.scatter(group["pinball_sum"], group["cum_return"], label=f"{m}/{c}", marker=markers[i % len(markers)])
        for _, row in group.iterrows():
            ax.annotate(f"W{int(row['window'])}", (row["pinball_sum"], row["cum_return"]),
                        textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    ax.set_xlabel("Pinball Loss (ql + qh)")
    ax.set_ylabel("Final Cumulative Return")
    ax.set_title("Pinball Loss vs Cumulative Return per Rolling Window")
    ax.legend()
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pinball_vs_cum_return.png"), dpi=200)
    plt.close()

    # --- Correlation between Pinball Loss and Cumulative Return for each window removed outlier ---
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for i, ((m, c), group) in enumerate(metrics_df.groupby(["model", "criterion"])):
        # Remove outliers based on 1.5*IQR rule
        q1 = group["pinball_sum"].quantile(0.25)
        q3 = group["pinball_sum"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_group = group[(group["pinball_sum"] >= lower_bound) & (group["pinball_sum"] <= upper_bound)]
        
        ax.scatter(filtered_group["pinball_sum"], filtered_group["cum_return"], label=f"{m}/{c}", marker=markers[i % len(markers)])
        for _, row in filtered_group.iterrows():
            ax.annotate(f"W{int(row['window'])}", (row["pinball_sum"], row["cum_return"]),
                        textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    ax.set_xlabel("Pinball Loss (ql + qh)")
    ax.set_ylabel("Final Cumulative Return")
    ax.set_title("Pinball Loss vs Cumulative Return per Rolling Window (Outliers Removed)")
    ax.legend()
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pinball_vs_cum_return_no_outlier.png"), dpi=200)
    plt.close()

    # -- Correlation between Coverage and Cumulative Return for each window ---
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for i, ((m, c), group) in enumerate(metrics_df.groupby(["model", "criterion"])):
        ax.scatter(group["coverage"], group["cum_return"], label=f"{m}/{c}", marker=markers[i % len(markers)])
        for _, row in group.iterrows():
            ax.annotate(f"W{int(row['window'])}", (row["coverage"], row["cum_return"]),
                        textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    ax.set_xlabel("Coverage Rate")
    ax.set_ylabel("Final Cumulative Return")
    ax.set_title("Coverage vs Cumulative Return per Rolling Window")
    ax.legend()
    ax.grid(True, linewidth=0.3)    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coverage_vs_cum_return.png"), dpi=200)
    plt.close()