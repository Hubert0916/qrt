"""
Statistical Significance Test for QRT/QRF Benchmarking
------------------------------------------------------
This script loads multiple metrics.csv files from the output directory,
aggregates them, and performs paired statistical tests (Wilcoxon & T-test)
to determine if the performance difference between two models is significant.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_metrics(root_dir="./output"):
    """
    Recursively searches for all metrics.csv files under root_dir
    and combines them into a single DataFrame.
    """
    # Search pattern: ./output/*/metrics.csv
    search_path = os.path.join(root_dir, "*", "metrics.csv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Warning: No metrics.csv files found in {root_dir}")
        return pd.DataFrame()

    df_list = []
    for file in files:
        try:
            temp_df = pd.read_csv(file)
            # Ensure quantile columns are floats
            if 'ql' in temp_df.columns:
                temp_df['ql'] = temp_df['ql'].astype(float)
            if 'qh' in temp_df.columns:
                temp_df['qh'] = temp_df['qh'].astype(float)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully loaded {len(files)} files. Total rows: {len(combined_df)}")
    return combined_df

def test_significance(
    df, 
    model_a='QRT', 
    model_b='QRF', 
    metric='annualized_return',
    target_criterion='loss'
):
    """
    Performs Paired Statistical Tests (Wilcoxon & T-test).
    """
    print(f"\n{'='*60}")
    print(f"Comparison: {model_a} vs {model_b}")
    print(f"Metric: {metric} | Criterion Filter: {target_criterion}")
    print(f"{'='*60}")

    # 1. Filter Data
    # Only compare results generated using the same split criterion
    subset = df[df['criterion'] == target_criterion].copy()
    
    if subset.empty:
        print(f"Error: No data found for criterion='{target_criterion}'.")
        return

    # 2. Create Unique Scenario ID (Window + Quantile Pair)
    # This ensures we are matching the exact same test conditions
    subset['scenario_id'] = (
        subset['window'].astype(str) + "_" + 
        subset['ql'].astype(str) + ":" + 
        subset['qh'].astype(str)
    )

    # 3. Pivot table to align model results side-by-side
    pivot = subset.pivot(index='scenario_id', columns='model', values=metric)

    # Check if models exist in data
    if model_a not in pivot.columns or model_b not in pivot.columns:
        print(f"Error: Model {model_a} or {model_b} not found in data.")
        print(f"Available models: {pivot.columns.tolist()}")
        return

    # Drop NaNs to ensure paired samples
    data = pivot[[model_a, model_b]].dropna()
    print(f"Valid Paired Samples (N): {len(data)}")

    if len(data) < 2:
        print("Not enough samples to perform statistical tests.")
        return

    # 4. Calculate Differences
    # Determine direction logic based on metric type
    is_loss_metric = any(x in metric.lower() for x in ['loss', 'pinball', 'error', 'mse'])
    
    if is_loss_metric:
        # For Loss: Lower is better. 
        # Diff = A - B. If Diff > 0, it means A is larger (worse) than B. 
        # So Positive Diff => B is better.
        diff = data[model_a] - data[model_b]
        desc = f"Positive Diff => {model_b} has LOWER {metric} (Better)"
    else:
        # For Return: Higher is better.
        # Diff = B - A. If Diff > 0, it means B is larger (better) than A.
        diff = data[model_b] - data[model_a]
        desc = f"Positive Diff => {model_b} has HIGHER {metric} (Better)"

    print(f"Diff Definition: {desc}")
    print(f"Mean Diff: {diff.mean():.6f}")
    print(f"{model_b} Win Rate: {(diff > 0).mean():.2%}")

    # 5. Statistical Tests
    # Wilcoxon Signed-Rank Test (Recommended for financial data)
    try:
        w_stat, p_val_w = stats.wilcoxon(data[model_a], data[model_b])
    except ValueError:
        # Happens if all differences are exactly zero
        p_val_w = 1.0
    
    # Paired T-test (Assumes normal distribution)
    t_stat, p_val_t = stats.ttest_rel(data[model_a], data[model_b])

    print("-" * 30)
    print(f"Wilcoxon P-value: {p_val_w:.6f} {'***' if p_val_w < 0.01 else '**' if p_val_w < 0.05 else ''}")
    print(f"T-test P-value:   {p_val_t:.6f} {'***' if p_val_t < 0.01 else '**' if p_val_t < 0.05 else ''}")
    print("-" * 30)

    if p_val_w < 0.05:
        print("Conclusion: >> Statistically Significant Difference <<")
    else:
        print("Conclusion: >> No Significant Difference (Random Noise) <<")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Load data from all subdirectories
    df_all = load_all_metrics("./output")

    if not df_all.empty:
        # --- Example 1: Compare Leaf Models (Annualized Return) ---
        # Adjust 'model_a' and 'model_b' based on the names in your CSV
        test_significance(
            df_all, 
            model_a='QRT_leaf', 
            model_b='QRF_leaf', 
            metric='annualized_return', 
            target_criterion='r2'  # Ensure this matches your data
        )

        # --- Example 2: Compare Loss (Pinball Sum) ---
        # test_significance(
        #     df_all, 
        #     model_a='QRT_leaf', 
        #     model_b='QRF_leaf', 
        #     metric='pinball_sum', 
        #     target_criterion='loss'
        # )