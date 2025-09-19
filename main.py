import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models.quantile_regression_forest import QuantileRegressionForest
from models.quantile_regression_tree import QuantileRegressionTree
from utils.config import train_period, test_period
from utils.data_loader import rolling_time, load_data

parser = argparse.ArgumentParser(
    description="Train quantile models and compute cumulative return")
parser.add_argument("--model", type=str, default="QuantileRegressionForest",
                    choices=["QuantileRegressionForest", "QuantileRegressionTree"])
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--min_samples_leaf", type=int, default=10)
parser.add_argument("--qh", type=float, default=0.9)
parser.add_argument("--ql", type=float, default=0.1)
args = parser.parse_args()


# TO DO: Implement trading strategy based on quantile predictions
def trading_rule(test_df, qh, ql):
    """依高低 quantile 訊號建立簡單多空策略，計算累積報酬"""
    test_df = test_df.copy()
    test_df['return'] = 0.0

    for idx, row in test_df.iterrows():
        up = row[f'pred_q{qh}']
        low = row[f'pred_q{ql}']
        # 進多：預測高 quantile 為正且大於基準
        if up > 0 and up > (row['OPENPRC_1d'] - row['BIDLO_-4d']) / row['BIDLO_-4d']:
            test_df.at[idx, 'return'] = up
        # 進空：預測低 quantile 為負且低於基準
        elif low < 0 and low < (row['OPENPRC_1d'] - row['ASKHI_-4d']) / row['ASKHI_-4d']:
            test_df.at[idx, 'return'] = -low

    test_df['total_return'] = test_df['return'].cumsum()
    return test_df


if __name__ == "__main__":
    os.makedirs(f"output/{args.model}", exist_ok=True)

    print("================================")
    print(f"Using model: {args.model}")
    print(f"Train Period: {train_period}, Test Period: {test_period}")
    print("================================")

    time = rolling_time(
        "data/esg_tfidf_with_return_cleaned.csv", train_period, test_period)

    cumulative_returns = []

    for i in range(time):
        print(f"Rolling period {i+1}/{time}")
        train_data, test_data = load_data(
            "data/esg_tfidf_with_return_cleaned.csv", train_period, test_period, i)

        x_cols = [c for c in train_data.columns if c.endswith('_TFIDF')]
        X_train, y_train = train_data[x_cols], train_data["報酬率"]
        X_test = test_data[x_cols]

        if args.model == "QuantileRegressionForest":
            model = QuantileRegressionForest(
                n_estimators=50,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                split_criterion="loss",
                max_threshold_candidates=128,
                include_oob=True,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_test_pred_h = model.predict(X_test, quantile=args.qh)
            y_test_pred_l = model.predict(X_test, quantile=args.ql)
        else:
            model_h = QuantileRegressionTree(
                max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, feature_names=x_cols)
            model_h.fit(X_train, y_train, args.qh)
            model_l = QuantileRegressionTree(
                max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, feature_names=x_cols)
            model_l.fit(X_train, y_train, args.ql)
            y_test_pred_h = model_h.predict(X_test)
            y_test_pred_l = model_l.predict(X_test)

        test_data[f'pred_q{args.qh}'] = y_test_pred_h
        test_data[f'pred_q{args.ql}'] = y_test_pred_l

        test_data = trading_rule(test_data, args.qh, args.ql)
        cumulative_returns.append(test_data['total_return'].iloc[-1])

        print(
            f"Cumulative return for this period: {test_data['total_return'].iloc[-1]:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(test_data['日期'], test_data['total_return'], marker='o')
        plt.title(f"Cumulative Return — Rolling {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/{args.model}/cumulative_return_{i+1}.png")
        plt.close()

    print("================================")
    print(
        f"Average cumulative return across all windows: {np.mean(cumulative_returns):.4f}")
