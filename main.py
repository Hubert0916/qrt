from config import train_period, test_period
from data_loader import load_data, rolling_time
from quantile_forest import RandomForestQuantileRegressor
from models.quantile_regression_tree import QuantileRegressionTree
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train a quantile regression model")
parser.add_argument("--model", type=str, default="RandomForestQuantileRegressor", help="Model name")
parser.add_argument("--max_depth", type=int, default=10, help="Max depth of the tree")
parser.add_argument("--min_samples_leaf", type=int, default=5, help="Min samples per leaf")
parser.add_argument("--qh", type=float, help="Quantiles high")
parser.add_argument("--ql", type=float, help="Quantiles low")
args = parser.parse_args()

if __name__ == "__main__":
    print("================================")
    print(f"Using model: {args.model}")
    print(f"Max depth: {args.max_depth}")
    print(f"Min samples leaf: {args.min_samples_leaf}")
    print("================================")
    print(f"Train Period: {train_period}, Test Period: {test_period}")
    time = rolling_time('data/esg_tfidf_with_return_cleaned.csv', train_period, test_period)
    for i in range(time):
        print(f"Rolling period {i+1}/{time}")
        train_data, test_data = load_data('data/esg_tfidf_with_return_cleaned.csv', train_period, test_period, i)
        x_cols = [c for c in train_data.columns if c.endswith('_TFIDF')]
        x_train = train_data[x_cols]
        y_train = train_data["報酬率"]
        x_test = test_data[x_cols]

        if args.model == "RandomForestQuantileRegressor":
            model = RandomForestQuantileRegressor(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
            model.fit(x_train, y_train)
            y_test_pred_h = model.predict(x_test, quantiles=args.qh)
            y_test_pred_l = model.predict(x_test, quantiles=args.ql)

        elif args.model == "QuantileRegressionTree":
            model_h = QuantileRegressionTree(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, feature_names=x_cols)
            model_h.fit(x_train, y_train, quantiles=args.qh)
            model_l = QuantileRegressionTree(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, feature_names=x_cols)
            model_l.fit(x_train, y_train, quantiles=args.ql)
            y_test_pred_h = model_h.predict(x_test)
            y_test_pred_l = model_l.predict(x_test)
        
        test_data[f'pred_q{args.qh}'] = y_test_pred_h
        test_data[f'pred_q{args.ql}'] = y_test_pred_l
        test_data['return'] = 0

        for row in test_data.iterrows():
            if row[1][f'pred_q{args.qh}'] > 0 and row[1][f'pred_q{args.ql}'] > (row[1]['OPENPRC_1d']-row[1]['BIDLO_-4d'])/row[1]['BIDLO_-4d']:
                test_data.at[row[0], 'return'] = row[1][f'pred_q{args.qh}']
            elif row[1][f'pred_q{args.ql}'] < 0 and row[1][f'pred_q{args.ql}'] < (row[1]['OPENPRC_1d']-row[1]['ASKHI_-4d'])/row[1]['ASKHI_-4d']:
                test_data.at[row[0], 'return'] = -row[1][f'pred_q{args.ql}']
        test_data['total_return'] = test_data['return'].cumsum()
        print(test_data['total_return'])
        print(f"Cumulative return for this period: {test_data['total_return'].tail(1).values[0]}")
        plt.figure(figsize=(10, 6))
        plt.plot(test_data['日期'], test_data['total_return'], marker='o')
        plt.title(f'Cumulative Return over Time (Rolling Period {i+1})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'output/{args.model}/cumulative_return_rolling_{i+1}.png')
        print("================================")