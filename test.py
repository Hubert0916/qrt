from models.qunatile_regression_model_node_tree import NodeQuantileRegressionTree
from models.quantile_regression_forest import QuantileRegressionForest
from utils.data_loader import load_data, rolling_time
import pandas as pd

if __name__ == "__main__":
     n_windows = rolling_time("data/esg_tfidf_with_return_cleaned.csv", 5, 1)
     for win in range(n_windows):
        print(f"\n=== Rolling window {win + 1}/{n_windows} ===")
        train_df, test_df = load_data("data/esg_tfidf_with_return_cleaned.csv", 5, 1, win)
        x_cols = [c for c in train_df.columns if c.endswith("_TFIDF")]
        X_train, y_train = train_df[x_cols], train_df["報酬率"]
        X_test, y_test = test_df[x_cols], test_df["報酬率"]

        model = NodeQuantileRegressionTree(max_depth=3)
        model.fit(X_train, y_train, quantile=0.7)
        preds = model.predict(X_test)
        