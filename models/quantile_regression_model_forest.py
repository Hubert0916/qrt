# models/quantile_regression_model_forest.py

from typing import List, Optional, Type, Union, Tuple

import numpy as np
import pandas as pd

from models.quantile_regression_model_leaf_tree import QuantileRegressionModelTree


class QuantileRegressionModelForest:
    """
    Quantile Regression Forest using Bagging (Bootstrap Aggregating).

    This forest trains multiple QuantileRegressionModelTree instances on
    bootstrapped subsets of data. During prediction, it aggregates the
    predictions from all trees by calculating the mean.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    quantile : float, default=0.5
        Target quantile to predict (0 < q < 1).
    split_criterion : {'loss', 'mse', 'r2'}, default='loss'
        Split objective passed to each tree.
    max_depth : int, default=5
        Maximum depth per tree.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each leaf.
    bootstrap : bool, default=True
        If True, sample with replacement per tree (bagging).
    max_features : {int, 'sqrt', 'log2', None}, default='sqrt'
        Number of features to consider per tree. If None, use all features.
    max_threshold_candidates : int, default=128
        Cap on thresholds evaluated per feature (efficiency/quality trade-off).
    random_thresholds : bool, default=False
        If True, subsample thresholds randomly; else select deterministically.
    random_state : int, optional
        Base RNG seed; each tree is offset by its index for reproducibility.
    tree_cls : Type[QuantileRegressionModelTree], optional
        The tree class to use for the ensemble.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        quantile: float = 0.5,
        split_criterion: str = "loss",
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        max_features: Union[int, str, None] = "sqrt",
        max_threshold_candidates: int = 128,
        random_thresholds: bool = False,
        random_state: Optional[int] = None,
        tree_cls: Type[QuantileRegressionModelTree] = QuantileRegressionModelTree,
    ):
        self.n_estimators = n_estimators
        self.quantile = quantile
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.max_threshold_candidates = max_threshold_candidates
        self.random_thresholds = random_thresholds
        self.random_state = random_state
        self.tree_cls = tree_cls

        # Learned state
        # Stores tuples of (TreeInstance, FeatureIndices)
        self.estimators_: List[Tuple[QuantileRegressionModelTree, List[int]]] = []
        self.feature_names_: List[str] = []
        self._rng = np.random.default_rng(random_state)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _select_feature_subset(self, n_total_features: int) -> List[int]:
        """
        Randomly select feature indices based on `max_features`.
        """
        all_indices = np.arange(n_total_features)

        if isinstance(self.max_features, int):
            k = min(self.max_features, n_total_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_total_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_total_features)))
        else:
            # None or unrecognized implies using all features
            return list(all_indices)

        return self._rng.choice(all_indices, size=k, replace=False).tolist()

    # --------------------------------------------------------------------- #
    # Fit / Predict
    # --------------------------------------------------------------------- #

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "QuantileRegressionModelForest":
        """
        Train the forest using bootstrap aggregation (bagging).

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Training features.
        y : array-like or pandas.Series, shape (n_samples,)
            Target variable.

        Returns
        -------
        self : QuantileRegressionModelForest
            Fitted estimator.
        """
        self.estimators_.clear()

        # Handle Input Data
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self.feature_names_ = [
                f"feature_{i}" for i in range(X_np.shape[1])
            ]

        y_np = np.asarray(y)
        n_samples, n_features = X_np.shape

        print(
            f"Training {self.n_estimators} trees (Bagging, q={self.quantile})..."
        )

        for i in range(self.n_estimators):
            # 1. Bootstrap Sampling
            if self.bootstrap:
                indices = self._rng.choice(
                    n_samples, size=n_samples, replace=True
                )
            else:
                indices = np.arange(n_samples)

            X_bag = X_np[indices]
            y_bag = y_np[indices]

            # 2. Feature Subsampling (Random Subspace)
            feature_indices = self._select_feature_subset(n_features)
            selected_feat_names = [self.feature_names_[j] for j in feature_indices]

            # 3. Initialize Tree
            tree_seed = (
                self.random_state + i if self.random_state is not None else None
            )
            tree = self.tree_cls(
                split_criterion=self.split_criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                feature_names=selected_feat_names,
                random_state=tree_seed,
                random_features=True,
                random_thresholds=self.random_thresholds,
                max_threshold_candidates=self.max_threshold_candidates,
            )

            # 4. Train Tree on subset of features
            # Note: We pass only the selected columns to the tree
            tree.fit(
                X_bag[:, feature_indices],
                y_bag,
                quantile=self.quantile
            )

            # 5. Store the tree and the feature indices it uses
            self.estimators_.append((tree, feature_indices))

            # Logging
            if (i + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"  Completed {i + 1}/{self.n_estimators} trees")

        print("Training completed!")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict by aggregating predictions from all trees (Bagging).

        The final prediction is the average of the predictions returned
        by individual trees.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted quantiles (averaged across trees).
        """
        if not self.estimators_:
            raise RuntimeError("The forest has not been fitted yet.")

        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        n_samples = X_np.shape[0]

        # Array to accumulate predictions: shape (n_estimators, n_samples)
        all_tree_preds = np.zeros((len(self.estimators_), n_samples))

        print(f"Predicting for {n_samples} samples using Bagging...")

        for i, (tree, feature_indices) in enumerate(self.estimators_):
            # Extract only the features used by this specific tree
            X_subset = X_np[:, feature_indices]
            
            # Predict using the single tree (which uses its leaf models)
            all_tree_preds[i, :] = tree.predict(X_subset)

        # Aggregate: Average the predictions across all trees
        # Note: Averaging quantiles is the standard approach for
        # regression forests, though theoretically distinct from
        # aggregating distributions.
        final_preds = np.mean(all_tree_preds, axis=0)

        return final_preds