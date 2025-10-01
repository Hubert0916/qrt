# models/quantile_regression_forest.py
"""
Quantile Regression Forest
"""

from typing import List, Optional, Type, Union

import numpy as np
import pandas as pd

from .quantile_regression_tree import QuantileRegressionTree


class QuantileRegressionForest:
    """
    Traditional Random Forest ensemble with tree prediction averaging.

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
        tree_cls: Type[QuantileRegressionTree] = QuantileRegressionTree,  # for different tree implementations
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

        # Learned state.
        self.trees_: List[QuantileRegressionTree] = []
        # Feature names for external arrays.
        self.feature_names_: List[str] = []
        # RNG for bootstrapping and feature subspace selection.
        self._rng = np.random.default_rng(random_state)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _select_feature_subset(self, feature_names: List[str]) -> List[str]:
        """
        Pick a feature subspace per tree based on `max_features`.
        """
        n_features = len(feature_names)
        if isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            # Use all features when None or unrecognized value is given.
            return feature_names
        return self._rng.choice(feature_names, size=k, replace=False).tolist()



    # --------------------------------------------------------------------- #
    # Fit / Predict
    # --------------------------------------------------------------------- #

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ):
        """
        Train the forest using traditional bagging approach.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Training features.
        y : array-like or pandas.Series, shape (n_samples,)
            Target variable.

        Returns
        -------
        self : QuantileRegressionForest
            Fitted estimator.
        """
        self.trees_.clear()

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self.feature_names_ = [
                f"feature_{i}" for i in range(X_np.shape[1])]
        y_np = np.asarray(y)

        n = X_np.shape[0]

        for i in range(self.n_estimators):
            # Bootstrap sampling (bagging).
            if self.bootstrap:
                idx = self._rng.choice(n, size=n, replace=True)
            else:
                idx = np.arange(n)
            X_bag, y_bag = X_np[idx], y_np[idx]

            # Choose a feature subspace per tree.
            feats = self._select_feature_subset(self.feature_names_)
            f_idx = [self.feature_names_.index(f) for f in feats]

            # Build the tree with lightweight configuration.
            tree = self.tree_cls(
                split_criterion=self.split_criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                feature_names=feats,
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
                random_features=True,  # Per-node random subspace inside tree.
                random_thresholds=self.random_thresholds,
                max_threshold_candidates=self.max_threshold_candidates,
            )
            tree.fit(X_bag[:, f_idx], y_bag, quantile=self.quantile)
            self.trees_.append(tree)

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        quantile: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict by averaging individual tree predictions.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Samples to predict.
        quantile : float, optional
            If provided, overrides the forest's default quantile.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted values (average of tree predictions).
        """
        if quantile is None:
            quantile = self.quantile

        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_)

        # Collect predictions from all trees
        all_predictions = []
        for tree in self.trees_:
            # Only use the features this tree was trained on
            tree_features = tree.feature_names if hasattr(tree, 'feature_names') else self.feature_names_
            X_tree = X_df[tree_features]
            tree_preds = tree.predict(X_tree)
            all_predictions.append(tree_preds)

        # Average the predictions
        all_predictions = np.array(all_predictions)  # Shape: (n_trees, n_samples)
        avg_predictions = np.mean(all_predictions, axis=0)
        
        return avg_predictions
