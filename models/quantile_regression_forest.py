# models/quantile_regression_forest.py
"""
Quantile Regression Forest
"""

from typing import Dict, List, Optional, Type, Union
from collections import defaultdict

import numpy as np
import pandas as pd

from .quantile_regression_tree import QuantileRegressionTree
from .qunatile_regression_model_leaf_tree import LeafQuantileRegressionTree


class QuantileRegressionForest:
    """
    Ensemble of QuantileRegressionTree with per-leaf sample aggregation.

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
    include_oob : bool, default=True
        If True and bootstrap=True, enrich leaf sample bags with OOB samples.
    min_leaf_agg : int, default=8
        Minimum total samples required across leaves when aggregating a
        prediction; otherwise use global fallback quantile.
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
        max_features: Union[int, str, None] = None,
        max_threshold_candidates: int = 128,
        random_thresholds: bool = False,
        include_oob: bool = True,
        min_leaf_agg: int = 8,
        random_state: Optional[int] = None,
        tree_cls: Type[QuantileRegressionTree] = QuantileRegressionTree,
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
        self.include_oob = include_oob
        self.min_leaf_agg = min_leaf_agg
        self.random_state = random_state
        self.tree_cls = tree_cls

        # Learned state.
        self.trees_: List[QuantileRegressionTree] = []
        # For each tree, a mapping: leaf_id -> list of y values (in-bag + OOB).
        self.leaf_values_: List[Dict[int, List[float]]] = []
        # Global fallback quantile used if per-sample aggregation is too small.
        self._fallback_quantile: Optional[float] = None
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

    def _get_leaf_node(self, tree: QuantileRegressionTree, x: np.ndarray) -> int:
        """
        Route a single sample to a leaf node ID using the stored split structure.

        Notes
        -----
        This mirrors the tree's predict path but returns the terminal node ID
        instead of a prediction. If a feature is missing or an inconsistency is
        encountered, we break and return the last reachable node.
        """
        node_id = 0
        while node_id in tree.children_map:
            children = tree.children_map.get(node_id, [])
            if not children:
                break
            feat = children[0]["feature_name"]
            try:
                fidx = self.feature_names_.index(feat)
                v = float(x[fidx])
            except (ValueError, IndexError):
                # Missing feature or index mismatch; stop traversal gracefully.
                break

            nxt = None
            for ch in children:
                thr = ch["numeric_threshold"]
                if ch["condition"] == "<" and v < thr:
                    nxt = ch["node_id"]
                    break
                if ch["condition"] == ">=" and v >= thr:
                    nxt = ch["node_id"]
                    break

            if nxt is None:
                # No child condition matched; stop at current node.
                break
            node_id = nxt
        return node_id

    # --------------------------------------------------------------------- #
    # Fit / Predict
    # --------------------------------------------------------------------- #

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ):
        """
        Train the forest and cache per-leaf sample bags (with optional OOB).

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
        self.leaf_values_.clear()

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self.feature_names_ = [
                f"feature_{i}" for i in range(X_np.shape[1])]
        y_np = np.asarray(y)

        n = X_np.shape[0]
        self._fallback_quantile = float(np.quantile(y_np, self.quantile))

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

            # Aggregate in-bag samples per leaf.
            leaf_ids = np.array([self._get_leaf_node(tree, x) for x in X_bag])
            mp = defaultdict(list)
            for lid, yi in zip(leaf_ids, y_bag):
                mp[lid].append(float(yi))

            # Optional OOB enrichment: push unseen samples through this tree.
            if self.include_oob and self.bootstrap:
                oob_mask = np.ones(n, dtype=bool)
                oob_mask[idx] = False
                X_oob = X_np[oob_mask]
                y_oob = y_np[oob_mask]
                if X_oob.size:
                    for x, yi in zip(X_oob, y_oob):
                        lid = self._get_leaf_node(tree, x)
                        mp[lid].append(float(yi))

            # Freeze the mapping for this tree.
            self.leaf_values_.append(dict(mp))

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        quantile: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict using different strategies based on tree type.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Samples to predict.
        quantile : float, optional
            If provided, overrides the forest's default quantile.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted values.
        """
        if quantile is None:
            quantile = self.quantile

        if isinstance(X, pd.DataFrame):
            X_df = X
            X_np = X.values
        else:
            X_np = np.asarray(X)
            X_df = pd.DataFrame(X_np, columns=self.feature_names_)

        # Check if we're using LeafQuantileRegressionTree (model-based)
        if self.tree_cls.__name__ == 'LeafQuantileRegressionTree':
            return self._predict_model_based(X_df, quantile)
        else:
            return self._predict_sample_based(X_np, quantile)
    
    def _predict_model_based(self, X_df: pd.DataFrame, quantile: float) -> np.ndarray:
        """
        Predict using model-based trees (e.g., LeafQuantileRegressionTree).
        Each tree directly predicts using its internal QuantileRegressor models.
        """
        all_predictions = []
        
        for tree in self.trees_:
            # Get the features this tree was trained on
            tree_features = getattr(tree, 'feature_names', self.feature_names_)
            X_tree = X_df[tree_features]
            
            # Use tree's direct predict method (which uses QuantileRegressor in leaves)
            tree_preds = tree.predict(X_tree)
            all_predictions.append(tree_preds)
        
        # Average the predictions from all trees
        all_predictions = np.array(all_predictions)  # Shape: (n_trees, n_samples)
        avg_predictions = np.mean(all_predictions, axis=0)
        
        return avg_predictions
    
    def _predict_sample_based(self, X_np: np.ndarray, quantile: float) -> np.ndarray:
        """
        Predict using sample aggregation (traditional quantile forest approach).
        Aggregates leaf samples and computes quantiles.
        """
        preds: List[float] = []

        for x in X_np:
            bag: List[float] = []

            # Collect samples from all trees' leaf nodes
            for tree, leaf_map in zip(self.trees_, self.leaf_values_):
                lid = self._get_leaf_node(tree, x)
                values = leaf_map.get(lid)
                if values:
                    bag.extend(values)

            if len(bag) >= self.min_leaf_agg:
                preds.append(
                    float(np.quantile(np.asarray(bag, dtype=float), quantile)))
            else:
                # Fall back to global quantile if local bag is too small.
                preds.append(self._fallback_quantile)

        return np.asarray(preds, dtype=float)
