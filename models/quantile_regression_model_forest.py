# models/dynamic_quantile_forest.py
"""
Dynamic Quantile Forest that trains models on-the-fly during prediction.
"""

from typing import Dict, List, Optional, Type, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import warnings

from .qunatile_regression_model_leaf_tree import LeafQuantileRegressionTree


class QuantileRegressionModelForest:
    """
    Dynamic Quantile Forest that trains QuantileRegressor models during prediction.
    
    This forest collects all training samples from corresponding leaf nodes across
    all trees, then trains a fresh QuantileRegressor model for each prediction.

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
        Minimum total samples required across leaves when training dynamic model.
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
        include_oob: bool = True,
        min_leaf_agg: int = 8,
        random_state: Optional[int] = None,
        tree_cls: Type[LeafQuantileRegressionTree] = LeafQuantileRegressionTree,
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

        # Learned state
        self.trees_: List[LeafQuantileRegressionTree] = []
        self.leaf_data_: List[Dict[int, Dict[str, np.ndarray]]] = []  # Store X,y for each leaf
        self._fallback_quantile: Optional[float] = None
        self.feature_names_: List[str] = []
        self._rng = np.random.default_rng(random_state)

        # Cache original training data
        self._X_original: Optional[np.ndarray] = None
        self._y_original: Optional[np.ndarray] = None

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

    def _get_leaf_node(self, tree: LeafQuantileRegressionTree, x: np.ndarray, tree_features: List[str] = None) -> int:
        """
        Route a single sample to a leaf node ID using the stored split structure.
        """
        if tree_features is None:
            tree_features = getattr(tree, 'feature_names', self.feature_names_)
            
        node_id = 0
        while node_id in tree.children_map:
            children = tree.children_map.get(node_id, [])
            if not children:
                break
            feat = children[0]["feature_name"]
            try:
                # Use the tree's specific feature mapping
                if feat in tree_features:
                    global_feat_idx = self.feature_names_.index(feat)
                    v = float(x[global_feat_idx])
                else:
                    # Feature not available in this tree, stop traversal
                    break
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
        Train the forest and cache per-leaf sample data (X, y pairs).

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Training features.
        y : array-like or pandas.Series, shape (n_samples,)
            Target variable.

        Returns
        -------
        self : DynamicQuantileForest
            Fitted estimator.
        """
        self.trees_.clear()
        self.leaf_data_.clear()

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_np = X.values
        else:
            X_np = np.asarray(X)
            self.feature_names_ = [
                f"feature_{i}" for i in range(X_np.shape[1])]
        y_np = np.asarray(y)

        # Cache original training data for dynamic model fitting
        self._X_original = X_np.copy()
        self._y_original = y_np.copy()

        n = X_np.shape[0]
        self._fallback_quantile = float(np.quantile(y_np, self.quantile))

        print(f"Training {self.n_estimators} trees for Dynamic Quantile Forest...")

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

            # Build the tree
            tree = self.tree_cls(
                split_criterion=self.split_criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                feature_names=feats,
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
                random_features=True,
                random_thresholds=self.random_thresholds,
                max_threshold_candidates=self.max_threshold_candidates,
            )
            tree.fit(X_bag[:, f_idx], y_bag, quantile=self.quantile)
            self.trees_.append(tree)

            # Store feature mapping for this tree
            tree._forest_feature_mapping = {
                'tree_features': feats,
                'global_indices': f_idx,
                'tree_to_global': {i: f_idx[i] for i in range(len(feats))}
            }

            # Collect X, y data for each leaf node
            leaf_ids = np.array([self._get_leaf_node(tree, x, feats) for x in X_bag])
            leaf_data = defaultdict(lambda: {'X': [], 'y': []})
            
            for lid, xi, yi in zip(leaf_ids, X_bag, y_bag):
                leaf_data[lid]['X'].append(xi)
                leaf_data[lid]['y'].append(yi)

            # Optional OOB enrichment: add unseen samples to leaf data
            if self.include_oob and self.bootstrap:
                oob_mask = np.ones(n, dtype=bool)
                oob_mask[idx] = False
                X_oob = X_np[oob_mask]
                y_oob = y_np[oob_mask]
                if X_oob.size:
                    for x, yi in zip(X_oob, y_oob):
                        lid = self._get_leaf_node(tree, x, feats)
                        leaf_data[lid]['X'].append(x)
                        leaf_data[lid]['y'].append(yi)

            # Convert lists to arrays and store
            processed_leaf_data = {}
            for lid, data in leaf_data.items():
                processed_leaf_data[lid] = {
                    'X': np.array(data['X']),
                    'y': np.array(data['y'])
                }
            
            self.leaf_data_.append(processed_leaf_data)

            if (i + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"  Completed {i + 1}/{self.n_estimators} trees")

        print("Training completed!")
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        quantile: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict by collecting leaf data and training dynamic QuantileRegressor models.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Samples to predict.
        quantile : float, optional
            If provided, overrides the forest's default quantile.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted quantiles.
        """
        if quantile is None:
            quantile = self.quantile

        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        preds: List[float] = []

        print(f"Dynamic prediction for {len(X_np)} samples...")

        for i, x in enumerate(X_np):
            # Collect all X, y data from corresponding leaf nodes across all trees
            all_X = []
            all_y = []

            for tree, leaf_data in zip(self.trees_, self.leaf_data_):
                # Get the tree's specific feature names
                tree_features = getattr(tree, '_forest_feature_mapping', {}).get('tree_features', self.feature_names_)
                lid = self._get_leaf_node(tree, x, tree_features)
                
                if lid in leaf_data:
                    leaf_X = leaf_data[lid]['X']
                    leaf_y = leaf_data[lid]['y']
                    
                    if len(leaf_X) > 0:
                        all_X.extend(leaf_X)
                        all_y.extend(leaf_y)

            # Train dynamic QuantileRegressor if we have enough data
            if len(all_X) >= self.min_leaf_agg:
                try:
                    # Convert to numpy arrays
                    X_combined = np.array(all_X)
                    y_combined = np.array(all_y)
                    
                    # Train QuantileRegressor on combined leaf data
                    model = QuantileRegressor(
                        quantile=quantile,
                        alpha=0.0,
                        solver="highs",
                        fit_intercept=True
                    )
                    model.fit(X_combined, y_combined)
                    
                    # Make prediction for the current sample
                    x_pred = x.reshape(1, -1)
                    pred = model.predict(x_pred)[0]
                    preds.append(float(pred))
                    
                except Exception as e:
                    # If model fitting fails, use fallback
                    warnings.warn(f"Dynamic model fitting failed for sample {i}: {e}")
                    preds.append(self._fallback_quantile)
            else:
                # Not enough data, use fallback quantile
                preds.append(self._fallback_quantile)

            # Progress indicator
            if (i + 1) % max(1, len(X_np) // 10) == 0:
                print(f"  Predicted {i + 1}/{len(X_np)} samples")

        return np.asarray(preds, dtype=float)

    def predict_with_sample_info(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        quantile: Optional[float] = None,
    ) -> tuple[np.ndarray, List[Dict]]:
        """
        Predict with additional information about samples used for each prediction.

        Returns
        -------
        predictions : np.ndarray
            Predicted quantiles.
        sample_info : List[Dict]
            Information about samples used for each prediction.
        """
        if quantile is None:
            quantile = self.quantile

        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        preds: List[float] = []
        sample_info: List[Dict] = []

        for i, x in enumerate(X_np):
            # Collect all X, y data from corresponding leaf nodes across all trees
            all_X = []
            all_y = []
            leaf_counts = defaultdict(int)

            for tree_idx, (tree, leaf_data) in enumerate(zip(self.trees_, self.leaf_data_)):
                tree_features = getattr(tree, '_forest_feature_mapping', {}).get('tree_features', self.feature_names_)
                lid = self._get_leaf_node(tree, x, tree_features)
                
                if lid in leaf_data:
                    leaf_X = leaf_data[lid]['X']
                    leaf_y = leaf_data[lid]['y']
                    
                    if len(leaf_X) > 0:
                        all_X.extend(leaf_X)
                        all_y.extend(leaf_y)
                        leaf_counts[f'tree_{tree_idx}_leaf_{lid}'] = len(leaf_X)

            # Record sample information
            info = {
                'total_samples': len(all_X),
                'leaf_contributions': dict(leaf_counts),
                'used_fallback': False
            }

            # Train dynamic QuantileRegressor if we have enough data
            if len(all_X) >= self.min_leaf_agg:
                try:
                    # Convert to numpy arrays
                    X_combined = np.array(all_X)
                    y_combined = np.array(all_y)
                    
                    # Train QuantileRegressor on combined leaf data
                    model = QuantileRegressor(
                        quantile=quantile,
                        alpha=0.0,
                        solver="highs",
                        fit_intercept=True
                    )
                    model.fit(X_combined, y_combined)
                    
                    # Make prediction for the current sample
                    x_pred = x.reshape(1, -1)
                    pred = model.predict(x_pred)[0]
                    preds.append(float(pred))
                    
                except Exception as e:
                    # If model fitting fails, use fallback
                    info['used_fallback'] = True
                    info['error'] = str(e)
                    preds.append(self._fallback_quantile)
            else:
                # Not enough data, use fallback quantile
                info['used_fallback'] = True
                preds.append(self._fallback_quantile)

            sample_info.append(info)

        return np.asarray(preds, dtype=float), sample_info