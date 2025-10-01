# models/quantile_regression_forest.py
"""
Quantile Regression Forest with parallel training support.
"""

from typing import Dict, List, Optional, Type, Union, Tuple
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings

import numpy as np
import pandas as pd

from .quantile_regression_tree import QuantileRegressionTree


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
    n_jobs : int, optional
        Number of parallel jobs for training and prediction. If None, use 1 job.
        If -1, use all available CPU cores.
    verbose : bool, default=False
        Enable verbose output during training and prediction.
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
        tree_cls: Type[QuantileRegressionTree] = QuantileRegressionTree,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
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
        self.n_jobs = n_jobs
        self.verbose = verbose

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
    # Parallel Training Functions
    # --------------------------------------------------------------------- #

    @staticmethod
    def _train_single_tree(args: Tuple) -> Tuple[QuantileRegressionTree, Dict[int, List[float]]]:
        """
        Train a single tree in parallel process.
        
        Parameters
        ----------
        args : tuple
            Contains all necessary arguments for tree training.
            
        Returns
        -------
        tuple
            (trained_tree, leaf_values_dict)
        """
        (tree_idx, X_np, y_np, feature_names, quantile, split_criterion, 
         max_depth, min_samples_leaf, max_threshold_candidates, random_thresholds,
         bootstrap, include_oob, random_state, tree_cls, max_features) = args
        
        # Set up RNG for this tree
        tree_rng = np.random.default_rng(random_state + tree_idx if random_state is not None else None)
        n = X_np.shape[0]
        
        # Bootstrap sampling
        if bootstrap:
            idx = tree_rng.choice(n, size=n, replace=True)
        else:
            idx = np.arange(n)
        X_bag, y_bag = X_np[idx], y_np[idx]
        
        # Feature subset selection
        n_features = len(feature_names)
        if isinstance(max_features, int):
            k = min(max_features, n_features)
        elif max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            k = n_features
            
        if k < n_features:
            feats = tree_rng.choice(feature_names, size=k, replace=False).tolist()
        else:
            feats = feature_names
            
        f_idx = [feature_names.index(f) for f in feats]
        
        # Build tree
        tree = tree_cls(
            split_criterion=split_criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            feature_names=feats,
            random_state=random_state + tree_idx if random_state is not None else None,
            random_features=True,
            random_thresholds=random_thresholds,
            max_threshold_candidates=max_threshold_candidates,
        )
        tree.fit(X_bag[:, f_idx], y_bag, quantile=quantile)
        
        # Collect leaf values
        leaf_values = QuantileRegressionForest._collect_leaf_values(
            tree, X_bag, y_bag, X_np, y_np, idx, include_oob, bootstrap, feature_names
        )
        
        return tree, leaf_values
        
    @staticmethod
    def _collect_leaf_values(tree, X_bag, y_bag, X_np, y_np, idx, include_oob, bootstrap, feature_names):
        """Collect leaf values for a single tree."""
        # Helper function to get leaf node
        def get_leaf_node(tree_obj, x):
            node_id = 0
            while node_id in tree_obj.children_map:
                children = tree_obj.children_map.get(node_id, [])
                if not children:
                    break
                feat = children[0]["feature_name"]
                try:
                    fidx = feature_names.index(feat)
                    v = float(x[fidx])
                except (ValueError, IndexError):
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
                    break
                node_id = nxt
            return node_id
            
        # Aggregate in-bag samples per leaf
        leaf_ids = np.array([get_leaf_node(tree, x) for x in X_bag])
        mp = defaultdict(list)
        for lid, yi in zip(leaf_ids, y_bag):
            mp[lid].append(float(yi))
            
        # Optional OOB enrichment
        if include_oob and bootstrap:
            n = X_np.shape[0]
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[idx] = False
            X_oob = X_np[oob_mask]
            y_oob = y_np[oob_mask]
            if X_oob.size:
                for x, yi in zip(X_oob, y_oob):
                    lid = get_leaf_node(tree, x)
                    mp[lid].append(float(yi))
                    
        return dict(mp)

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

        # Determine number of parallel jobs
        if self.n_jobs is None:
            n_jobs = 1
        elif self.n_jobs == -1:
            n_jobs = mp.cpu_count()
        else:
            n_jobs = min(self.n_jobs, mp.cpu_count())

        # Prepare arguments for parallel tree training
        if n_jobs == 1:
            # Sequential training (fallback)
            for i in range(self.n_estimators):
                args = (i, X_np, y_np, self.feature_names_, self.quantile, 
                       self.split_criterion, self.max_depth, self.min_samples_leaf,
                       self.max_threshold_candidates, self.random_thresholds,
                       self.bootstrap, self.include_oob, self.random_state,
                       self.tree_cls, self.max_features)
                
                tree, leaf_values = self._train_single_tree(args)
                self.trees_.append(tree)
                self.leaf_values_.append(leaf_values)
                

        else:
            # Parallel training
            try:
                # Prepare all arguments
                args_list = []
                for i in range(self.n_estimators):
                    args = (i, X_np, y_np, self.feature_names_, self.quantile,
                           self.split_criterion, self.max_depth, self.min_samples_leaf,
                           self.max_threshold_candidates, self.random_thresholds,
                           self.bootstrap, self.include_oob, self.random_state,
                           self.tree_cls, self.max_features)
                    args_list.append(args)
                
                # Train trees in parallel
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # Submit all jobs
                    future_to_idx = {
                        executor.submit(self._train_single_tree, args): i 
                        for i, args in enumerate(args_list)
                    }
                    
                    # Collect results as they complete
                    results = [None] * self.n_estimators
                    completed_count = 0
                    
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            tree, leaf_values = future.result()
                            results[idx] = (tree, leaf_values)
                            completed_count += 1
                                
                        except Exception as e:
                            warnings.warn(f"Tree {idx} failed to train: {e}")
                            # Create a fallback dummy tree if needed
                            results[idx] = None
                    
                    # Store results in order
                    for tree, leaf_values in results:
                        if tree is not None:
                            self.trees_.append(tree)
                            self.leaf_values_.append(leaf_values)
                            
            except Exception as e:
                warnings.warn(f"Parallel training failed: {e}. Falling back to sequential training.")
                # Fallback to sequential training
                self.trees_.clear()
                self.leaf_values_.clear()
                return self._fit_sequential(X_np, y_np)
            
        return self
        
    def _fit_sequential(self, X_np: np.ndarray, y_np: np.ndarray):
        """Fallback sequential training method."""
        for i in range(self.n_estimators):
            args = (i, X_np, y_np, self.feature_names_, self.quantile, 
                   self.split_criterion, self.max_depth, self.min_samples_leaf,
                   self.max_threshold_candidates, self.random_thresholds,
                   self.bootstrap, self.include_oob, self.random_state,
                   self.tree_cls, self.max_features)
            
            tree, leaf_values = self._train_single_tree(args)
            self.trees_.append(tree)
            self.leaf_values_.append(leaf_values)
            
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        quantile: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict the requested quantile by aggregating per-tree leaf samples.

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
        
        # Determine if parallel prediction is beneficial
        n_samples = X_np.shape[0]
        use_parallel = (
            self.n_jobs != 1 and 
            n_samples > 100 and  # Only parallelize for larger datasets
            len(self.trees_) > 10  # And when we have enough trees
        )
        
        if use_parallel and self.n_jobs != 1:
            return self._predict_parallel(X_np, quantile)
        else:
            return self._predict_sequential(X_np, quantile)
            
    def _predict_sequential(self, X_np: np.ndarray, quantile: float) -> np.ndarray:
        """Sequential prediction method."""
        preds: List[float] = []

        for x in X_np:
            bag: List[float] = []

            for tree, leaf_map in zip(self.trees_, self.leaf_values_):
                lid = self._get_leaf_node(tree, x)
                values = leaf_map.get(lid)
                if values:
                    bag.extend(values)

            if len(bag) >= self.min_leaf_agg:
                preds.append(
                    float(np.quantile(np.asarray(bag, dtype=float), quantile)))
            else:
                preds.append(self._fallback_quantile)

        return np.asarray(preds, dtype=float)
        
    def _predict_parallel(self, X_np: np.ndarray, quantile: float) -> np.ndarray:
        """Parallel prediction method."""
        n_samples = X_np.shape[0]
        
        # Determine number of parallel jobs
        if self.n_jobs == -1:
            n_jobs = mp.cpu_count()
        else:
            n_jobs = min(self.n_jobs or 1, mp.cpu_count())
            
        # Split samples into chunks for parallel processing
        chunk_size = max(1, n_samples // (n_jobs * 2))  # Create more chunks than workers
        chunks = [X_np[i:i+chunk_size] for i in range(0, n_samples, chunk_size)]
        
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit prediction jobs
                futures = [
                    executor.submit(self._predict_chunk, chunk, quantile, i)
                    for i, chunk in enumerate(chunks)
                ]
                
                # Collect results
                results = [None] * len(chunks)
                for future in as_completed(futures):
                    chunk_idx, chunk_preds = future.result()
                    results[chunk_idx] = chunk_preds
                    
            # Concatenate results
            all_preds = np.concatenate([r for r in results if r is not None])
            return all_preds
            
        except Exception as e:
            if self.verbose:
                print(f"Parallel prediction failed: {e}. Falling back to sequential.")
            return self._predict_sequential(X_np, quantile)
            
    def _predict_chunk(self, X_chunk: np.ndarray, quantile: float, chunk_idx: int) -> Tuple[int, np.ndarray]:
        """Predict for a chunk of samples."""
        preds = []
        
        for x in X_chunk:
            bag: List[float] = []
            
            for tree, leaf_map in zip(self.trees_, self.leaf_values_):
                lid = self._get_leaf_node(tree, x)
                values = leaf_map.get(lid)
                if values:
                    bag.extend(values)
                    
            if len(bag) >= self.min_leaf_agg:
                preds.append(float(np.quantile(np.asarray(bag, dtype=float), quantile)))
            else:
                preds.append(self._fallback_quantile)
                
        return chunk_idx, np.asarray(preds, dtype=float)
        
    def predict_interval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict prediction intervals by calling predict twice.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Samples to predict.
        lower_quantile : float, default=0.1
            Lower quantile for the interval.
        upper_quantile : float, default=0.9
            Upper quantile for the interval.

        Returns
        -------
        (lower, upper) : tuple of np.ndarray
            Element-wise interval bounds.
        """
        if not (0.0 < lower_quantile < upper_quantile < 1.0):
            raise ValueError("Quantiles must satisfy 0 < lower < upper < 1")
            
        lower_preds = self.predict(X, quantile=lower_quantile)
        upper_preds = self.predict(X, quantile=upper_quantile)
        
        return lower_preds, upper_preds
