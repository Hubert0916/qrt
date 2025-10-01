# models/quantile_regression_tree.py
"""
High-performance Quantile Regression Tree (QRT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import QuantileRegressor


@dataclass
class _Node:
    """Internal node representation stored during breadth-first building."""
    node_id: int
    parent_id: Optional[int]
    depth: int
    feature_name: Optional[str] = None
    condition: Optional[str] = None  # Either '<' or '>='.
    threshold: Optional[float] = None
    model: Optional[object] = None  # QuantileRegressor model for leaf nodes
    indices: Optional[np.ndarray] = None  # Index view into the training set.


class NodeQuantileRegressionTree:
    """
    Quantile regression tree supporting three splitting criteria.

    Parameters
    ----------
    split_criterion : {'r2', 'loss', 'mse'}, default='loss'
        Split objective per node. For 'r2' higher is better; for 'loss' and
        'mse' lower is better.
    max_depth : int, default=5
        Maximum tree depth (root has depth 1).
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child after a split.
    feature_names : list[str], optional
        Feature names; inferred from DataFrame X if not provided.
    random_state : int, optional
        RNG seed for reproducibility.
    random_features : bool, default=False
        If True, sample √d features at each split (random subspace).
    random_thresholds : bool, default=False
        If True, subsample candidate thresholds per feature at each node.
    max_threshold_candidates : int, optional
        Cap on number of thresholds evaluated per feature.
    """

    def __init__(
        self,
        split_criterion: str = "loss",
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        feature_names: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        random_features: bool = True,
        random_thresholds: bool = True,
        max_threshold_candidates: Optional[int] = 128,
    ) -> None:
        if split_criterion not in {"r2", "loss", "mse"}:
            raise ValueError("split_criterion must be 'r2', 'loss', or 'mse'")

        self.split_criterion = split_criterion
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.feature_names = feature_names
        self.random_state = random_state
        self.random_features = random_features
        self.random_thresholds = random_thresholds
        self.max_threshold_candidates = max_threshold_candidates

        # Fitted attributes.
        self.tree_nodes: Dict[int, Dict] = {}
        self.children_map: Dict[int, List[Dict]] = {}
        self.node_models: Dict[int, QuantileRegressor] = {}  # Store models for leaf nodes
        self.quantile: Optional[float] = None

        # Training cache.
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(random_state)

        # Interval trees for predict_interval().
        self.lower_tree: Optional["NodeQuantileRegressionTree"] = None
        self.upper_tree: Optional["NodeQuantileRegressionTree"] = None

    # ------------------------------ Utilities -------------------------------- #

    @staticmethod
    def _pinball_loss(y: np.ndarray, q: float, q_value) -> float:
        """
        Compute pinball (quantile) loss for predictions.

        Loss is q * |r| when r <= 0 and (1 - q) * |r| when r > 0, where
        r = y - q_value.

        Parameters
        ----------
        y : np.ndarray
            Observed targets.
        q : float
            Quantile in (0, 1).
        q_value : float or np.ndarray
            Predicted q-th quantile(s). Can be scalar or array.

        Returns
        -------
        float
            Sum of pinball losses across y.
        """
        r = y - q_value
        neg = r <= 0
        return q * (-r[neg]).sum() + (1.0 - q) * r[~neg].sum()

    # --------------------------- Split evaluation ---------------------------- #

    def _evaluate_split_loss(
        self,
        X_left: np.ndarray,
        y_left: np.ndarray,
        X_right: np.ndarray,
        y_right: np.ndarray,
        quantile: float,
    ) -> float:
        """
        Evaluate a split by total pinball loss using QuantileRegressor models.

        Trains QuantileRegressor models on left and right subsets, then computes
        pinball loss using model predictions.

        Returns
        -------
        float
            Left loss + right loss (lower is better).
        """

        def model_loss(X_subset: np.ndarray, y_subset: np.ndarray) -> float:
            if y_subset.size == 0:
                return 0.0
            
            # Try to fit QuantileRegressor model
            model = self._fit_quantile_model(X_subset, y_subset, quantile)

            # if model is None:
            #     # Fallback to quantile-based prediction if model fails
            #     k = int(np.floor(quantile * (y_subset.size - 1)))
            #     qv = np.partition(y_subset, k)[k]
            #     return self._pinball_loss(y_subset, quantile, qv)
            
            # Use model predictions to calculate pinball loss
            predictions = model.predict(X_subset)
            return self._pinball_loss(y_subset, quantile, predictions)

        return model_loss(X_left, y_left) + model_loss(X_right, y_right)

    @staticmethod
    def _evaluate_split_mse_prefix(
        count_L: np.ndarray,
        sum_L: np.ndarray,
        sumsq_L: np.ndarray,
        total_n: int,
        total_sum: float,
        total_sumsq: float,
    ) -> np.ndarray:
        """
        Vectorized within-group SSE for all cut positions via prefix stats.

        Parameters are left-side prefix counts/sums/sumsq at each cut and the
        global totals. Returns total SSE (left + right) per cut.
        """
        # Left statistics.
        nL = count_L
        sL = sum_L
        ssL = sumsq_L

        # Right statistics.
        nR = total_n - nL
        sR = total_sum - sL
        ssR = total_sumsq - ssL

        # SSE = sum(y^2) - (sum(y)^2 / n) for each side; guard n == 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            varL_times_n = ssL - np.where(nL > 0, (sL * sL) / nL, 0.0)
            varR_times_n = ssR - np.where(nR > 0, (sR * sR) / nR, 0.0)

        return varL_times_n + varR_times_n

    def _evaluate_split_r2(
        self,
        X_left: np.ndarray,
        y_left: np.ndarray,
        X_right: np.ndarray,
        y_right: np.ndarray,
        quantile: float,
    ) -> float:
        """
        Evaluate a split by R² using QuantileRegressor model predictions.

        Returns
        -------
        float
            R² score across concatenated children (higher is better).
        """
        if y_left.size + y_right.size == 0:
            return -np.inf

        preds: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        for X_subset, y_subset in [(X_left, y_left), (X_right, y_right)]:
            if y_subset.size == 0:
                continue
                
            # Try to fit QuantileRegressor model
            model = self._fit_quantile_model(X_subset, y_subset, quantile)
            # if model is None:
            #     # Fallback to quantile-based prediction if model fails
            #     k = int(np.floor(quantile * (y_subset.size - 1)))
            #     qv = np.partition(y_subset, k)[k]
            #     preds.append(np.full(y_subset.size, qv, dtype=float))
            # else:
            #     # Use model predictions
            predictions = model.predict(X_subset)
            preds.append(predictions)
            
            labels.append(y_subset)

        if not labels:
            return -np.inf

        y_all = np.concatenate(labels)
        p_all = np.concatenate(preds)
        return r2_score(y_all, p_all)

    # ---------------------- Best split for a single feature ------------------ #

    def _best_split_for_feature(
        self,
        feature_idx: int,
        idx: np.ndarray,
        quantile: float,
    ) -> Tuple[Optional[float], float]:
        """
        Find best threshold for a given feature on samples `idx`.

        Returns
        -------
        (threshold, score)
            For 'mse' and 'loss', lower score is better.
            For 'r2', higher score is better.
        """
        Xf = self._X[idx, feature_idx]
        y = self._y[idx]

        # Stable sort ensures reproducibility for equal feature values.
        order = np.argsort(Xf, kind="mergesort")
        Xs = Xf[order]
        ys = y[order]

        # Candidate cuts are between distinct feature values.
        diffs = np.diff(Xs)
        cut_positions = np.nonzero(diffs > 0)[0]  # split after these indices
        if cut_positions.size == 0:
            default = np.inf if self.split_criterion in {
                "loss", "mse"} else -np.inf
            # print("No valid cut positions found.")
            return None, default

        # Optional subsampling of candidate thresholds.
        if (
            self.max_threshold_candidates is not None
            and cut_positions.size > self.max_threshold_candidates
        ):
            if self.random_thresholds:
                cut_positions = self._rng.choice(
                    cut_positions,
                    size=self.max_threshold_candidates,
                    replace=False,
                )
                cut_positions.sort()
                # print(f"cut_positions size: {len(cut_positions)}")
            else:
                # Uniform stride selection for determinism.
                stride = max(1, cut_positions.size //
                             self.max_threshold_candidates)
                cut_positions = cut_positions[::stride][
                    : self.max_threshold_candidates
                ]

        # MSE path: vectorized evaluation via prefix statistics.
        if self.split_criterion == "mse":
            ones = np.ones_like(ys)
            prefix_n = np.cumsum(ones)
            prefix_sum = np.cumsum(ys)
            prefix_sumsq = np.cumsum(ys * ys)

            total_n = ys.size
            total_sum = prefix_sum[-1]
            total_sumsq = prefix_sumsq[-1]

            # Left stats at each candidate position.
            count_L = prefix_n[cut_positions]
            sum_L = prefix_sum[cut_positions]
            sumsq_L = prefix_sumsq[cut_positions]

            total_sse = self._evaluate_split_mse_prefix(
                count_L, sum_L, sumsq_L, total_n, total_sum, total_sumsq
            )
            best_idx = int(np.argmin(total_sse))
            best_score = float(total_sse[best_idx])
            pos = int(cut_positions[best_idx])
            thr = (Xs[pos] + Xs[pos + 1]) * 0.5
            return thr, best_score

        # Quantile-loss / R² path: iterate over contiguous slices.
        best_thr: Optional[float] = None

        if self.split_criterion == "r2":
            best_score = -np.inf

            def better(cur: float, best: float) -> bool:
                return cur > best
        else:
            best_score = np.inf

            def better(cur: float, best: float) -> bool:
                return cur < best

        # Get full feature matrix for this node
        X_node = self._X[idx]
        
        for pos in np.nditer(cut_positions):
            pos = int(pos)
            yL = ys[: pos + 1]
            yR = ys[pos + 1:]

            # Enforce leaf size constraint early.
            if yL.size < self.min_samples_leaf or yR.size < self.min_samples_leaf:
                continue

            # Get corresponding X subsets (need to use original order mapping)
            left_mask = order <= pos
            right_mask = order > pos
            XL = X_node[left_mask]
            XR = X_node[right_mask]

            if self.split_criterion == "loss":
                score = self._evaluate_split_loss(XL, yL, XR, yR, quantile)
            else:
                score = self._evaluate_split_r2(XL, yL, XR, yR, quantile)

            if better(score, best_score):
                best_score = float(score)
                best_thr = float((Xs[pos] + Xs[pos + 1]) * 0.5)

        return best_thr, best_score

    # -------------------- Best split across candidate features ---------------- #

    def _get_best_split(
        self,
        idx: np.ndarray,
        quantile: float,
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Choose the (feature, threshold) pair that optimizes the split objective.

        Returns
        -------
        (best_feature_index, best_threshold)
            Either may be None if no valid split exists.
        """
        n_features = self._X.shape[1]

        if self.random_features:
            mtry = max(1, int(np.sqrt(n_features)))
            features = self._rng.choice(n_features, size=mtry, replace=False)
        else:
            features = np.arange(n_features)

        if self.split_criterion == "r2":
            best_score = -np.inf

            def better(cur: float, best: float) -> bool:
                return cur > best
        else:
            best_score = np.inf

            def better(cur: float, best: float) -> bool:
                return cur < best

        best_feat: Optional[int] = None
        best_thr: Optional[float] = None

        for f in features:
            # print(len(features))
            # print(f"Evaluating feature {f}...")
            thr, score = self._best_split_for_feature(f, idx, quantile)
            if thr is None:
                continue
            if better(score, best_score):
                best_score = score
                best_feat = int(f)
                best_thr = float(thr)

        return best_feat, best_thr

    # ------------------------------- Tree building --------------------------- #
    
    def _fit_quantile_model(self, X_subset: np.ndarray, y_subset: np.ndarray, quantile: float) -> Optional[QuantileRegressor]:
        """
        Safely fit a QuantileRegressor model for a subset of data.
        
        Parameters
        ----------
        X_subset : np.ndarray
            Feature matrix for the subset.
        y_subset : np.ndarray  
            Target values for the subset.
        quantile : float
            Target quantile.
            
        Returns
        -------
        QuantileRegressor or None
            Fitted model, or None if fitting fails.
        """
        if X_subset.shape[0] == 0 or y_subset.size == 0:
            return None
            
        try:
            model = QuantileRegressor(
                quantile=quantile,
                alpha=0.0,
                solver="highs", 
                fit_intercept=True
            )
            model.fit(X_subset, y_subset)
            return model
        except Exception:
            # If model fitting fails (e.g., numerical issues), return None
            return None

    def _terminal_model(self, idx: np.ndarray, quantile: float) -> QuantileRegressor:
        """
        Create and fit a QuantileRegressor model for indices `idx`.
        """
        X_leaf = self._X[idx]
        y_leaf = self._y[idx]
        
        # Create and fit QuantileRegressor
        model = QuantileRegressor(
            quantile=quantile,
            alpha=0.0,
            solver="highs",
            fit_intercept=True
        )
        model.fit(X_leaf, y_leaf)
        return model

    def _build_tree(self, indices: np.ndarray, quantile: float) -> None:
        """
        Build the tree breadth-first with index views to avoid data copies.
        """
        next_id = 0
        root = _Node(node_id=next_id, parent_id=None, depth=1, indices=indices)
        next_id += 1

        q: deque[_Node] = deque([root])
        self.tree_nodes.clear()
        self.children_map.clear()
        self.node_models.clear()

        while q:
            node = q.popleft()
            idx = node.indices
            depth = node.depth

            # Stop if depth limit reached or not enough samples to split.
            min_needed = max(2 * self.min_samples_leaf, 1)
            if depth >= self.max_depth or idx.size <= min_needed:
                node.model = self._terminal_model(idx, quantile)
                self._register_node(node)
                continue

            feat, thr = self._get_best_split(idx, quantile)
            if feat is None or thr is None:
                node.model = self._terminal_model(idx, quantile)
                self._register_node(node)
                continue

            # Internal node (stores split rule).
            node.feature_name = self.feature_names[feat]
            node.threshold = float(thr)
            self._register_node(node)

            # Partition indices by the chosen rule.
            Xf = self._X[idx, feat]
            left_mask = Xf < thr
            right_mask = ~left_mask
            left_idx = idx[left_mask]
            right_idx = idx[right_mask]

            # If any child violates min_samples_leaf, convert to leaf.
            if (
                left_idx.size < self.min_samples_leaf
                or right_idx.size < self.min_samples_leaf
            ):
                model = self._terminal_model(idx, quantile)
                self.node_models[node.node_id] = model
                self.children_map.pop(node.node_id, None)
                continue

            # Enqueue valid children for further splitting.
            left = _Node(
                node_id=next_id,
                parent_id=node.node_id,
                depth=depth + 1,
                feature_name=node.feature_name,
                condition="<",
                threshold=node.threshold,
                indices=left_idx,
            )
            next_id += 1

            right = _Node(
                node_id=next_id,
                parent_id=node.node_id,
                depth=depth + 1,
                feature_name=node.feature_name,
                condition=">=",
                threshold=node.threshold,
                indices=right_idx,
            )
            next_id += 1

            q.append(left)
            q.append(right)

    def _register_node(self, node: _Node) -> None:
        """
        Persist a node into `tree_nodes` and update the parent→children map.
        """
        node_dict = dict(
            node_id=node.node_id,
            parent_id=node.parent_id,
            feature_name=node.feature_name,
            condition=node.condition,
            numeric_threshold=node.threshold,
        )
        self.tree_nodes[node.node_id] = node_dict
        
        # Store the model if this is a leaf node (has a model prediction)
        if hasattr(node, 'model') and node.model is not None:
            self.node_models[node.node_id] = node.model
            
        if node.parent_id is not None:
            self.children_map.setdefault(node.parent_id, []).append(node_dict)

    # ------------------------------ Fit / Predict ---------------------------- #

    def fit(self, X, y, quantile: float):
        """
        Fit a single-quantile tree.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape (n_samples, n_features)
            Training features.
        y : array-like or pandas.Series, shape (n_samples,)
            Target variable.
        quantile : float
            Desired quantile in (0, 1).

        Returns
        -------
        self : QuantileRegressionTree
            Fitted estimator.
        """
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be between 0 and 1")

        # Reset RNG for reproducible sub-sampling.
        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)

        # Convert inputs to NumPy, track feature names if a DataFrame is given.
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.asarray(X)

        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.asarray(y)

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Use float views without copying when possible.
        self._X = X.astype(float, copy=False)
        self._y = y.astype(float, copy=False)
        self.quantile = float(quantile)

        all_idx = np.arange(self._X.shape[0])
        self._build_tree(all_idx, self.quantile)
        return self

    def fit_interval(
        self,
        X,
        y,
        lower_quantile: float,
        upper_quantile: float,
    ):
        """
        Fit two trees at `lower_quantile` and `upper_quantile` for intervals.

        Returns
        -------
        self : QuantileRegressionTree
            Estimator holding two sub-trees for predict_interval().
        """
        if not (0.0 < lower_quantile < upper_quantile < 1.0):
            raise ValueError("Quantiles must satisfy 0 < lower < upper < 1")

        # Helper to clone configuration.
        def new_tree() -> "NodeQuantileRegressionTree":
            return NodeQuantileRegressionTree(
                split_criterion=self.split_criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                feature_names=self.feature_names,
                random_state=self.random_state,
                random_features=self.random_features,
                random_thresholds=self.random_thresholds,
                max_threshold_candidates=self.max_threshold_candidates,
            )

        self.lower_tree = new_tree().fit(X, y, lower_quantile)
        self.upper_tree = new_tree().fit(X, y, upper_quantile)
        return self

    def _predict_one(self, row: pd.Series | np.ndarray) -> float:
        """
        Route a single sample down the stored split structure to get a prediction from the leaf model.
        """
        cur = self.tree_nodes[0]
        while True:
            children = self.children_map.get(cur["node_id"], [])
            if not children:
                # This is a leaf node - use the stored model for prediction
                node_id = cur["node_id"]
                if node_id in self.node_models:
                    model = self.node_models[node_id]
                    # Prepare the input as a 2D array for sklearn
                    if isinstance(row, pd.Series):
                        X_pred = row[self.feature_names].values.reshape(1, -1)
                    else:
                        X_pred = np.array(row).reshape(1, -1)
                    return float(model.predict(X_pred)[0])
                else:
                    return float(np.nan)

            feat = children[0]["feature_name"]
            thr = children[0]["numeric_threshold"]

            # Fetch feature value from either a Series or a raw array.
            if isinstance(row, pd.Series):
                v = float(row[feat])
            else:
                v = float(row[self.feature_names.index(feat)])

            # Choose the child whose condition is satisfied.
            next_node = None
            for ch in children:
                if ch["condition"] == "<" and v < thr:
                    next_node = ch
                    break
                if ch["condition"] == ">=" and v >= thr:
                    next_node = ch
                    break

            if next_node is None:
                # Fallback: should not happen in a well-formed tree
                return float(np.nan)

            cur = next_node

    def predict(self, X) -> np.ndarray:
        """
        Predict the fitted quantile for each row in X.
        """
        if not self.tree_nodes:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' first.")

        if isinstance(X, pd.DataFrame):
            Xdf = X
        else:
            Xdf = pd.DataFrame(X, columns=self.feature_names)

        preds = [self._predict_one(row) for _, row in Xdf.iterrows()]
        return np.asarray(preds, dtype=float)

    def predict_interval(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict lower and upper quantiles using sub-trees trained by
        `fit_interval()`.

        Returns
        -------
        (lower, upper) : tuple of np.ndarray
            Element-wise interval bounds.
        """
        if self.lower_tree is None or self.upper_tree is None:
            raise ValueError(
                "No interval trees trained. Call fit_interval() first.")
        return self.lower_tree.predict(X), self.upper_tree.predict(X)

    def get_tree_structure(self) -> pd.DataFrame:
        """
        Export a flat view of the tree for inspection or debugging.

        Returns
        -------
        pandas.DataFrame
            Columns: Node ID, Parent ID, Feature Name, Condition,
            Numeric Threshold, Prediction.
        """
        if not self.tree_nodes:
            raise ValueError("Model has not been fitted yet.")

        rows: List[List[object]] = []
        for node_id in sorted(self.tree_nodes.keys()):
            n = self.tree_nodes[node_id]
            rows.append(
                [
                    n["node_id"],
                    n["parent_id"],
                    n["feature_name"],
                    n["condition"],
                    n["numeric_threshold"],
                ]
            )

        return pd.DataFrame(
            rows,
            columns=[
                "Node ID",
                "Parent ID",
                "Feature Name",
                "Condition",
                "Numeric Threshold",
            ],
        )