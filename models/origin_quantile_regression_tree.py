"""
Base Quantile Regression Tree implementation with multiple splitting criteria.

This module implements a quantile regression tree that supports both R² 
maximization and quantile loss minimization as splitting criteria.
"""

import numpy as np
import pandas as pd
from collections import deque
from sklearn.metrics import r2_score


class QuantileRegressionTree:
    """
    A unified quantile regression tree implementation supporting two splitting criteria.

    This class implements a decision tree for quantile regression that can use either
    R² maximization or quantile loss minimization as the splitting criterion.

    Parameters
    ----------
    split_criterion : {'r2', 'loss'}, default='loss'
        The criterion used for splitting nodes:
        - 'r2': Maximize R² score using quantile predictions
        - 'loss': Minimize quantile loss (pinball loss)
    max_depth : int, default=5
        The maximum depth of the tree.
    min_samples_leaf : int, default=1
        The minimum number of samples required at a leaf node.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    tree_nodes : dict
        Dictionary containing tree node information.
    children_map : dict
        Mapping from parent nodes to their children.
    feature_names : list
        Names of the features used for training.
    quantile : float
        The quantile level used for training.
    """

    def __init__(self, split_criterion='loss', max_depth=5, min_samples_leaf=1, feature_names=None, random_state=None, random_features=False):
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.random_features = random_features

        # Tree structure (populated after training)
        self.tree_nodes = None
        self.children_map = None
        self.feature_names = feature_names
        self.quantile = None

        # Validate parameters
        if split_criterion not in ['r2', 'loss', 'mse']:
            raise ValueError("split_criterion must be 'r2', 'loss', or 'mse'")

    def _r2_split_evaluation(self, groups, quantile):
        """
        Evaluate split quality using R² score.

        Parameters
        ----------
        groups : list of lists
            Groups of data points after splitting.
        quantile : float
            Target quantile level.

        Returns
        -------
        float
            R² score for the split. Higher values indicate better splits.
        """
        all_labels = []
        all_preds = []

        for group in groups:
            if not group:
                continue
            group_labels = [float(row[-1]) for row in group]
            group_pred = np.quantile(group_labels, quantile)
            all_labels.extend(group_labels)
            all_preds.extend([group_pred] * len(group_labels))

        if len(all_labels) == 0:
            return -float('inf')

        return r2_score(all_labels, all_preds)

    def _quantile_loss_evaluation(self, groups, quantile):
        """
        Evaluate split quality using quantile loss (pinball loss).

        Parameters
        ----------
        groups : list of lists
            Groups of data points after splitting.
        quantile : float
            Target quantile level.

        Returns
        -------
        float
            Total quantile loss for the split. Lower values indicate better splits.
        """
        total_loss = 0.0

        for group in groups:
            if not group:
                continue
            labels = np.array([float(row[-1]) for row in group])
            q_val = np.quantile(labels, quantile)
            r = labels - q_val
            total_loss += np.sum(
                np.where(r <= 0,
                         quantile * np.abs(r),
                         (1 - quantile) * np.abs(r))
            )

        return total_loss

    def _evaluate_split(self, groups, quantile):
        """
        Evaluate split quality using the specified criterion.

        Parameters
        ----------
        groups : list of lists
            Groups of data points after splitting.
        quantile : float
            Target quantile level.

        Returns
        -------
        float
            Split quality score according to the chosen criterion.
        """
        if self.split_criterion == 'mse':
            total = 0.0
            for group in groups:
                if not group:
                    continue
                vals = np.array([row[-1] for row in group], dtype=float)
                total += vals.var() * len(vals)
            return total
        elif self.split_criterion == 'r2':
            return self._r2_split_evaluation(groups, quantile)
        elif self.split_criterion == 'loss':
            return self._quantile_loss_evaluation(groups, quantile)

    def _split_numeric(self, feature_idx, dataset, quantile):
        """
        Find the best binary split for a numeric feature.

        Parameters
        ----------
        feature_idx : int
            Index of the feature to split on.
        dataset : list of lists
            Dataset containing feature values and target values.
        quantile : float
            Target quantile level.

        Returns
        -------
        tuple
            Best threshold and corresponding groups, or (None, None) if no split found.
        """
        values = sorted({float(row[feature_idx]) for row in dataset})

        if len(values) < 2:
            return None, None

        if self.split_criterion == 'r2':
            best_thr, best_groups, best_score = None, None, -float('inf')
            def is_better(new, best): return new > best
        else:  # loss
            best_thr, best_groups, best_score = None, None, float('inf')
            def is_better(new, best): return new < best

        # Try thresholds between adjacent unique values
        for v1, v2 in zip(values[:-1], values[1:]):
            thr = (v1 + v2) / 2.0

            left = [r for r in dataset if float(r[feature_idx]) < thr]
            right = [r for r in dataset if float(r[feature_idx]) >= thr]

            # Skip empty groups
            if not left or not right:
                continue

            current_score = self._evaluate_split([left, right], quantile)

            if is_better(current_score, best_score):
                best_score, best_thr, best_groups = current_score, thr, {
                    '<': left, '>=': right}

        return best_thr, best_groups

    def _get_best_split(self, dataset, quantile):
        """
        Find the best feature and threshold for splitting a dataset.

        Parameters
        ----------
        dataset : list of lists
            Dataset containing feature values and target values.
        quantile : float
            Target quantile level.

        Returns
        -------
        dict
            Dictionary containing the best feature index, groups, and threshold.
        """
        if self.split_criterion == 'r2':
            best_score = -float('inf')
            def is_better(new, best): return new > best
        elif self.split_criterion in ['loss', 'mse']:
            best_score = float('inf')
            def is_better(new, best): return new < best
        else:
            raise ValueError(f"Unknown split_criterion {self.split_criterion}")

        best_feat = None
        best_groups = None
        best_threshold = None

        n_features = len(dataset[0]) - 1
        if self.random_features:
            mtry = max(1, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, mtry, replace=False)
        else:
            feature_indices = range(n_features)

        for feature_idx in feature_indices:
            threshold, groups = self._split_numeric(
                feature_idx, dataset, quantile)
            if groups is None:
                continue
            current_score = self._evaluate_split(
                list(groups.values()), quantile)
            if is_better(current_score, best_score):
                best_score = current_score
                best_feat = feature_idx
                best_groups = groups
                best_threshold = threshold

        return {
            'feature': best_feat,
            'groups': best_groups,
            'threshold': best_threshold
        }

    def _terminal_value(self, group, quantile):
        """
        Calculate the prediction value for a leaf node.

        Parameters
        ----------
        group : list of lists
            Data points in the leaf node.
        quantile : float
            Target quantile level.

        Returns
        -------
        float
            Quantile value for the group.
        """
        labels = [float(row[-1]) for row in group]
        return np.quantile(labels, quantile)

    def _build_tree(self, dataset, feature_names, quantile):
        """
        Build the decision tree using breadth-first construction.

        Parameters
        ----------
        dataset : list of lists
            Training dataset containing feature values and target values.
        feature_names : list
            Names of the features.
        quantile : float
            Target quantile level.

        Returns
        -------
        list
            Tree data structure as a list of node information.
        """
        root_node = {
            'dataset': dataset,
            'node_id': 0,
            'parent_id': None,
            'depth': 1,
            'feature_from_parent': None,
            'condition_from_parent': None,
            'numeric_threshold': None,
        }

        queue = deque([root_node])
        node_counter = 1
        tree_data = []

        while queue:
            node = queue.popleft()
            data_here = node['dataset']
            depth = node['depth']

            # Find the best split for current node
            best = self._get_best_split(data_here, quantile)
            feat_idx = best['feature']
            groups = best['groups']
            threshold = best['threshold']

            # Check stopping conditions
            if feat_idx is None or depth >= self.max_depth:
                pred = self._terminal_value(data_here, quantile)
                tree_data.append([
                    node['node_id'],
                    node['parent_id'],
                    node['feature_from_parent'],
                    node['condition_from_parent'],
                    node['numeric_threshold'],
                    pred
                ])
                continue

            # Add internal node
            tree_data.append([
                node['node_id'],
                node['parent_id'],
                node['feature_from_parent'],
                node['condition_from_parent'],
                node['numeric_threshold'],
                None
            ])

            feat_name = feature_names[feat_idx]

            # Process child nodes
            for condition, group in groups.items():
                if len(group) <= self.min_samples_leaf:
                    # Create leaf node directly
                    pred = self._terminal_value(group, quantile)
                    tree_data.append([
                        node_counter,
                        node['node_id'],
                        feat_name,
                        condition,
                        threshold,
                        pred
                    ])
                    node_counter += 1
                else:
                    # Create child node and add to queue for further splitting
                    child_node = {
                        'dataset': group,
                        'node_id': node_counter,
                        'parent_id': node['node_id'],
                        'depth': depth + 1,
                        'feature_from_parent': feat_name,
                        'condition_from_parent': condition,
                        'numeric_threshold': threshold,
                    }
                    node_counter += 1
                    queue.append(child_node)

        return tree_data

    def _build_tree_dict(self, tree_list):
        """
        Convert the tree list to a dictionary structure.
        """
        tree_nodes = {}
        children_map = {}

        for row in tree_list:
            node_id, parent_id, feature_name, condition, numeric_threshold, pred = row
            node_dict = {
                'node_id': node_id,
                'parent_id': parent_id,
                'feature_name': feature_name,
                'condition': condition,
                'numeric_threshold': numeric_threshold,
                'prediction': pred
            }
            tree_nodes[node_id] = node_dict

            if parent_id is not None:
                children_map.setdefault(parent_id, []).append(node_dict)

        return tree_nodes, children_map

    def fit(self, X, y, quantile):
        """
        Fit the quantile regression tree to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        quantile : float
            Target quantile level (0 < quantile < 1).
        feature_names : list, optional
            Names of the features. If None, generic names are generated.

        Returns
        -------
        self : QuantileRegressionTree
            Returns self for method chaining.
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        feature_names = self.feature_names

        # Validate inputs
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

        # Handle input format conversion
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Combine features and target into dataset format
        dataset = np.column_stack([X, y]).tolist()

        # Store parameters
        self.feature_names = feature_names
        self.quantile = quantile

        # Build the tree
        tree_list = self._build_tree(dataset, feature_names, quantile)
        self.tree_nodes, self.children_map = self._build_tree_dict(tree_list)

        return self

    def _predict_sample(self, sample):
        """
        Predict the quantile value for a single sample.

        Parameters
        ----------
        sample : dict
            Sample data with feature names as keys.

        Returns
        -------
        float
            Predicted quantile value.
        """
        current_node = self.tree_nodes[0]

        while True:
            node_id = current_node['node_id']
            children = self.children_map.get(node_id, [])

            # If leaf node, return prediction
            if not children:
                return current_node['prediction']

            # Get splitting feature
            splitting_feature = children[0]['feature_name']
            sample_value = sample[splitting_feature]
            found_child = None

            # Convert to numeric value
            try:
                sample_value_numeric = float(sample_value)
            except:
                sample_value_numeric = None

            # Find the appropriate child node
            for child in children:
                if child['condition'] == '<':
                    if sample_value_numeric is not None and sample_value_numeric < child['numeric_threshold']:
                        found_child = child
                        break
                elif child['condition'] == '>=':
                    if sample_value_numeric is not None and sample_value_numeric >= child['numeric_threshold']:
                        found_child = child
                        break

            # If no matching child found, use average of available predictions
            if found_child is None:
                candidate_preds = [child['prediction']
                                   for child in children if child['prediction'] is not None]
                if candidate_preds:
                    return np.mean(candidate_preds)
                else:
                    return None

            current_node = found_child

            # If current node has prediction, return it directly
            if current_node['prediction'] is not None:
                return current_node['prediction']

    def predict(self, X):
        """
        Predict quantile values for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted quantile values.
        """
        # Check if model has been fitted
        if self.tree_nodes is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' first.")

        # Handle input format
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)

        predictions = []
        for _, row in X_df.iterrows():
            sample = row.to_dict()
            pred = self._predict_sample(sample)
            predictions.append(pred)

        return np.array(predictions, dtype=float)

    def predict_interval(self, X, lower_quantile, upper_quantile):
        """
        預測分位數區間

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.
        lower_quantile : float
            Lower quantile bound (e.g., 0.1 for 10th percentile).
        upper_quantile : float
            Upper quantile bound (e.g., 0.9 for 90th percentile).

        Returns:
        --------
        lower_predictions : array of shape (n_samples,)
            Lower quantile predictions.
        upper_predictions : array of shape (n_samples,)
            Upper quantile predictions.
        """
        # Create two trees: lower and upper bounds
        lower_tree = QuantileRegressionTree(
            split_criterion=self.split_criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

        upper_tree = QuantileRegressionTree(
            split_criterion=self.split_criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

        # Need training data to build interval tree
        # Need to save training data or redesign
        raise NotImplementedError(
            "predict_interval requires access to training data. Use fit_interval method instead.")

    def get_tree_structure(self):
        """
        Get the tree structure as a pandas DataFrame.

        Returns
        -------
        tree_df : pandas.DataFrame
            DataFrame representation of the tree structure.
        """
        if self.tree_nodes is None:
            raise ValueError("Model has not been fitted yet.")

        tree_data = []
        for node_id in sorted(self.tree_nodes.keys()):
            node = self.tree_nodes[node_id]
            tree_data.append([
                node['node_id'],
                node['parent_id'],
                node['feature_name'],
                node['condition'],
                node['numeric_threshold'],
                node['prediction']
            ])

        columns = ["Node ID", "Parent ID", "Feature Name",
                   "Condition", "Numeric Threshold", "Prediction"]
        return pd.DataFrame(tree_data, columns=columns)


def predict_quantile_interval(X_train, y_train, X_test, lower_quantile, upper_quantile,
                              split_criterion='loss', max_depth=5, min_samples_leaf=1,
                              feature_names=None, random_state=None):
    """
    Convenience function for quantile interval prediction.

    This function trains two separate quantile regression trees for the lower
    and upper bounds of the prediction interval.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target values.
    X_test : array-like
        Test feature matrix.
    lower_quantile : float
        Lower quantile bound (e.g., 0.1 for 10th percentile).
    upper_quantile : float
        Upper quantile bound (e.g., 0.9 for 90th percentile).
    split_criterion : {'r2', 'loss'}, default='loss'
        Splitting criterion for the trees.
    max_depth : int, default=5
        Maximum depth of the trees.
    min_samples_leaf : int, default=1
        Minimum samples per leaf node.
    feature_names : list, optional
        Names of the features.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Tuple containing (lower_predictions, upper_predictions).
    """
    # Train lower bound tree
    lower_tree = QuantileRegressionTree(
        split_criterion=split_criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    lower_tree.fit(X_train, y_train, lower_quantile, feature_names)

    # Train upper bound tree
    upper_tree = QuantileRegressionTree(
        split_criterion=split_criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    upper_tree.fit(X_train, y_train, upper_quantile, feature_names)

    # Make predictions
    lower_predictions = lower_tree.predict(X_test)
    upper_predictions = upper_tree.predict(X_test)

    return lower_predictions, upper_predictions
