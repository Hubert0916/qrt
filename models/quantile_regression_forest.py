import numpy as np
from collections import defaultdict
from models.quantile_regression_tree import QuantileRegressionTree

class QuantileRegressionForest:
    """
    Quantile Regression Forest wrapper，
    利用已存在的 QuantileRegressionTree (不改動其程式)，
    透過將同一訓練集丟給每棵樹，並記錄每棵樹每個葉節點對應的 y 分布，
    來做 conditional quantile estimate。
    """
    def __init__(self,
                 n_estimators=100,
                 quantile=0.5,
                 max_depth=5,
                 min_size=1,
                 bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.quantile    = quantile
        self.max_depth   = max_depth
        self.min_size    = min_size
        self.bootstrap   = bootstrap
        self.random_state= random_state

        self.trees       = []   # 存放 QuantileRegressionTree instance
        self.leaf_values = []   # 對應每棵樹：leaf_id -> [y1, y2, …] 映射

    def _apply_tree(self, tree, X):
        """
        根據 tree.tree_nodes 和 tree.children_map，
        模仿 apply()：把每筆 X 對應到葉節點的 node_id。
        """
        node_ids = []
        for x in X:
            node_id = 0
            # 走到葉節節點為止
            while node_id in tree.children_map:
                found_child = False
                for child in tree.children_map[node_id]:
                    feat = child['feature_name']
                    try:
                        idx = tree.feature_names.index(feat)
                    except ValueError:
                        # 如果特徵名稱不存在，跳過這個子節點
                        continue
                    
                    try:
                        val = float(x[idx])
                    except (ValueError, IndexError):
                        # 如果無法轉換或索引超出範圍，跳過
                        continue
                    
                    if child['condition'] == '<' and val < child['numeric_threshold']:
                        node_id = child['node_id']
                        found_child = True
                        break
                    elif child['condition'] == '>=' and val >= child['numeric_threshold']:
                        node_id = child['node_id']
                        found_child = True
                        break
                
                if not found_child:
                    # 如果條件都不符合或發生錯誤，選擇第一個有效的子節點
                    if tree.children_map[node_id]:
                        node_id = tree.children_map[node_id][0]['node_id']
                    else:
                        # 如果沒有子節點，停止遍歷
                        break
            node_ids.append(node_id)
        return node_ids

    def fit(self, X, y):
        """
        1. 建 n_estimators 棵 QuantileRegressionTree（用 paper 建議的切分）
        2. 建立每棵樹的 leaf_id -> [y 值清單] 映射
        """
        self.trees = []
        self.leaf_values = []
        rng = np.random.default_rng(self.random_state)
        
        # 統一處理輸入格式，確保 feature_names 一致性
        if hasattr(X, 'shape'):
            n = X.shape[0]
        else:
            n = len(X)
            
        # 確保 feature_names 一致性
        if hasattr(X, 'columns'):
            # pandas DataFrame
            feature_names = list(X.columns)
            X = X.values  # 轉換為 numpy array 以確保索引一致性
        else:
            # numpy array 或其他格式
            X = np.array(X)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        y = np.array(y)  # 確保 y 也是 numpy array
        
        # 計算 fallback quantile，用於預測時的錯誤處理
        self._fallback_quantile = np.quantile(y, self.quantile)

        for i in range(self.n_estimators):
            # bagging
            idx = rng.choice(n, size=n, replace=self.bootstrap)
            Xs, ys = X[idx], y[idx]

            # fit 一棵樹 (quantile 參數留給底層，但我們只用它的結構)
            tree = QuantileRegressionTree(
                split_criterion='mse',
                max_depth=self.max_depth,
                min_size=self.min_size,
                random_state=(None if self.random_state is None else self.random_state + i)
            )
            tree.fit(Xs, ys, quantile=self.quantile, feature_names=feature_names)
            self.trees.append(tree)

            # 把訓練資料各自分配到葉節點，收集 y
            leaf_ids = self._apply_tree(tree, Xs)
            mapping = defaultdict(list)
            for leaf_id, yi in zip(leaf_ids, ys):
                mapping[leaf_id].append(yi)
            self.leaf_values.append(mapping)

        return self

    def predict_quantile(self, X, alpha):
        """
        對每筆測試樣本，
        - 依序丟進每棵樹，找出它對應的 leaf_id
        - 從 pre-computed leaf_values 拿出所有 y，再 aggregated 去算 α 分位數
        """
        # 統一處理輸入格式
        if hasattr(X, 'values'):
            # pandas DataFrame
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        preds = []
        for x in X_arr:
            agg = []
            for tree, mapping in zip(self.trees, self.leaf_values):
                try:
                    leaf_id = self._apply_tree(tree, [x])[0]
                    leaf_values = mapping.get(leaf_id, [])
                    agg.extend(leaf_values)
                except (IndexError, KeyError, ValueError):
                    # 如果某棵樹出現問題，跳過該樹
                    continue
            
            # 確保有足夠的數據進行分位數計算
            if len(agg) > 0:
                preds.append(np.quantile(agg, alpha))
            else:
                # 如果沒有有效數據，使用訓練集目標值的分位數作為 fallback
                if hasattr(self, '_fallback_quantile'):
                    preds.append(self._fallback_quantile)
                else:
                    preds.append(0.0)  # 最後的 fallback
                    
        return np.array(preds)

    def predict(self, X):
        return self.predict_quantile(X, self.quantile)

    def predict_interval(self, X, lower_q, upper_q):
        lower = self.predict_quantile(X, lower_q)
        upper = self.predict_quantile(X, upper_q)
        return lower, upper
