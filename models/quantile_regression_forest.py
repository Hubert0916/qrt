import numpy as np
from collections import defaultdict
from models.quantile_regression_tree import QuantileRegressionTree

class QuantileRegressionForest:
    """
    Quantile Regression Forest wrapper，
    利用已存在的 QuantileRegressionTree，
    透過將同一訓練集丟給每棵樹，並記錄每棵樹每個葉節點對應的 y 分布，
    來做 conditional quantile estimate。
    """
    def __init__(self,
                 n_estimators=100,
                 quantile=0.5,
                 max_depth=5,
                 min_samples_leaf=1,
                 bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.quantile    = quantile
        self.max_depth   = max_depth
        self.min_samples_leaf    = min_samples_leaf
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
        
        if hasattr(X, 'shape'):
            n = X.shape[0]
        else:
            n = len(X)
            
        if hasattr(X, 'columns'):
            # pandas DataFrame
            feature_names = list(X.columns)
            X = X.values  
        else:
            X = np.array(X)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        y = np.array(y)  
        
        self._fallback_quantile = np.quantile(y, self.quantile)

        # 🌲 建立 n_estimators 棵量化回歸樹
        for i in range(self.n_estimators):
            # 🎲 Bootstrap 抽樣：從原始資料中有放回地抽取樣本
            # 每棵樹使用不同的資料子集以增加多樣性
            idx = rng.choice(n, size=n, replace=self.bootstrap)
            Xs, ys = X[idx], y[idx]  # 第 i 棵樹的訓練資料

            # 🌱 建立第 i 棵量化回歸樹
            tree = QuantileRegressionTree(
                split_criterion='r2',    # 使用 R² 評估分割品質
                max_depth=self.max_depth,  # 樹的最大深度限制
                min_samples_leaf=self.min_samples_leaf,    # 葉節點最小樣本數
                # 確保每棵樹有不同的隨機種子以增加多樣性
                random_state=(None if self.random_state is None else self.random_state + i)
            )
            # 訓練樹，quantile 參數用於分割準則計算
            tree.fit(Xs, ys, quantile=self.quantile, feature_names=feature_names)
            self.trees.append(tree)  # 將訓練好的樹加入森林

            # 📊 建立葉節點 ID 到 y 值的映射
            # 這是 QRF 的核心：記錄每個葉節點包含的 y 值分布
            leaf_ids = self._apply_tree(tree, Xs)  # 獲取每個訓練樣本對應的葉節點 ID
            mapping = defaultdict(list)  # 建立映射：leaf_id -> [y 值列表]
            
            # 將每個 y 值分配到對應的葉節點
            for leaf_id, yi in zip(leaf_ids, ys):
                mapping[leaf_id].append(yi)  # 記錄該葉節點包含的 y 值
            
            # 儲存第 i 棵樹的葉節點映射，供預測時使用
            self.leaf_values.append(mapping)

        return self

    def predict_quantile(self, X, alpha):
        """
        預測指定分位數 alpha 的值
        
        核心演算法：
        1. 對每個測試樣本，將其通過所有樹找到對應的葉節點
        2. 從每個葉節點收集訓練時的 y 值
        3. 聚合所有 y 值後計算 alpha 分位數
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            測試樣本特徵
        alpha : float, range (0, 1)
            目標分位數，例如 0.5 表示中位數
            
        Returns
        -------
        predictions : ndarray, shape (n_samples,)
            每個樣本的 alpha 分位數預測值
        """
        # 🔄 統一輸入格式：確保 X 是 numpy array
        if hasattr(X, 'values'):
            # pandas DataFrame → numpy array
            X_arr = X.values
        else:
            X_arr = np.array(X)
            
        preds = []  # 儲存每個樣本的預測結果
        
        # 🎯 對每個測試樣本進行預測
        for x in X_arr:
            agg = []  # 聚合來自所有樹的 y 值
            
            # 🌳 遍歷森林中的每棵樹
            for tree, mapping in zip(self.trees, self.leaf_values):
                try:
                    # 找到該樣本在此樹中對應的葉節點 ID
                    leaf_id = self._apply_tree(tree, [x])[0]
                    
                    # 從預先計算的映射中取得該葉節點的 y 值列表
                    leaf_values = mapping.get(leaf_id, [])
                    
                    # 將這些 y 值加入聚合列表
                    agg.extend(leaf_values)
                except (IndexError, KeyError, ValueError):
                    # 如果某棵樹處理失敗（例如資料格式問題），跳過該樹
                    # 這提供了容錯機制，即使部分樹失效也能繼續預測
                    continue
            
            # 📊 計算聚合後的分位數
            if len(agg) > 0:
                # 有足夠資料時，計算 alpha 分位數
                preds.append(np.quantile(agg, alpha))
            else:
                # 🚨 容錯機制：如果沒有有效資料，使用備用值
                if hasattr(self, '_fallback_quantile'):
                    # 使用訓練時計算的備用分位數
                    preds.append(self._fallback_quantile)
                else:
                    # 最後的備用方案
                    preds.append(0.0)
                    
        return np.array(preds)  # 返回 numpy array 格式的預測結果

    def predict(self, X):
        """
        便利方法：使用初始化時設定的預設分位數進行預測
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            測試樣本特徵
            
        Returns
        -------
        predictions : ndarray, shape (n_samples,)
            使用預設分位數的預測值
        """
        return self.predict_quantile(X, self.quantile)

    def predict_interval(self, X, lower_q, upper_q):
        """
        便利方法：預測指定的預測區間
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            測試樣本特徵
        lower_q : float, range (0, 1)
            區間下界分位數，例如 0.05 表示 5% 分位數
        upper_q : float, range (0, 1)
            區間上界分位數，例如 0.95 表示 95% 分位數
            
        Returns
        -------
        lower : ndarray, shape (n_samples,)
            預測區間的下界值
        upper : ndarray, shape (n_samples,)
            預測區間的上界值
            
        Examples
        --------
        >>> # 預測 90% 預測區間 (5%-95%)
        >>> lower, upper = qrf.predict_interval(X_test, 0.05, 0.95)
        >>> # 預測 50% 預測區間 (25%-75%)
        >>> lower, upper = qrf.predict_interval(X_test, 0.25, 0.75)
        """
        lower = self.predict_quantile(X, lower_q)  # 計算下界
        upper = self.predict_quantile(X, upper_q)  # 計算上界
        return lower, upper
