#!/usr/bin/env python3
"""
Test the modified QuantileRegressionTree with model-based split evaluation.
"""

import numpy as np
import pandas as pd
from models.qunatile_regression_model_node_tree import QuantileRegressionTree

def test_modified_split_evaluation():
    """Test the modified split evaluation methods."""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    # Create nonlinear relationships
    y = (X[:, 0] ** 2 + 0.5 * X[:, 1] * X[:, 2] + 
         0.2 * np.sin(X[:, 0] * 3) + 
         0.1 * np.random.randn(n_samples))
    
    X_df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    y_series = pd.Series(y)
    
    print("=== Testing Modified Split Evaluation ===")
    print(f"Data shape: {X_df.shape}")
    
    # Test different split criteria
    criteria = ['loss', 'r2', 'mse']
    
    for criterion in criteria:
        print(f"\n--- Testing {criterion} criterion ---")
        
        try:
            tree = QuantileRegressionTree(
                split_criterion=criterion,
                max_depth=3,
                min_samples_leaf=8,
                random_state=42
            )
            
            print(f"Fitting tree with {criterion} criterion...")
            tree.fit(X_df, y_series, quantile=0.5)
            
            print(f"Tree nodes: {len(tree.tree_nodes)}")
            print(f"Leaf models: {len(tree.node_models)}")
            
            # Make predictions
            preds = tree.predict(X_df[:10])
            print(f"Sample predictions: {preds[:5]}")
            
            # Calculate MAE
            all_preds = tree.predict(X_df)
            mae = np.mean(np.abs(all_preds - y_series))
            print(f"Training MAE: {mae:.4f}")
            
            print(f"✅ {criterion} criterion works!")
            
        except Exception as e:
            print(f"❌ Error with {criterion} criterion: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Testing Complete ===")

def test_split_evaluation_details():
    """Test the split evaluation methods directly."""
    
    print("\n=== Testing Split Evaluation Methods Directly ===")
    
    # Create simple test data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(20)
    
    tree = QuantileRegressionTree()
    tree._X = X.astype(float)
    tree._y = y.astype(float)
    tree.feature_names = ['f1', 'f2']
    
    # Split data manually
    mid = len(y) // 2
    X_left, y_left = X[:mid], y[:mid]
    X_right, y_right = X[mid:], y[mid:]
    
    print(f"Left subset: {X_left.shape}, Right subset: {X_right.shape}")
    
    # Test loss evaluation
    try:
        loss_score = tree._evaluate_split_loss(X_left, y_left, X_right, y_right, 0.5)
        print(f"Loss evaluation score: {loss_score:.4f}")
    except Exception as e:
        print(f"Loss evaluation error: {e}")
    
    # Test R² evaluation  
    try:
        r2_score = tree._evaluate_split_r2(X_left, y_left, X_right, y_right, 0.5)
        print(f"R² evaluation score: {r2_score:.4f}")
    except Exception as e:
        print(f"R² evaluation error: {e}")
    
    # Test model fitting
    try:
        model = tree._fit_quantile_model(X_left, y_left, 0.5)
        if model is not None:
            pred = model.predict(X_left[:1])
            print(f"Model fitting successful, sample prediction: {pred[0]:.4f}")
        else:
            print("Model fitting returned None")
    except Exception as e:
        print(f"Model fitting error: {e}")

if __name__ == "__main__":
    test_modified_split_evaluation()
    test_split_evaluation_details()