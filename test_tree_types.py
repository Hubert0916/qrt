#!/usr/bin/env python3
"""
Test script to verify different tree types produce different results.
"""

import numpy as np
import pandas as pd
from models.quantile_regression_forest import QuantileRegressionForest
from models.quantile_regression_tree import QuantileRegressionTree
from models.qunatile_regression_model_leaf_tree import LeafQuantileRegressionTree

def test_different_tree_types():
    """Test that different tree types produce different predictions."""
    
    print("🧪 Testing Different Tree Types in Forest")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 
         0.2 * np.sin(X[:, 3] * 3) + 
         0.1 * np.random.randn(n_samples))
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X_df[:train_size], X_df[train_size:]
    y_train, y_test = y_series[:train_size], y_series[train_size:]
    
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {n_features}")
    
    # Test parameters
    forest_params = {
        'n_estimators': 10,
        'max_depth': 3,
        'min_samples_leaf': 5,
        'quantile': 0.5,
        'random_state': 42,
        'max_features': None,  # Use all features
        'bootstrap': True
    }
    
    results = {}
    
    # Test 1: Standard QuantileRegressionTree
    print(f"\n--- Testing QuantileRegressionTree (Sample-based) ---")
    forest_standard = QuantileRegressionForest(
        tree_cls=QuantileRegressionTree,
        **forest_params
    )
    forest_standard.fit(X_train, y_train)
    preds_standard = forest_standard.predict(X_test)
    mae_standard = np.mean(np.abs(preds_standard - y_test))
    
    results['standard'] = {
        'predictions': preds_standard,
        'mae': mae_standard,
        'tree_type': 'QuantileRegressionTree'
    }
    
    print(f"MAE: {mae_standard:.4f}")
    print(f"Sample predictions: {preds_standard[:5]}")
    
    # Test 2: LeafQuantileRegressionTree  
    print(f"\n--- Testing LeafQuantileRegressionTree (Model-based) ---")
    forest_model = QuantileRegressionForest(
        tree_cls=LeafQuantileRegressionTree,
        **forest_params
    )
    forest_model.fit(X_train, y_train)
    preds_model = forest_model.predict(X_test)
    mae_model = np.mean(np.abs(preds_model - y_test))
    
    results['model'] = {
        'predictions': preds_model,
        'mae': mae_model,
        'tree_type': 'LeafQuantileRegressionTree'
    }
    
    print(f"MAE: {mae_model:.4f}")
    print(f"Sample predictions: {preds_model[:5]}")
    
    # Compare results
    print(f"\n" + "=" * 50)
    print("📊 COMPARISON RESULTS")
    print("=" * 50)
    
    # Check if predictions are different
    pred_diff = np.abs(preds_standard - preds_model)
    max_diff = np.max(pred_diff)
    mean_diff = np.mean(pred_diff)
    
    print(f"Prediction Differences:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Standard deviation of differences: {np.std(pred_diff):.6f}")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  Standard Tree MAE: {mae_standard:.4f}")
    print(f"  Model-based Tree MAE: {mae_model:.4f}")
    print(f"  MAE Difference: {abs(mae_standard - mae_model):.4f}")
    
    # Detailed comparison
    print(f"\nDetailed Prediction Comparison (first 10 samples):")
    print(f"{'Index':<5} {'Actual':<8} {'Standard':<10} {'Model':<10} {'Diff':<8}")
    print("-" * 45)
    
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        std_pred = preds_standard[i]
        mod_pred = preds_model[i]
        diff = abs(std_pred - mod_pred)
        print(f"{i:<5} {actual:<8.3f} {std_pred:<10.3f} {mod_pred:<10.3f} {diff:<8.3f}")
    
    # Statistical test
    are_different = mean_diff > 1e-6  # Threshold for "significantly different"
    
    print(f"\n🎯 CONCLUSION:")
    if are_different:
        print(f"✅ SUCCESS: Different tree types produce different predictions!")
        print(f"   Mean difference: {mean_diff:.6f} > threshold (1e-6)")
    else:
        print(f"❌ ISSUE: Tree types produce similar predictions")
        print(f"   Mean difference: {mean_diff:.6f} ≤ threshold (1e-6)")
        print(f"   This suggests the implementation may not be working correctly.")
    
    return results, are_different

def test_tree_internal_differences():
    """Test that individual trees are actually different."""
    
    print(f"\n" + "=" * 50) 
    print("🔍 TREE INTERNAL STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(50)
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    y_series = pd.Series(y)
    
    # Create forests
    forest_std = QuantileRegressionForest(
        tree_cls=QuantileRegressionTree,
        n_estimators=3,
        max_depth=2,
        random_state=42
    )
    
    forest_model = QuantileRegressionForest(
        tree_cls=LeafQuantileRegressionTree, 
        n_estimators=3,
        max_depth=2,
        random_state=42
    )
    
    forest_std.fit(X_df, y_series)
    forest_model.fit(X_df, y_series)
    
    print(f"Standard Forest Trees: {len(forest_std.trees_)}")
    print(f"Model Forest Trees: {len(forest_model.trees_)}")
    
    # Analyze first tree from each
    tree_std = forest_std.trees_[0]
    tree_model = forest_model.trees_[0]
    
    print(f"\nTree Structure Comparison:")
    print(f"Standard tree nodes: {len(tree_std.tree_nodes) if hasattr(tree_std, 'tree_nodes') else 'N/A'}")
    print(f"Model tree nodes: {len(tree_model.tree_nodes) if hasattr(tree_model, 'tree_nodes') else 'N/A'}")
    
    print(f"Standard tree type: {type(tree_std).__name__}")
    print(f"Model tree type: {type(tree_model).__name__}")
    
    # Test single predictions
    test_sample = X_df.iloc[0:1]
    
    try:
        pred_std = tree_std.predict(test_sample)
        pred_model = tree_model.predict(test_sample)
        
        print(f"\nSingle Tree Predictions for same sample:")
        print(f"Standard tree: {pred_std}")
        print(f"Model tree: {pred_model}")
        print(f"Difference: {abs(pred_std[0] - pred_model[0]) if len(pred_std) > 0 and len(pred_model) > 0 else 'N/A'}")
        
    except Exception as e:
        print(f"Error in single tree prediction: {e}")

if __name__ == "__main__":
    try:
        print("🚀 Starting Tree Type Comparison Test")
        
        results, success = test_different_tree_types()
        test_tree_internal_differences()
        
        if success:
            print(f"\n🎉 Test completed successfully! Tree types are working differently.")
        else:
            print(f"\n⚠️ Test shows potential issues with tree type differentiation.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()