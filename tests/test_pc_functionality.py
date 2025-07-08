"""Test script for PC functionality in optimalgiv"""

import pandas as pd
import numpy as np
import os
os.environ['PYTHON_JULIACALL_THREADS'] = '1'
import optimalgiv as og

# Create test data
np.random.seed(42)
n_entities = 20
n_time = 10
n_obs = n_entities * n_time

df = pd.DataFrame({
    'id': np.repeat(range(1, n_entities + 1), n_time),
    't': np.tile(range(1, n_time + 1), n_entities),
    'S': np.repeat(1.0 / n_entities, n_obs),
    'q': np.random.randn(n_obs),
    'p': np.tile(np.random.randn(n_time), n_entities),
    'x1': np.random.randn(n_obs),
    'x2': np.random.randn(n_obs)
})
df.id = df.id.astype(str)
print("Testing PC functionality in optimalgiv...")
print(f"Data shape: {df.shape}")

# Test 1: Basic PC extraction with default options
print("\nTest 1: Basic PC extraction (2 components)")
try:
    model1 = og.giv(
        df, 
        formula="q + id & endog(p) ~ x1 + pc(2)",
        id="id",
        t="t", 
        weight="S",
        algorithm="iv",
        guess=np.ones(n_entities) * 0.5,
        save_df=True,
        quiet=True
    )
    print(f"✓ Model converged: {model1.converged}")
    print(f"✓ Number of PCs extracted: {model1.n_pcs}")
    if model1.pc_factors is not None:
        print(f"✓ PC factors shape: {model1.pc_factors.shape}")
    if model1.pc_loadings is not None:
        print(f"✓ PC loadings shape: {model1.pc_loadings.shape}")
    
    # Check if PC columns are in saved dataframe
    if model1.df is not None:
        pc_cols = [col for col in model1.df.columns if col.startswith('pc_')]
        print(f"✓ PC columns in dataframe: {pc_cols}")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")

# Test 2: PC extraction with custom pca_option
print("\nTest 2: PC extraction with custom pca_option")
try:
    model2 = og.giv(
        df, 
        formula="q + endog(p) ~ x1 + x2 + pc(3)",
        id="id",
        t="t", 
        weight="S",
        algorithm="iv",
        guess=0.5,
        pca_option={
            "impute_method": "zero",
            "demean": True,
            "maxiter": 50,
            "algorithm": "DeflatedHeteroPCA",
            "algorithm_options": {
                "t_block": 5,
                "condition_number_threshold": 3.0
            }
        },
        save_df=True,
        quiet=True
    )
    print(f"✓ Model converged: {model2.converged}")
    print(f"✓ Number of PCs extracted: {model2.n_pcs}")
    if model2.pc_factors is not None:
        print(f"✓ PC factors shape: {model2.pc_factors.shape}")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")

# Test 3: No PC extraction (control case)
print("\nTest 3: No PC extraction (control)")
try:
    model3 = og.giv(
        df, 
        formula="q + endog(p) ~ x1 + x2",
        id="id",
        t="t", 
        weight="S",
        algorithm="iv",
        guess=0.5,
        quiet=True
    )
    print(f"✓ Model converged: {model3.converged}")
    print(f"✓ Number of PCs: {model3.n_pcs} (should be 0)")
    print(f"✓ PC factors: {model3.pc_factors} (should be None)")
    print(f"✓ PC loadings: {model3.pc_loadings} (should be None)")
except Exception as e:
    print(f"✗ Test 3 failed: {e}")

print("\nAll PC functionality tests completed!")