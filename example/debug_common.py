import os
import random
import csv

# --- CRITICAL FIX FOR M3 MAC MUTEX DEADLOCK ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
# ----------------------------------------------

import numpy as np
import pandas as pd

# --- 0. Seeding & Determinism ---
def set_global_seeds(seed=42):
    """Forces deterministic behavior. Call this before EVERY major operation."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Force TF deterministic ops (optional, but good for parity checks)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- 1. Data Generation ---
def make_synthetic_data(input_names, output_names, n_cases=2, ndays=105):
    # We rely on set_global_seeds being called right before this
    dates = pd.date_range("2021-01-01", periods=ndays * 2, freq="D")
    dfs = []
    for case_id in range(1, n_cases + 1):
        data = {"datetime": dates, "case": case_id}
        for i, feat in enumerate(input_names):
            # Deterministic signal
            data[feat] = np.sin(np.linspace(0, (i+1)*np.pi, len(dates))) + case_id * 0.5
            # Add Noise (Dependent on RNG state)
            data[feat] += np.random.normal(0, 0.1, len(dates))
            
        for out in output_names:
            val = np.zeros(len(dates))
            for feat in input_names:
                val += data[feat]
            data[out] = val * 0.1 + np.random.normal(0, 0.05, len(dates))
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs).reset_index(drop=True)

# --- 2. History Dumper ---
def dump_history_to_csv(history_obj, output_path):
    """Converts Keras history object to CSV for easy diffing."""
    if not hasattr(history_obj, 'history'):
        print(f"  ⚠️ No history found to log for {output_path}")
        return

    hist = history_obj.history
    if not hist:
        print(f"  ⚠️ History dict is empty for {output_path}")
        return

    keys = sorted(hist.keys())
    epochs = range(1, len(hist[keys[0]]) + 1)
    
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + keys)
            for i, ep in enumerate(epochs):
                # Format to 8 decimal places to catch tiny divergences
                row = [ep] + [f"{hist[k][i]:.8f}" for k in keys]
                writer.writerow(row)
        print(f"  📄 Logged metrics to {output_path}")
    except Exception as e:
        print(f"  ❌ Failed to write CSV {output_path}: {e}")

# --- 2b. Weight Inspection Tools ---
def print_head_stats(model, head_name):
    """Prints mean/std of a specific layer's weights to confirm initialization."""
    try:
        layer = model.get_layer(head_name)
        weights = layer.get_weights()
        if not weights:
            print(f"  ⚠️  Layer {head_name} has no weights!")
            return None, None
        w, b = weights
        print(f"  📊 {head_name}: W_mean={w.mean():.5f}, W_std={w.std():.5f}, B_mean={b.mean():.5f}")
        return w, b
    except ValueError:
        print(f"  ❌ Layer {head_name} not found in model.")
        return None, None

def assert_heads_match(model, head1, head2):
    """Verifies two heads have IDENTICAL weights (for transfer parity)."""
    print(f"  🔍 Checking parity: {head1} == {head2}...")
    w1, b1 = print_head_stats(model, head1)
    w2, b2 = print_head_stats(model, head2)
    
    if w1 is None or w2 is None:
        return

    if np.allclose(w1, w2) and np.allclose(b1, b2):
        print(f"  ✅ SUCCESS: {head1} and {head2} are identical.")
    else:
        diff = np.abs(w1 - w2).mean()
        print(f"  ❌ FAILURE: Weights differ! Mean Diff: {diff:.6f}")
        print(" (This implies random initialization occurred instead of transfer)")

# --- 3. Constants ---
INPUT_NAMES = ["northern_flow", "exports", "sjr_flow", "cu_delta"]
OUTPUT_NAMES = {"x2": 100.0, "ec": 2000.0}
NDAYS = 10
EPOCHS = 3