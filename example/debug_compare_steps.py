
import pandas as pd
from sklearn.metrics import r2_score
import os

def compare_step_outputs(step_name, baseline_prefix, new_prefix):
    """
    Loads, merges, and compares the output CSVs for a given step from two different runs.
    """
    print(f"--- Comparing Step: {step_name} ---")
    
    baseline_path = f"{baseline_prefix}_{step_name}_xvalid.csv"
    new_path = f"{new_prefix}_{step_name}_xvalid.csv"

    # Check if files exist
    if not os.path.exists(baseline_path) or not os.path.exists(new_path):
        print(f"Skipping step {step_name}: one or both output files not found.")
        print(f"  - Searched for: {baseline_path}")
        print(f"  - Searched for: {new_path}")
        return

    # Load data
    df_baseline = pd.read_csv(baseline_path, parse_dates=['datetime'])
    df_new = pd.read_csv(new_path, parse_dates=['datetime'])

    # Merge on common keys
    df_merged = pd.merge(df_baseline, df_new, on=['datetime', 'case'], suffixes=('_base', '_new'))

    # --- Verification ---
    # 1. Check that the validation folds are identical
    try:
        assert (df_merged['fold_base'] == df_merged['fold_new']).all()
        print("✅ Fold assignment is identical.")
    except AssertionError:
        print("❌ ERROR: Fold assignments have diverged.")
        return

    # 2. Compare prediction columns
    pred_cols_base = [c for c in df_baseline.columns if '_pred' in c]
    
    if not pred_cols_base:
        print("No prediction columns found to compare.")
        return

    print("Comparing prediction columns:")
    for col_name in pred_cols_base:
        base_col = f"{col_name}_base"
        new_col = f"{col_name}_new"
        
        if base_col in df_merged and new_col in df_merged:
            r2 = r2_score(df_merged[base_col], df_merged[new_col])
            print(f"  - R² between '{base_col}' and '{new_col}': {r2:.6f}")
        else:
            print(f"  - Could not find pair for {col_name}")

    print("-" * (len(step_name) + 20))


if __name__ == "__main__":
    # --- Configuration ---
    # These prefixes should point to the Trial1 outputs from your MSTAGE and MSCEN runs
    baseline_prefix = 'example/output/dsm2_base_gru2_v2.1_MSTAGE_BSSN_Trial1'
    new_prefix = 'example/output/dsm2_base_gru2_MSCEN_v2.1_BSSN_Trial1'

    # --- Analysis ---
    # Compare Step 1: The initial base model
    compare_step_outputs('dsm2_base', baseline_prefix, new_prefix)

    # Compare Step 2: After the first fine-tuning/transfer step
    compare_step_outputs('dsm2.schism', baseline_prefix, new_prefix)

    # Compare Step 3: The final contrastive/multi-head step
    # Note: The output names might differ here. 
    # MSTAGE uses 'base.suisun', MSCEN uses 'base.multi'
    print("\n--- Comparing Final Step (MSTAGE vs. MSCEN) ---")
    mstage_final_path = f"{baseline_prefix}_base.suisun_xvalid.csv"
    mscen_final_path = f"{new_prefix}_base.multi_xvalid.csv"

    if os.path.exists(mstage_final_path) and os.path.exists(mscen_final_path):
        df_mstage = pd.read_csv(mstage_final_path, parse_dates=['datetime'])
        df_mscen = pd.read_csv(mscen_final_path, parse_dates=['datetime'])
        
        # In MSCEN, the columns are named 'out_base_unscaled_...'. We rename for comparison.
        mscen_rename_map = {
            'out_base_unscaled_emmaton_ec_obs': 'emmaton_ec_obs',
            'out_base_unscaled_emmaton_ec_pred': 'emmaton_ec_pred',
            'out_base_unscaled_jersey_point_ec_obs': 'jersey_point_ec_obs',
            'out_base_unscaled_jersey_point_ec_pred': 'jersey_point_ec_pred'
        }
        df_mscen.rename(columns=mscen_rename_map, inplace=True)

        # Merge and compare
        df_final_merged = pd.merge(df_mstage, df_mscen, on=['datetime', 'case'], suffixes=('_mstage', '_mscen'))
        
        pred_cols_mstage = [c for c in df_mstage.columns if '_pred' in c]
        print("Comparing prediction columns for 'base' output:")
        for col_name in pred_cols_mstage:
            mstage_col = f"{col_name}_mstage"
            mscen_col = f"{col_name}_mscen"
            if mstage_col in df_final_merged and mscen_col in df_final_merged:
                r2 = r2_score(df_final_merged[mstage_col], df_final_merged[mscen_col])
                print(f"  - R² between '{mstage_col}' and '{mscen_col}': {r2:.6f}")
    else:
        print("Skipping final step comparison: one or both output files not found.")
        print(f"  - Searched for: {mstage_final_path}")
        print(f"  - Searched for: {mscen_final_path}")
