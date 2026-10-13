import os
import yaml
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from casanntra.staged_learning import process_config

# Number of combination is multiplicative so define hyperparam space accordingly
HYPERPARAM_SPACE = {
    "contrast_weight": [0.5, 1.0, 2.0],  
    "freeze_bool": [True, False],       
    "arch_type": ["LSTM", "GRU"],        
    "init_lr": [0.003, 0.008],           
    "main_lr": [0.0004, 0.001],           
    "init_epochs": [10],
    "main_epochs": [35, 75, 100],
}

BASE_CONFIG_FILE = "transfer_config.yml"  
STEPS_TO_RUN = ["dsm2_base", "dsm2.schism", "base.suisun"]
MASTER_SUMMARY = "gridsearch_master_results.csv" # After gridsearch is done this will store all metrics


MODEL_NAMES = ["base.suisun", "base.suisun-secondary"]
STATIONS = [
    "cse","bdl","rsl","emm2","jer","sal","frk","bac",
    "oh4","x2","mal","god","gzl","vol"
]
OUTPUT_PREFIXES = {
    "base.suisun": "schism_base.suisun_gru2",
    "base.suisun-secondary": "schism_base.suisun_gru2",
}

def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "nse": np.nan, "pearson_r": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]

    mae = np.mean(np.abs(yt - yp))
    mse = np.mean((yt - yp)**2)
    rmse = np.sqrt(mse)

    denom = np.sum((yt - np.mean(yt))**2)
    if denom == 0:
        nse = np.nan
    else:
        nse = 1.0 - np.sum((yt - yp)**2)/denom

    corr = np.corrcoef(yt, yp)[0,1] if len(yt) > 1 else np.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "nse": nse,
        "pearson_r": corr,
    }

def load_and_merge(model_name, trial_suffix):
    """
    Merges the ref CSV + ANN CSV from your cross-validation pipeline.
    e.g. "schism_base.suisun_gru2_Trial1_xvalid_ref_out_unscaled.csv"
         "schism_base.suisun_gru2_Trial1_xvalid_0.csv"
    or the "secondary" versions. 
    """
    base_prefix = OUTPUT_PREFIXES[model_name]
    full_prefix = f"{base_prefix}_{trial_suffix}"

    if model_name == "base.suisun-secondary":
        ref_csv = f"{full_prefix}_xvalid_ref_out_secondary_unscaled.csv"
        ann_csv = f"{full_prefix}_xvalid_1.csv"
    else:
        ref_csv = f"{full_prefix}_xvalid_ref_out_unscaled.csv"
        ann_csv = f"{full_prefix}_xvalid_0.csv"

    print(f"[DEBUG load_and_merge] => Searching for REF: {ref_csv}")
    print(f"[DEBUG load_and_merge] => Searching for ANN: {ann_csv}")

    if not os.path.exists(ref_csv):
        raise FileNotFoundError(f"[load_and_merge] REF CSV not found => {ref_csv}")
    if not os.path.exists(ann_csv):
        raise FileNotFoundError(f"[load_and_merge] ANN CSV not found => {ann_csv}")

    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])

    print(f"[DEBUG load_and_merge] => df_ref.shape={df_ref.shape}, df_ann.shape={df_ann.shape}")

    merged = pd.merge(df_ref, df_ann, how="inner", on=["datetime","case"], suffixes=("","_pred"))
    print(f"[DEBUG load_and_merge] => after merge => merged.shape={merged.shape}")
    if not merged.empty:
        print("[DEBUG load_and_merge] => merged columns:", merged.columns.tolist())
        print(merged.head(5).to_string())
    return merged

def plot_timeseries_all_cases(df_merged, station, model_name, out_dir, n_cases=7):
    stcol = station
    stpred = station + "_pred"
    if stcol not in df_merged.columns or stpred not in df_merged.columns:
        print(f"[plot_timeseries_all_cases] Missing => {station}, skip {model_name}")
        return

    fig, axes = plt.subplots(n_cases,1,figsize=(8,2.5*n_cases),constrained_layout=True)
    if n_cases==1:
        axes=[axes]

    for i, ax in enumerate(axes):
        case_id = i+1
        subdf = df_merged[df_merged["case"]==case_id]
        if i==0:
            ax.set_title(f"[{model_name}] => Station={station}")

        ax.plot(subdf["datetime"], subdf[stcol],   color="0.1", label="Model")
        ax.plot(subdf["datetime"], subdf[stpred], label="ANN")
        ax.set_ylabel("Norm EC")
        ax.set_title(f"Case={case_id}, #rows={subdf.shape[0]}")
    axes[0].legend()

    fpath = os.path.join(out_dir, f"{model_name}_{station}_timeseries.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"[plot_timeseries_all_cases] => wrote plot => {fpath}")

def evaluate_and_plot(trial_dir, hyperparams, trial_suffix):
    """
    For each model_name, merges ref+ann, does station-level metrics, 
    writes them to "trial_evaluation_metrics.csv" in `trial_dir`,
    plus timeseries plots inside subfolders "<trial_dir>/<model_name>/".
    Returns summary stats (mean_nse_base, mean_nse_suisun, mean_nse_overall, mean_r2).
    """
    print(f"[DEBUG evaluate_and_plot] => trial_dir={trial_dir}, trial_suffix={trial_suffix}")
    rows=[]
    for model_name in MODEL_NAMES:
        print(f"[DEBUG evaluate_and_plot] => model_name={model_name}")
        try:
            df_merged = load_and_merge(model_name, trial_suffix)
        except FileNotFoundError as exc:
            print(f"[evaluate_and_plot] => skipping {model_name}, reason: {exc}")
            continue

        if df_merged.empty:
            print(f"[DEBUG evaluate_and_plot] => merged is empty => skipping {model_name}")
            continue

        model_out_dir = os.path.join(trial_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        print(f"[DEBUG evaluate_and_plot] => model_out_dir={model_out_dir}, #rows in df_merged={df_merged.shape[0]}")

        for station in STATIONS:
            if station not in df_merged.columns:
                print(f"[DEBUG evaluate_and_plot] => station '{station}' not in merged columns => skip")
                continue
            stpred = station+"_pred"
            if stpred not in df_merged.columns:
                print(f"[DEBUG evaluate_and_plot] => station pred '{stpred}' not in columns => skip")
                continue

            sub_n = df_merged[[station, stpred]].dropna().shape[0]
            print(f"[DEBUG evaluate_and_plot] => computing metrics for station={station}, #non-na rows={sub_n}")

            met = compute_metrics(df_merged[station], df_merged[stpred])
            row = {
                "model": model_name,
                "station": station,
                "mae": round(met["mae"],4),
                "rmse": round(met["rmse"],4),
                "nse": round(met["nse"],4),
                "pearson_r": round(met["pearson_r"],4),
            }
            for k,v in hyperparams.items():
                row[k] = v
            rows.append(row)
            plot_timeseries_all_cases(df_merged, station, model_name, model_out_dir, n_cases=7)

    if not rows:
        print("[DEBUG evaluate_and_plot] => no rows => returning None")
        return None

    df_trial = pd.DataFrame(rows)
    df_trial_csv = os.path.join(trial_dir,"trial_evaluation_metrics.csv")
    df_trial.to_csv(df_trial_csv, index=False)
    print(f"[evaluate_and_plot] => wrote station-level metrics => {df_trial_csv}, rowcount={df_trial.shape[0]}")

    df_base = df_trial[df_trial["model"]=="base.suisun"]
    df_suisun = df_trial[df_trial["model"]=="base.suisun-secondary"]
    mean_nse_base   = round(df_base["nse"].mean(),4) if len(df_base)>0 else np.nan
    mean_nse_suisun = round(df_suisun["nse"].mean(),4) if len(df_suisun)>0 else np.nan
    overall_nse     = round(df_trial["nse"].mean(),4)

    df_trial["r2"] = df_trial["pearson_r"]**2
    mean_r2 = round(df_trial["r2"].mean(),4)

    print(f"[DEBUG evaluate_and_plot] => mean_nse_base={mean_nse_base}, mean_nse_suisun={mean_nse_suisun}, overall_nse={overall_nse}, mean_r2={mean_r2}")
    return {
        "mean_nse_base": mean_nse_base,
        "mean_nse_suisun": mean_nse_suisun,
        "mean_nse_overall": overall_nse,
        "mean_r2": mean_r2
    }


def load_yml(path):
    print(f"[DEBUG load_yml] => loading config from {path}")
    with open(path,"r") as f:
        return yaml.safe_load(f)

def save_yml(obj, path):
    print(f"[DEBUG save_yml] => saving updated config to {path}")
    with open(path,"w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# GridSearch + Evaluate
def main():
    print(f"[DEBUG main] => Checking if old scoreboard {MASTER_SUMMARY} exists..")
    if os.path.exists(MASTER_SUMMARY):
        print(f"[DEBUG main] => removing old {MASTER_SUMMARY}")
        os.remove(MASTER_SUMMARY)

    print(f"[DEBUG main] => Loading base config file => {BASE_CONFIG_FILE}")
    base_cfg = load_yml(BASE_CONFIG_FILE)

    keys = list(HYPERPARAM_SPACE.keys())
    combos = list(itertools.product(*[HYPERPARAM_SPACE[k] for k in keys]))
    all_combos = [dict(zip(keys, c)) for c in combos]
    print(f"[DEBUG main] => total combos = {len(all_combos)}")
    print(f"[DEBUG main] => combos = {all_combos}")

    trial_counter=0
    for combo in all_combos:
        trial_counter+=1
        trial_name = f"Trial{trial_counter}"
        print(f"\n=== Starting {trial_name} with => {combo}")

        mod_cfg = copy.deepcopy(base_cfg)

        for i, step in enumerate(mod_cfg["steps"]):
            old_prefix = step["output_prefix"]
            step["output_prefix"] = f"{step['output_prefix']}_{trial_name}"
            print(f"[DEBUG] step[{i}] => old_prefix={old_prefix}, new_prefix={step['output_prefix']}")

            if step.get("save_model_fname") not in [None,"None"]:
                old_save = step["save_model_fname"]
                step["save_model_fname"] = f"{old_save}_{trial_name}"
                print(f"           => save_model_fname = {step['save_model_fname']}")
            if step.get("load_model_fname") not in [None,"None"]:
                old_load = step["load_model_fname"]
                step["load_model_fname"] = f"{old_load}_{trial_name}"
                print(f"           => load_model_fname = {step['load_model_fname']}")

            step["init_train_rate"] = combo["init_lr"]
            step["main_train_rate"] = combo["main_lr"]
            step["init_epochs"] = combo["init_epochs"]
            step["main_epochs"] = combo["main_epochs"]
            print(f"=> init_lr={step['init_train_rate']}, main_lr={step['main_train_rate']}")
            print(f"=> init_epochs={step['init_epochs']}, main_epochs={step['main_epochs']}")

            bargs = step.get("builder_args", {})
            bargs["contrast_weight"] = combo["contrast_weight"]
            bargs["freeze_bool"] = combo["freeze_bool"]
            bargs["arch_type"] = combo["arch_type"]
            step["builder_args"]=bargs
            print(f"=> builder_args={bargs}")

        tmp_config_file = f"tmp_{trial_name}.yml"
        save_yml(mod_cfg, tmp_config_file)

        trial_dir = f"Contrastive_{trial_name}_results"
        print(f"[DEBUG main] => creating trial_dir={trial_dir}")
        os.makedirs(trial_dir, exist_ok=True)

        print(f"[DEBUG main] => calling process_config with {tmp_config_file}, steps={STEPS_TO_RUN}")
        error_happened = False
        try:
            process_config(tmp_config_file, STEPS_TO_RUN)
            print(f"[DEBUG main] => process_config finished for {trial_name}")
        except Exception as ex:
            print(f"[{trial_name}] => training error => {ex}")
            traceback.print_exc()
            error_happened = True
        finally:
            if os.path.exists(tmp_config_file):
                print(f"[DEBUG main] => removing temp config => {tmp_config_file}")
                os.remove(tmp_config_file)

        if error_happened:
            print(f"[DEBUG main] => skipping evaluate for {trial_name} because error")
            continue

        print(f"[DEBUG main] => calling evaluate_and_plot for {trial_dir}, suffix={trial_name}")
        summary = evaluate_and_plot(trial_dir, combo, trial_name)
        if summary is None:
            print(f"[{trial_name}] => No station data => skipping scoreboard entry.")
            continue

        rowd = {
            "trial_name": trial_name,
            "mean_nse_base":    summary["mean_nse_base"],
            "mean_nse_suisun":  summary["mean_nse_suisun"],
            "mean_nse_overall": summary["mean_nse_overall"],
            "mean_r2":          summary["mean_r2"]
        }
        for k,v in combo.items():
            rowd[k]=v

        print(f"[DEBUG main] => writing summary row => {rowd}")
        mode = "a" if os.path.exists(MASTER_SUMMARY) else "w"
        df_temp = pd.DataFrame([rowd])
        df_temp.to_csv(MASTER_SUMMARY, mode=mode, header=(not os.path.exists(MASTER_SUMMARY)), index=False)

    if os.path.exists(MASTER_SUMMARY):
        print(f"[DEBUG main] => Reading final scoreboard => {MASTER_SUMMARY}")
        df = pd.read_csv(MASTER_SUMMARY)
        df_sorted = df.sort_values("mean_nse_overall", ascending=False)
        print("\n=== FINAL SCOREBOARD ===")
        print(df_sorted)
    else:
        print("No successful trials => no summary at all.")


if __name__=="__main__":
    main()