import os
import yaml
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from casanntra.staged_learning import process_config


#NEW ------------------------------------------------------------------
#  Tag to keep every artefact unique across re‑runs
RUN_ID = "direct_v3"
#NEW ------------------------------------------------------------------


#OLD ------------------------------------------------------------------
# HYPERPARAM_SPACE = {
#     "freeze_bool": [True, False],
#     "arch_type":   ["LSTM", "GRU"],
#     ...
# }
#OLD ------------------------------------------------------------------

#NEW ------------------------------------------------------------------
HYPERPARAM_SPACE = {
    # ------- encoder architecture & freezing -------
    "feature_layers": [
        [
            {"type": "GRU", "units": 36, "trainable": True, "name": "lay1"},
            {"type": "GRU", "units": 18, "trainable": True, "name": "lay2"},
        ],
        [
            {"type": "GRU", "units": 32, "trainable": True, "name": "lay1"},
            {"type": "GRU", "units": 16, "trainable": True, "name": "lay2"},
        ],
    ],
    "freeze_schedule": [
        [0, 0, 0],   # train all layers in every step
        [0, 1, 1],   # freeze 1st layer after DSM2; keep it frozen
        [0, 1, 2],   # freeze 1st layer after DSM2; freeze 1‑2 after SCHISM
    ],
    "ndays": [90, 105, 120],
    # -------------- learning rates / epochs --------------
    # DSM2
    "dsm2_init_lr":     [0.008],
    "dsm2_main_lr":     [0.001],    
    "dsm2_init_epochs": [10],
    "dsm2_main_epochs": [100],
    # Schism Base
    "schism_init_lr":     [0.003],
    "schism_main_lr":     [0.001],
    "schism_init_epochs": [10],
    "schism_main_epochs": [35, 75],
    # Schism Suisun
    "suisun_init_lr":     [0.003, 0.001],
    "suisun_main_lr":     [0.001, 0.0004],
    "suisun_init_epochs": [10],
    "suisun_main_epochs": [35, 75],
}
#NEW ------------------------------------------------------------------


BASE_CONFIG_FILE = "transfer_config_direct.yml"
STEPS_TO_RUN     = ["dsm2_base", "dsm2.schism", "base.suisun"]

#OLD ------------------------------------------------------------------
# MASTER_SUMMARY = "gridsearch_direct_v2_master_results.csv"
#OLD ------------------------------------------------------------------
#NEW ------------------------------------------------------------------
MASTER_SUMMARY = f"gridsearch_{RUN_ID}_master_results.csv"
#NEW ------------------------------------------------------------------


#OLD ------------------------------------------------------------------
# BASE_PREFIX   = "dsm2.schism_base_gru2"
# SUISUN_PREFIX = "schism_base.suisun_gru2"
#OLD ------------------------------------------------------------------
#NEW ------------------------------------------------------------------
BASE_PREFIX   = f"dsm2.schism_base_gru2_{RUN_ID}"
SUISUN_PREFIX = f"schism_base.suisun_gru2_{RUN_ID}"
#NEW ------------------------------------------------------------------


STATIONS = [
    "cse","bdl","rsl","emm2","jer","sal","frk","bac","oh4","x2",
    "mal","god","gzl","vol","pct","nsl2","tms","anh","trp"
]

# ---------------- compute_metrics, load_and_merge, plot_timeseries,
# ---------------- evaluate_and_plot remain UNCHANGED
# (functions are identical to the version you supplied)
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "nse": np.nan, "pearson_r": np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]

    mae  = np.mean(np.abs(yt - yp))
    mse  = np.mean((yt - yp)**2)
    rmse = np.sqrt(mse)

    denom = np.sum((yt - np.mean(yt))**2)
    nse   = 1.0 - np.sum((yt - yp)**2)/denom if denom > 0 else np.nan
    corr  = np.corrcoef(yt, yp)[0,1] if len(yt) > 1 else np.nan

    return {"mae": mae, "rmse": rmse, "nse": nse, "pearson_r": corr}


def load_and_merge(step_prefix, trial_suffix):
    ref_csv = f"{step_prefix}_direct_{trial_suffix}_xvalid_ref_out_unscaled.csv"
    ann_csv = f"{step_prefix}_direct_{trial_suffix}_xvalid.csv"

    if not os.path.exists(ref_csv):
        print(f"(load_and_merge): missing {ref_csv}")
        return None
    if not os.path.exists(ann_csv):
        print(f"(load_and_merge): missing {ann_csv}")
        return None

    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])
    merged = pd.merge(df_ref, df_ann, on=["datetime","case"], how="inner", suffixes=("","_pred"))
    if merged.empty:
        print(f"(load_and_merge): empty merge, skip {ref_csv}")
        return None
    return merged


def plot_timeseries_all_cases(df_merged, station, step_label, out_dir, n_cases=7):
    stcol  = station
    stpred = station + "_pred"
    if (df_merged is None) or (stcol not in df_merged.columns) or (stpred not in df_merged.columns):
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(n_cases, 1, figsize=(8, 2.5*n_cases), constrained_layout=True)
    axes = axes if n_cases > 1 else [axes]

    for i, ax in enumerate(axes):
        case_id = i + 1
        subdf = df_merged[df_merged["case"] == case_id]
        if i == 0:
            ax.set_title(f"{step_label} => {station}")
        ax.plot(subdf["datetime"], subdf[stcol], color="0.1", label="Ref")
        ax.plot(subdf["datetime"], subdf[stpred], label="ANN")
        ax.set_ylabel("Norm EC")
        ax.set_title(f"Case={case_id}, #rows={subdf.shape[0]}")
    axes[0].legend()
    plt.savefig(os.path.join(out_dir, f"{step_label}_{station}.png"), dpi=150)
    plt.close(fig)


def evaluate_and_plot(trial_dir, combo, trial_suffix):
    df_base   = load_and_merge(BASE_PREFIX,   trial_suffix)
    df_suisun = load_and_merge(SUISUN_PREFIX, trial_suffix)

    base_outdir   = os.path.join(trial_dir, "base_step")
    suisun_outdir = os.path.join(trial_dir, "suisun_step")

    rows = []
    for station in STATIONS:
        base_met   = {"mae": np.nan, "rmse": np.nan, "nse": np.nan, "pearson_r": np.nan}
        suisun_met = {"mae": np.nan, "rmse": np.nan, "nse": np.nan, "pearson_r": np.nan}

        if (df_base is not None) and (station in df_base.columns) and (station+"_pred" in df_base.columns):
            base_met = compute_metrics(df_base[station], df_base[station+"_pred"])
            plot_timeseries_all_cases(df_base, station, "base_step", base_outdir, n_cases=7)

        if (df_suisun is not None) and (station in df_suisun.columns) and (station+"_pred" in df_suisun.columns):
            suisun_met = compute_metrics(df_suisun[station], df_suisun[station+"_pred"])
            plot_timeseries_all_cases(df_suisun, station, "suisun_step", suisun_outdir, n_cases=7)

        rows.append(
            {
                "station": station,
                "base_mae":   round(base_met["mae"],4),   "base_rmse":   round(base_met["rmse"],4),
                "base_nse":   round(base_met["nse"],4),   "base_pearson_r":   round(base_met["pearson_r"],4),
                "suisun_mae": round(suisun_met["mae"],4), "suisun_rmse": round(suisun_met["rmse"],4),
                "suisun_nse": round(suisun_met["nse"],4), "suisun_pearson_r": round(suisun_met["pearson_r"],4),
                **combo
            }
        )

    if not rows:
        print("No station data (evaluate_and_plot) => skip")
        return None

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(trial_dir, "trial_evaluation_metrics.csv"), index=False)

    mean_base_nse   = df_metrics["base_nse"].mean(skipna=True)
    mean_suisun_nse = df_metrics["suisun_nse"].mean(skipna=True)
    mean_overall    = 0.5 * (mean_base_nse + mean_suisun_nse)

    return {
        "mean_nse_base":   round(mean_base_nse,4),
        "mean_nse_suisun": round(mean_suisun_nse,4),
        "mean_nse_overall":round(mean_overall,4)
    }

# ---------------------------------------------------------------------


def load_yml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def main():
    if os.path.exists(MASTER_SUMMARY):
        os.remove(MASTER_SUMMARY)

    base_cfg = load_yml(BASE_CONFIG_FILE)

    keys       = list(HYPERPARAM_SPACE.keys())
    combos     = list(itertools.product(*[HYPERPARAM_SPACE[k] for k in keys]))
    all_combos = [dict(zip(keys, c)) for c in combos]
    print(f"[INFO] total combos = {len(all_combos)}")

    trial_counter = 0
    for combo in all_combos:
        trial_counter += 1
        trial_name = f"Trial{trial_counter}"
        print(f"\n=== Starting {trial_name} => {combo}")

        mod_cfg = copy.deepcopy(base_cfg)
        #NEW ------------------------------------------------------------
        mod_cfg["model_builder_config"]["args"]["ndays"] = combo["ndays"]
        fsched = combo["freeze_schedule"]
        #NEW ------------------------------------------------------------

        for i, step in enumerate(mod_cfg["steps"]):
            old_prefix = step["output_prefix"]
            #OLD step["output_prefix"] = f"{old_prefix}_direct_{trial_name}"
            #NEW
            step["output_prefix"] = f"{old_prefix}_{RUN_ID}_direct_{trial_name}"

            if step.get("save_model_fname") not in [None, "None"]:
                #OLD step["save_model_fname"] = f"{step['save_model_fname']}_direct_{trial_name}"
                #NEW
                step["save_model_fname"] = f"{step['save_model_fname']}_{RUN_ID}_direct_{trial_name}"
            if step.get("load_model_fname") not in [None, "None"]:
                #OLD step["load_model_fname"] = f"{step['load_model_fname']}_direct_{trial_name}"
                #NEW
                step["load_model_fname"] = f"{step['load_model_fname']}_{RUN_ID}_direct_{trial_name}"

            # (learning‑rate / epoch logic unchanged)
            if i == 0:   # dsm2_base
                step["init_train_rate"] = combo["dsm2_init_lr"]
                step["main_train_rate"] = combo["dsm2_main_lr"]
                step["init_epochs"]     = combo["dsm2_init_epochs"]
                step["main_epochs"]     = combo["dsm2_main_epochs"]
            elif i == 1: # dsm2.schism
                step["init_train_rate"] = combo["schism_init_lr"]
                step["main_train_rate"] = combo["schism_main_lr"]
                step["init_epochs"]     = combo["schism_init_epochs"]
                step["main_epochs"]     = combo["schism_main_epochs"]
            else:        # base.suisun
                step["init_train_rate"] = combo["suisun_init_lr"]
                step["main_train_rate"] = combo["suisun_main_lr"]
                step["init_epochs"]     = combo["suisun_init_epochs"]
                step["main_epochs"]     = combo["suisun_main_epochs"]

            # Builder args ------------------------------------------------
            bargs = step.get("builder_args", {})
            #OLD bargs["freeze_bool"] = combo["freeze_bool"]
            #OLD bargs["arch_type"]   = combo["arch_type"]
            #NEW
            bargs["ndays"] = combo["ndays"]

            arch_template = copy.deepcopy(combo["feature_layers"])
            freeze_cnt = fsched[i] if i < len(fsched) else 0
            for idx, layer in enumerate(arch_template):
                layer["trainable"] = (idx >= freeze_cnt)
            bargs["feature_layers"] = arch_template

            step["builder_args"] = bargs
            # -------------------------------------------------------------

        tmp_config = f"tmp_direct_{trial_name}.yml"
        save_yml(mod_cfg, tmp_config)

        #OLD trial_dir = f"V2_Direct_{trial_name}_results"
        #NEW
        trial_dir = f"{RUN_ID}_{trial_name}_results"
        os.makedirs(trial_dir, exist_ok=True)

        error_flag = False
        try:
            process_config(tmp_config, STEPS_TO_RUN)
        except Exception as ex:
            print(f"[ERROR] => {trial_name} => {ex}")
            traceback.print_exc()
            error_flag = True
        finally:
            if os.path.exists(tmp_config):
                os.remove(tmp_config)

        if error_flag:
            continue

        summary = evaluate_and_plot(trial_dir, combo, trial_name)
        if summary is None:
            continue

        rowd = dict(
            trial_name       = trial_name,
            mean_nse_base    = summary["mean_nse_base"],
            mean_nse_suisun  = summary["mean_nse_suisun"],
            mean_nse_overall = summary["mean_nse_overall"],
            **combo
        )

        mode = "a" if os.path.exists(MASTER_SUMMARY) else "w"
        pd.DataFrame([rowd]).to_csv(
            MASTER_SUMMARY,
            mode   = mode,
            header = (not os.path.exists(MASTER_SUMMARY)),
            index  = False,
        )

    if os.path.exists(MASTER_SUMMARY):
        df = pd.read_csv(MASTER_SUMMARY)
        df_sorted = df.sort_values("mean_nse_overall", ascending=False)
        print("\n=== FINAL SCOREBOARD (DSM2‑Direct) ===")
        print(df_sorted)
    else:
        print("Error: No successful trials")


if __name__ == "__main__":
    main()
