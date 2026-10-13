import os, json, copy, itertools, traceback, hashlib
from pathlib import Path
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from casanntra.staged_learning import process_config
from cache_manager import CacheManager


# Options: "suisun", "slr", "cache", "franks"
SCENARIO = "suisun"
VERSION = "residual_V1"  # Residual contrast: source + delta heads, target = source + delta
# VERSION = "v9.3"  # No DSM2 pretraining: test if Step 1 provides value
# VERSION = "v8"  # Contrastive: test higher contrast_weight [0.5, 2.0, 5.0]

# V10: Include DSM2 pretraining (match V4/V8 pipeline for comparability)
SKIP_DSM2 = False

SCENARIO_CONFIG = {
    "suisun": {"input_prefix": "schism_suisun", "step_name": "base.suisun"},
    "slr":    {"input_prefix": "schism_slr",    "step_name": "base.slr"},
    "cache":  {"input_prefix": "schism_cache",  "step_name": "base.cache"},
    "franks": {"input_prefix": "schism_franks", "step_name": "base.franks"}}

_scen = SCENARIO_CONFIG[SCENARIO]
RUN_ID = f"{VERSION}_MSTAGE_{SCENARIO}"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DB = OUTPUT_DIR / "run_cache.sqlite"
CACHE = None

BASE_CONFIG_FILE = "transfer_config_multistage.yml"
STEPS_TO_RUN = ["dsm2.schism", _scen["step_name"]] if SKIP_DSM2 else ["dsm2_base", "dsm2.schism", _scen["step_name"]]
MASTER_SUMMARY = f"gridsearch_{RUN_ID}_master_results.csv"

# For evaluation: target (Stage 3), secondary (Stage 3 source head for contrastive),
# and dsm2.schism (Stage 2 for direct mode base metrics)
MODEL_NAMES = [_scen["step_name"], f"{_scen['step_name']}-secondary", "dsm2.schism"]
STATIONS = ["x2","mrz","pct","mal","god","vol","gzl","bdl","nsl2","cse","emm2","tms","anh","jer","sal","frk","srv","bac","rsl","oh4"]
ABS_PREFIX = (OUTPUT_DIR / f"schism_{_scen['step_name']}_gru2_{RUN_ID}").resolve()
BASE_PREFIX = (OUTPUT_DIR / f"dsm2.schism_base_gru2_{RUN_ID}").resolve()
OUTPUT_PREFIXES = {
    _scen["step_name"]: str(ABS_PREFIX),
    f"{_scen['step_name']}-secondary": str(ABS_PREFIX),
    "dsm2.schism": str(BASE_PREFIX)
}

HYPERPARAM_SPACE = {
    "ndays": [105],
    "contrast_weight": [1.0, 10.0, 100.0],  # Residual V1: test if decoupled delta head tolerates high cw

    "feature_layers": [
        [{"type":"GRU","units":38,"trainable":True,"name":"lay1","return_sequences":True},
         {"type":"GRU","units":19,"trainable":True,"name":"lay2","return_sequences":False}]],

    "freeze_schedule": [[0, 0, 1]],  # Freeze lay1 in step 3

    "dsm2_init_lr":[0.008], "dsm2_main_lr":[0.001],
    "dsm2_init_epochs":[10], "dsm2_main_epochs":[35],
    "schism_init_lr":[0.003], "schism_main_lr":[0.001],
    "schism_init_epochs":[10], "schism_main_epochs":[35],
    "scenario_init_lr":[0.001], "scenario_main_lr":[0.0005],
    "scenario_init_epochs":[10], "scenario_main_epochs":[35]}

def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
    if mask.sum() < 2:
        return {"mae":np.nan,"rmse":np.nan,"nse":np.nan,"pearson_r":np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(np.mean(np.abs(yt-yp)))
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    nse = 1.0 - np.sum((yt-yp)**2) / np.sum((yt-np.mean(yt))**2)
    r = float(np.corrcoef(yt,yp)[0,1]) if len(yt) > 1 else np.nan
    return {"mae":mae,"rmse":rmse,"nse":nse,"pearson_r":r}

def load_and_merge(model_name, trial_suffix):
    base_prefix = OUTPUT_PREFIXES[model_name]
    full_prefix = f"{base_prefix}_{trial_suffix}"
    if model_name.endswith("secondary"):
        ref_csv = f"{full_prefix}_xvalid_ref_out_secondary_unscaled.csv"
        ann_csv = f"{full_prefix}_xvalid_1.csv"
    else:
        ref_csv = f"{full_prefix}_xvalid_ref_out_unscaled.csv"
        # Try indexed format (_xvalid_0.csv) for contrastive, fall back to direct format (_xvalid.csv)
        ann_csv_indexed = f"{full_prefix}_xvalid_0.csv"
        ann_csv_direct = f"{full_prefix}_xvalid.csv"
        ann_csv = ann_csv_indexed if Path(ann_csv_indexed).exists() else ann_csv_direct

    if not Path(ref_csv).exists() or not Path(ann_csv).exists():
        raise FileNotFoundError(ref_csv + " / " + ann_csv)
    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])

    # Normalize datetime formats (ref uses ISO 'T' format, pred uses space format)
    df_ref["datetime"] = pd.to_datetime(df_ref["datetime"], errors="coerce")
    df_ann["datetime"] = pd.to_datetime(df_ann["datetime"], errors="coerce")

    df = pd.merge(df_ref, df_ann, on=["datetime","case"], how="inner", suffixes=("", "_pred"))
    df = df.sort_values(["case","datetime"]).reset_index(drop=True)
    return df

def plot_timeseries_all_cases(df, station, model_name, out_dir, n_cases=7):
    stcol, stpred = station, f"{station}_pred"
    if stcol not in df.columns or stpred not in df.columns:
        return

    df = df.copy()
    df[stcol] = pd.to_numeric(df[stcol], errors="coerce")
    df[stpred] = pd.to_numeric(df[stpred], errors="coerce")
    cases = [c for c in pd.unique(df["case"]) if pd.notnull(c)]
    try:
        cases = sorted(cases)
    except Exception:
        pass
    if not cases:
        return
    cases = cases[:max(1, min(n_cases, len(cases)))]

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5*n), constrained_layout=True)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for ax, case_id in zip(axes, cases):
        sub = df[df["case"] == case_id]
        ax.plot(sub["datetime"], sub[stcol], label="Ref")
        ax.plot(sub["datetime"], sub[stpred], label="ANN")
        ax.set_title(f"[{model_name}]  station={station}  case={case_id}")
    axes[0].legend()
    plt.savefig(out_dir / f"{model_name}_{station}.png", dpi=150)
    plt.close(fig)

def evaluate_and_plot(trial_dir, hyperparams, trial_suffix):
    rows = []
    for model_name in MODEL_NAMES:
        try:
            df = load_and_merge(model_name, trial_suffix)
        except FileNotFoundError as e:
            print(f"DEBUG | evaluation skipped: {e}")
            continue
        for station in STATIONS:
            if station not in df.columns:
                continue
            met = compute_metrics(df[station], df[f"{station}_pred"])
            rows.append({**hyperparams, "model":model_name, "station":station,
                         **{k:round(v,4) for k,v in met.items()}})
            plot_timeseries_all_cases(df, station, model_name, Path(trial_dir)/model_name)

    if not rows:
        return None
    df_trial = pd.DataFrame(rows)
    df_trial["r2"] = df_trial["pearson_r"]**2
    df_trial.to_csv(Path(trial_dir)/"trial_evaluation_metrics.csv", index=False)

    # Target: Stage 3 scenario model
    target_model = MODEL_NAMES[0]
    df_target = df_trial[df_trial["model"] == target_model]

    # Base: secondary (contrastive) or dsm2.schism (direct) - whichever exists
    secondary_model = MODEL_NAMES[1]  # Stage 3 source head (contrastive only)
    stage2_model = MODEL_NAMES[2]     # Stage 2 dsm2.schism (always exists)
    df_secondary = df_trial[df_trial["model"] == secondary_model]
    df_stage2 = df_trial[df_trial["model"] == stage2_model]
    df_base = df_secondary if len(df_secondary) > 0 else df_stage2

    # Compute mean metrics
    mean_nse_base = round(df_base["nse"].mean(), 4) if len(df_base) else np.nan
    mean_nse_target = round(df_target["nse"].mean(), 4) if len(df_target) else np.nan

    # Overall: 0.5 * (base + target) - same as gridsearch_direct.py
    if len(df_base) > 0 and len(df_target) > 0:
        mean_nse_overall = round(0.5 * (mean_nse_base + mean_nse_target), 4)
    else:
        mean_nse_overall = mean_nse_target if len(df_target) else np.nan

    df_for_overall = pd.concat([df_target, df_base]) if len(df_base) > 0 else df_target
    mean_r2 = round(df_for_overall["r2"].mean(), 4)

    return {
        "mean_nse_base": mean_nse_base,
        "mean_nse_target": mean_nse_target,
        "mean_nse_overall": mean_nse_overall,
        "mean_r2": mean_r2}

def load_yml(p):
    return yaml.safe_load(Path(p).read_text())

def save_yml(o,p):
    Path(p).write_text(yaml.safe_dump(o, sort_keys=False))

def _canon_transfer(val):
    """Map all empty / 'None' / 'null' spellings → None."""
    if val in (None, "None", "null", "", "NULL"):
        return None
    return str(val).lower()

def _expand_outdir(p: str) -> str:
    """Replace the `{output_dir}` placeholder with absolute path."""
    if p in (None, "None"):
        return p
    return p.replace("{output_dir}", str(OUTPUT_DIR))

def _dataset_key_from_outputs(outputs: dict) -> str:
    """
    Build a stable fingerprint of the dataset definition
    (station set + per-station scaling factors).

    We canonicalize the mapping and SHA-256 hash it. Any change—
    removing 'trp', adding a station, or tweaking a scale—changes the key.
    """
    canonical = json.dumps(
        {str(k).lower(): float(v) for k, v in outputs.items()},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    Path(MASTER_SUMMARY).unlink(missing_ok=True)

    base_cfg = load_yml(BASE_CONFIG_FILE)
    grid_keys  = list(HYPERPARAM_SPACE.keys())
    all_combos = [dict(zip(grid_keys, combo)) for combo in itertools.product(*[HYPERPARAM_SPACE[k] for k in grid_keys])]

    for t_idx, combo in enumerate(all_combos, start=1):
        trial_name = f"Trial{t_idx}"
        print(f"\n========= {trial_name} =========")
        print(json.dumps(combo, indent=2))
        mod_cfg = copy.deepcopy(base_cfg)
        mod_cfg["model_builder_config"]["args"]["ndays"] = combo["ndays"]
        fsched = combo["freeze_schedule"]
        parent_hash = None

        outputs_map = mod_cfg["model_builder_config"]["args"]["output_names"]
        dataset_key = _dataset_key_from_outputs(outputs_map)

        trial_dir = SCRIPT_DIR / f"{RUN_ID}_{trial_name}_results"
        trial_dir.mkdir(exist_ok=True)

        run_step_idx = 0  # Track index among steps we actually run (for freeze_schedule)
        for step_idx, step in enumerate(mod_cfg["steps"]):
            if step["name"] not in STEPS_TO_RUN:
                continue
            orig_prefix = step["output_prefix"]
            run_prefix  = f"{orig_prefix}_{RUN_ID}"
            step_name   = f"{run_prefix}_{trial_name}"
            step["output_prefix"] = step_name

            if step.get("save_model_fname") not in (None, "None"):
                step["save_model_fname"] = step_name
            if step.get("load_model_fname") not in (None, "None"):
                # V9: When skipping DSM2, don't load pretrained weights for SCHISM Base
                if SKIP_DSM2 and step["name"] == "dsm2.schism":
                    step["load_model_fname"] = None
                    # Also set transfer_type to None for fresh training
                    if "builder_args" in step:
                        step["builder_args"]["transfer_type"] = None
                else:
                    step["load_model_fname"] = f"{step['load_model_fname']}_{RUN_ID}_{trial_name}"


            for k in ("save_model_fname", "load_model_fname", "output_prefix"):
                step[k] = _expand_outdir(step.get(k))

            if step_idx == 0:   # DSM2-Base
                step.update(init_train_rate = combo["dsm2_init_lr"],
                            main_train_rate = combo["dsm2_main_lr"],
                            init_epochs = combo["dsm2_init_epochs"],
                            main_epochs = combo["dsm2_main_epochs"])
            elif step_idx == 1: # SCHISM‑Base
                step.update(init_train_rate = combo["schism_init_lr"],
                            main_train_rate = combo["schism_main_lr"],
                            init_epochs = combo["schism_init_epochs"],
                            main_epochs = combo["schism_main_epochs"])
            else:               # SCHISM-Scenario (suisun/slr/cache/franks)
                step.update(init_train_rate = combo["scenario_init_lr"],
                            main_train_rate = combo["scenario_main_lr"],
                            init_epochs = combo["scenario_init_epochs"],
                            main_epochs = combo["scenario_main_epochs"])

            bargs = step.get("builder_args", {})
            bargs.update(contrast_weight = combo["contrast_weight"], ndays = combo["ndays"])
            layers = copy.deepcopy(combo["feature_layers"])

            freezeN = fsched[run_step_idx] if run_step_idx < len(fsched) else 0
            for j,l in enumerate(layers):
                l["trainable"] = (j >= freezeN)
            bargs["feature_layers"] = layers
            step["builder_args"] = bargs

            recipe = {
                "input_prefix" : step["input_prefix"],
                "transfer_type" : _canon_transfer(bargs.get("transfer_type")),
                "feature_layers" : layers,
                "ndays" : combo["ndays"],
                "contrast_weight": combo["contrast_weight"],
                "freeze_index" : freezeN,
                "init_lr" : step["init_train_rate"],
                "main_lr" : step["main_train_rate"],
                "init_epochs" : step["init_epochs"],
                "main_epochs" : step["main_epochs"],
                "dataset_key" : dataset_key}

            abs_model_path  = str(Path(step["save_model_fname"]).resolve())
            abs_prefix_path = str(Path(step["output_prefix"]).resolve())

            print(f"INFO  | {trial_name} | {step['name']} | caching disabled ➜ TRAIN")
            step_yaml = trial_dir / f"config_{step['name']}.yml"
            step_config = {"output_dir": str(OUTPUT_DIR), "model_builder_config": mod_cfg["model_builder_config"], "steps": [step]}
            save_yml(step_config, step_yaml)
            print(f"INFO  | saved config -> {step_yaml}")

            try:
                process_config(step_yaml, [step["name"]])
            except Exception as e:
                print(f"ERROR | {trial_name} | {step['name']}: {e}")
                traceback.print_exc()
                break

            run_step_idx += 1  # Increment for next step's freeze_schedule lookup
            parent_hash = None

        summary = evaluate_and_plot(trial_dir, combo, trial_name)
        if summary is None:
            continue
        row  = {"trial_name": trial_name, **summary, **combo}
        mode = "a" if Path(MASTER_SUMMARY).exists() else "w"
        pd.DataFrame([row]).to_csv(MASTER_SUMMARY, mode=mode, header=not Path(MASTER_SUMMARY).exists(), index=False)

    if Path(MASTER_SUMMARY).exists():
        df = pd.read_csv(MASTER_SUMMARY)
        print("\n========= FINAL SCOREBOARD =========")
        print(df.sort_values("mean_nse_overall", ascending=False).head(15))
    else:
        print("No successful trials recorded.")

if __name__ == "__main__":
    main()
