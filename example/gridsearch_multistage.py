import os, json, copy, itertools, traceback, hashlib
from pathlib import Path
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from casanntra.staged_learning import process_config
from cache_manager import CacheManager


RUN_ID = "v1.2_MSTAGE"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"         
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DB = OUTPUT_DIR / "run_cache.sqlite"
CACHE = CacheManager(str(CACHE_DB))    

BASE_CONFIG_FILE = "transfer_config_contrastive.yml"
STEPS_TO_RUN = ["dsm2_base", "dsm2.schism", "base.suisun"]
MASTER_SUMMARY = f"gridsearch_{RUN_ID}_master_results.csv"
MODEL_NAMES = ["base.suisun", "base.suisun-secondary"]
STATIONS = [ "cse","bdl","rsl","emm2","jer","sal","frk","bac","oh4", "x2","mal","god","gzl","vol","pct","nsl2","tms","anh","trp"]
ABS_PREFIX = (OUTPUT_DIR / f"schism_base.suisun_gru2_{RUN_ID}").resolve()
OUTPUT_PREFIXES = {"base.suisun" : str(ABS_PREFIX), "base.suisun-secondary": str(ABS_PREFIX)}

HYPERPARAM_SPACE = {
    "contrast_weight": [1.0],

    "feature_layers": [
        [ {"type":"GRU","units":32,"trainable":True,"name":"lay1", "return_sequences":True},
          {"type":"GRU","units":16,"trainable":True,"name":"lay2", "return_sequences":True},
          {"type":"GRU","units":12,"trainable":True,"name":"lay3", "return_sequences":False} ]],

    "freeze_schedule": [
        [0,0,0],
        [0,0,1],
        [0,1,1]],

    "ndays":[105],

    "dsm2_init_lr":[0.008], "dsm2_main_lr":[0.001],
    "dsm2_init_epochs":[10], "dsm2_main_epochs":[100],
    "schism_init_lr":[0.003], "schism_main_lr":[0.001],
    "schism_init_epochs":[10], "schism_main_epochs":[35],
    "suisun_init_lr":[0.001], "suisun_main_lr":[0.0005],
    "suisun_init_epochs":[10], "suisun_main_epochs":[35]}

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
    if model_name == "base.suisun-secondary":
        ref_csv, ann_csv = (f"{full_prefix}_xvalid_ref_out_secondary_unscaled.csv", f"{full_prefix}_xvalid_1.csv")
    else:
        ref_csv, ann_csv = (f"{full_prefix}_xvalid_ref_out_unscaled.csv", f"{full_prefix}_xvalid_0.csv")

    if not Path(ref_csv).exists() or not Path(ann_csv).exists():
        raise FileNotFoundError(ref_csv + " / " + ann_csv)
    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])
    return pd.merge(df_ref, df_ann, on=["datetime","case"], suffixes=("", "_pred"))

def plot_timeseries_all_cases(df, station, model_name, out_dir, n_cases=7):
    stcol, stpred = station, f"{station}_pred"

    if stcol not in df.columns or stpred not in df.columns:
        return
    
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_cases, 1, figsize=(8, 2.5*n_cases), constrained_layout=True)
    axes = axes if n_cases > 1 else [axes]
    
    for i, ax in enumerate(axes):
        sub = df[df["case"] == i+1]
        if i == 0:
            ax.set_title(f"[{model_name}]  station={station}")
        ax.plot(sub["datetime"], sub[stcol], color="0.1", label="Ref")
        ax.plot(sub["datetime"], sub[stpred], label="ANN")
    
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
    df_base = df_trial[df_trial["model"] == "base.suisun-secondary"]
    df_suisun = df_trial[df_trial["model"] == "base.suisun"]
    return {
        "mean_nse_base" : round(df_base["nse"].mean(),4) if len(df_base) else np.nan,
        "mean_nse_suisun" : round(df_suisun["nse"].mean(),4) if len(df_suisun) else np.nan,
        "mean_nse_overall": round(df_trial["nse"].mean(),4),
        "mean_r2" : round(df_trial["r2"].mean(),4)}

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

        for step_idx, step in enumerate(mod_cfg["steps"]):
            orig_prefix = step["output_prefix"]
            run_prefix  = f"{orig_prefix}_{RUN_ID}"
            step_name   = f"{run_prefix}_{trial_name}"
            step["output_prefix"] = step_name

            if step.get("save_model_fname") not in (None, "None"):
                step["save_model_fname"] = step_name
            if step.get("load_model_fname") not in (None, "None"):
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
            else:               # SCHISM-Scenario
                step.update(init_train_rate = combo["suisun_init_lr"],
                            main_train_rate = combo["suisun_main_lr"],
                            init_epochs = combo["suisun_init_epochs"],
                            main_epochs = combo["suisun_main_epochs"])

            bargs = step.get("builder_args", {})
            bargs.update(contrast_weight = combo["contrast_weight"], ndays = combo["ndays"])
            layers = copy.deepcopy(combo["feature_layers"])

            freezeN = fsched[step_idx] if step_idx < len(fsched) else 0
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

            print(f"INFO  | {trial_name} | {step['name']} | querying cache …")
            hit, info = CACHE.check(step["name"], recipe,
                                    abs_model_path, abs_prefix_path,
                                    parent_hash)
            if hit:
                print(f"INFO  | {trial_name} | {step['name']} | cache HIT  ➜ SKIP")
                if not Path(f"{abs_model_path}.h5").exists():
                    raise RuntimeError(f"Cache hit but .h5 missing for {step['name']}")
                parent_hash = info
                continue

            print(f"INFO  | {trial_name} | {step['name']} | cache MISS ➜ TRAIN")
            tmp_yaml = SCRIPT_DIR / f"tmp_{trial_name}_{step['name']}.yml"
            save_yml({"output_dir": str(OUTPUT_DIR), "model_builder_config": mod_cfg["model_builder_config"], "steps": [step]}, tmp_yaml)

            try:
                process_config(tmp_yaml, [step["name"]])
            except Exception as e:
                print(f"ERROR | {trial_name} | {step['name']}: {e}")
                traceback.print_exc()
                break
            finally:
                tmp_yaml.unlink(missing_ok=True)

            key_hash, finalize = info
            finalize()                
            parent_hash = key_hash     

        trial_dir = SCRIPT_DIR / f"{RUN_ID}_{trial_name}_results"
        trial_dir.mkdir(exist_ok=True)
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
