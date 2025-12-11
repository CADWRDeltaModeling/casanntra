import os, json, copy, itertools, traceback, hashlib
from pathlib import Path
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from casanntra.staged_learning import process_config
from cache_manager import CacheManager

# OLD: RUN_ID = "MSCEN_v3.1_DEBUG"
RUN_ID = "MSCEN_v4.1"
SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name.lower() == "example":
    OUTPUT_DIR = (SCRIPT_DIR / "output").resolve()
else:
    OUTPUT_DIR = (SCRIPT_DIR / "example" / "output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DB = OUTPUT_DIR / "run_cache.sqlite"
CACHE = None  
DEBUG_EVAL = True  

BASE_CONFIG_FILE = (SCRIPT_DIR / "transfer_config_multiscenario.yml").as_posix()
STEPS_TO_RUN = ["dsm2_base", "dsm2.schism", "base.multi"]
MASTER_SUMMARY = (SCRIPT_DIR / f"gridsearch_{RUN_ID}_master_results.csv").as_posix()

STATIONS = ["cse","bdl","rsl","emm2","jer","sal","frk","bac","oh4", "x2","mal","god","gzl","vol","pct","nsl2","tms","anh","trp"]

HYPERPARAM_SPACE = {
    "trunk_layers": [
        [{"type": "GRU",  "units": 38, "return_sequences": True, "name": "lay1", "trainable": True},
         {"type": "GRU",  "units": 19, "return_sequences": False, "name": "lay2", "trainable": True}],
         [{"type": "GRU",  "units": 32, "return_sequences": True, "name": "lay1", "trainable": True},
         {"type": "GRU",  "units": 16, "return_sequences": False, "name": "lay2", "trainable": True}]],

    "freeze_schedule": [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
    
    "ndays": [90, 105, 120],
    "dsm2_init_lr": [0.008], "dsm2_main_lr": [0.001],
    "dsm2_init_epochs": [10], "dsm2_main_epochs": [35],
    "schism_init_lr": [0.003], "schism_main_lr": [0.001],
    "schism_init_epochs": [10], "schism_main_epochs": [35],
    "multi_init_lr": [0.001],          
    "multi_main_lr": [0.0005],    
    "multi_init_epochs": [10],
    "multi_main_epochs": [35],
    "source_weight": [1.0],
    "target_weight": [1.0],
    "contrast_weight": [0.5, 1.0],

    "per_scenario_branch": [False],
    "branch_layers": [[]]}

def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
    if mask.sum() < 2:
        return {"mae":np.nan,"rmse":np.nan,"nse":np.nan,"pearson_r":np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    mae  = float(np.mean(np.abs(yt-yp)))
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    denom = float(np.sum((yt - np.mean(yt))**2))
    nse  = float(1.0 - np.sum((yt-yp)**2) / denom) if denom > 0 else np.nan
    r    = float(np.corrcoef(yt,yp)[0,1]) if len(yt) > 1 else np.nan
    return {"mae":mae,"rmse":rmse,"nse":nse,"pearson_r":r}

def _first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(" / ".join(paths))

def load_and_merge_ms(prefix: str, tag: str):

    ref_csv = _first_existing(f"{prefix}_xvalid_ref_out_{tag}_unscaled.csv", f"{prefix}_xvalid_ref_{tag}_unscaled.csv", f"{prefix}_xvalid_ref_{tag}.csv")
    ann_csv = _first_existing(f"{prefix}_xvalid_out_{tag}_unscaled.csv", f"{prefix}_xvalid_{tag}_unscaled.csv", f"{prefix}_xvalid_{tag}.csv")

    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])

    # OLD: Coercing to numeric causes NaN cross-joins with mixed string/numeric cases
    # df_ref["case"] = pd.to_numeric(df_ref["case"], errors="coerce")
    # df_ann["case"] = pd.to_numeric(df_ann["case"], errors="coerce")
    df_ref["datetime"] = pd.to_datetime(df_ref["datetime"], errors="coerce")
    df_ann["datetime"] = pd.to_datetime(df_ann["datetime"], errors="coerce")

    df = pd.merge(df_ref, df_ann, on=["datetime","case"], how="inner", suffixes=("", "_pred"))
    df = df.sort_values(["case","datetime"]).reset_index(drop=True)
    return df

def plot_timeseries_all_cases(df, station, tag, out_dir, n_cases=7):
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
        ax.set_title(f"[{tag}]  station={station}  case={case_id}")
    axes[0].legend()
    plt.savefig(out_dir / f"{tag}_{station}.png", dpi=150)
    plt.close(fig)

def evaluate_and_plot(trial_dir, stations, tags, multi_prefix, hyperparams):
    rows = []
    for tag in tags:
        try:
            df = load_and_merge_ms(multi_prefix, tag)
        except FileNotFoundError as e:
            print(f"DEBUG | evaluation skipped for tag={tag}: {e}")
            continue
        for st in stations:
            if st not in df.columns or f"{st}_pred" not in df.columns:
                continue
            met = compute_metrics(df[st], df[f"{st}_pred"])
            rows.append({**hyperparams, "tag":tag, "station":st,
                         **{k:round(v,4) for k,v in met.items()}})
            plot_timeseries_all_cases(df, st, tag, Path(trial_dir)/tag)

    if not rows:
        return None

    df_trial = pd.DataFrame(rows)
    df_trial["r2"] = df_trial["pearson_r"]**2
    Path(trial_dir).mkdir(exist_ok=True, parents=True)
    df_trial.to_csv(Path(trial_dir)/"trial_evaluation_metrics.csv", index=False)

    summary = {}
    for tag in tags:
        sub = df_trial[df_trial["tag"] == tag]
        if len(sub):
            summary[f"mean_nse_{tag}"] = round(sub["nse"].mean(), 4)
    summary["mean_nse_overall"] = round(df_trial["nse"].mean(), 4)
    summary["mean_r2"] = round(df_trial["r2"].mean(), 4)
    return summary

def load_yml(p):  
    p = Path(p)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(p.read_text())

def save_yml(o,p): 
    Path(p).write_text(yaml.safe_dump(o, sort_keys=False))


def _canon_transfer(val):
    if val in (None, "None", "null", "", "NULL"):
        return None
    return str(val).lower()

def _expand_outdir(p: str) -> str:
    if p in (None, "None"):
        return p
    p = p.replace("{output_dir}", str(OUTPUT_DIR))
    q = Path(p)
    if not q.is_absolute():
        if str(q).startswith("example/"):
            if SCRIPT_DIR.name.lower() == "example":
                q = (SCRIPT_DIR / q.relative_to("example")).resolve()
            else:
                q = (SCRIPT_DIR / q).resolve()
        else:
            q = (OUTPUT_DIR / q.name).resolve()
    return q.as_posix()

def _dataset_key_from_outputs(outputs: dict) -> str:
    canonical = json.dumps(
        {str(k).lower(): float(v) for k, v in outputs.items()},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

def _scenarios_fingerprint(scenarios: list, source_data_prefix: str, source_mask) -> str:
    skinny = {
        "source_data_prefix": source_data_prefix,
        "source_input_mask_regex": list(source_mask) if isinstance(source_mask, (list, tuple)) else source_mask,
        "scenarios": [
            {
                "id": sc.get("id"),
                "input_prefix": sc.get("input_prefix"),
                "input_mask_regex": sc.get("input_mask_regex"),
            }
            for sc in (scenarios or [])
        ],
    }
    canonical = json.dumps(skinny, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _eval_step_mean_nse(step_name: str, transfer_type: str, step_prefix: str, tags: list, trial_name: str):
    """Compute mean NSE per tag for a single step and write to debug_runs."""
    if not DEBUG_EVAL:
        return
    debug_dir = Path(SCRIPT_DIR) / "debug_runs" / f"{RUN_ID}_{trial_name}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{step_name}_mean_nse.csv"
    rows = []

    for tag in tags:
        df = None
        try:
            df = load_and_merge_ms(step_prefix, tag)
        except FileNotFoundError:
            ref = Path(f"{step_prefix}_xvalid_ref_out_unscaled.csv")
            pred = Path(f"{step_prefix}_xvalid.csv")
            if ref.exists() and pred.exists():
                df_ref = pd.read_csv(ref, parse_dates=["datetime"])
                df_pred = pd.read_csv(pred, parse_dates=["datetime"])
                df = pd.merge(df_ref, df_pred, on=["datetime", "case"], suffixes=("", "_pred"))
        if df is None:
            continue
        nses = []
        for st in STATIONS:
            if st not in df.columns or f"{st}_pred" not in df.columns:
                continue
            met = compute_metrics(df[st], df[f"{st}_pred"])
            nses.append(met["nse"])
        if nses:
            rows.append({"tag": tag, "mean_nse": round(np.nanmean(nses), 4)})

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[debug] wrote step metrics -> {out_path}")

def main() -> None:
    
    Path(MASTER_SUMMARY).unlink(missing_ok=True)
    base_cfg = load_yml(BASE_CONFIG_FILE)
    grid_keys = list(HYPERPARAM_SPACE.keys())
    all_combos = [dict(zip(grid_keys, combo)) for combo in itertools.product(*[HYPERPARAM_SPACE[k] for k in grid_keys])]
    outputs_map = base_cfg["model_builder_config"]["args"]["output_names"]
    stations_in_yaml = list(outputs_map.keys())
    dataset_key = _dataset_key_from_outputs(outputs_map)

    for t_idx, combo in enumerate(all_combos, start=1):
        trial_name = f"Trial{t_idx}"
        print(f"\n{trial_name}: Starting hyperparameter combination")
        print(json.dumps(combo, indent=2))
        mod_cfg = copy.deepcopy(base_cfg)

        mod_cfg["model_builder_config"]["args"]["ndays"] = combo["ndays"]

        fsched = combo["freeze_schedule"]
        parent_hash = None

        multi_step_def = next(s for s in mod_cfg["steps"] if s["name"] == "base.multi")
        bargs_multi = multi_step_def.get("builder_args", {})
        sc_list = bargs_multi.get("scenarios", []) or []
        source_prefix  = bargs_multi.get("source_data_prefix", None)
        source_mask = bargs_multi.get("source_input_mask_regex", None)
        scenarios_key = _scenarios_fingerprint(sc_list, source_prefix, source_mask)

        tags_for_eval = ["base"] + [sc.get("id", f"scenario{i}") for i, sc in enumerate(sc_list, start=1)]

        multi_prefix_for_eval = None

        for step_idx, step in enumerate(mod_cfg["steps"]):
            orig_prefix = step["output_prefix"]
            base_name = Path(orig_prefix).name
            step_out = (OUTPUT_DIR / f"{base_name}_{RUN_ID}_{trial_name}").resolve().as_posix()
            step["output_prefix"] = step_out

            if step.get("save_model_fname") not in (None, "None"):
                step["save_model_fname"] = step_out
            if step.get("load_model_fname") not in (None, "None"):
                step["load_model_fname"] = (OUTPUT_DIR / f"{Path(step['load_model_fname']).name}_{RUN_ID}_{trial_name}").resolve().as_posix()

            for k in ("save_model_fname", "load_model_fname", "output_prefix"):
                step[k] = _expand_outdir(step.get(k))

            bargs = step.get("builder_args", {}) or {}

            if step["name"] == "base.multi":
                multi_prefix_for_eval = step["output_prefix"]
                bargs["per_scenario_branch"] = combo["per_scenario_branch"]
                bargs["branch_layers"] = copy.deepcopy(combo["branch_layers"])
                branch_flag = combo["per_scenario_branch"]
                branch_spec = combo["branch_layers"]
            else:
                branch_flag = None
                branch_spec = None

            if step_idx == 0:
                step.update(init_train_rate = combo["dsm2_init_lr"], main_train_rate = combo["dsm2_main_lr"], init_epochs = combo["dsm2_init_epochs"],
                            main_epochs = combo["dsm2_main_epochs"])
                
            elif step_idx == 1:
                step.update(init_train_rate = combo["schism_init_lr"], main_train_rate = combo["schism_main_lr"], init_epochs = combo["schism_init_epochs"],
                            main_epochs = combo["schism_main_epochs"])

            else:
                step.update(init_train_rate = combo["multi_init_lr"], main_train_rate = combo["multi_main_lr"], init_epochs = combo["multi_init_epochs"], 
                            main_epochs = combo["multi_main_epochs"])

            trunk_template = copy.deepcopy(combo["trunk_layers"])
            if trunk_template:
                trunk_template[-1]["return_sequences"] = False
            base_template = copy.deepcopy(trunk_template)
            freezeN = fsched[step_idx] if step_idx < len(fsched) else 0
            for j, layer in enumerate(trunk_template):
                layer["trainable"] = bool(j >= freezeN)
            for j, layer in enumerate(base_template):
                layer["trainable"] = bool(j >= freezeN)

            if bargs.get("transfer_type", None) in (None, "None", "direct", "Direct"):
                bargs["base_layers"] = copy.deepcopy(base_template)
            else:
                bargs["trunk_layers"] = copy.deepcopy(trunk_template)

            bargs["source_weight"] = combo["source_weight"]
            bargs["target_weight"] = combo["target_weight"]
            bargs["contrast_weight"] = combo["contrast_weight"]

            step["builder_args"] = bargs

            transfer_type = _canon_transfer(bargs.get("transfer_type"))
            
            recipe = {
                "input_prefix": step["input_prefix"],
                "transfer_type": transfer_type,
                "ndays": combo["ndays"],
                "freeze_index": freezeN,
                "init_lr": step["init_train_rate"],
                "main_lr": step["main_train_rate"],
                "init_epochs": step["init_epochs"],
                "main_epochs": step["main_epochs"],
                "dataset_key": dataset_key,
                "trunk_layers": trunk_template,
                "scenarios_key": scenarios_key if step["name"] == "base.multi" else None,
                "source_weight": combo["source_weight"],
                "target_weight": combo["target_weight"],
                "contrast_weight": combo["contrast_weight"],
                "per_scenario_branch": branch_flag,
                "branch_layers": branch_spec}

            abs_model_path = str(Path(step["save_model_fname"]).resolve())
            abs_prefix_path = str(Path(step["output_prefix"]).resolve())

            hit = False; info = (None, None)
            if CACHE is not None:
                print(f"{trial_name} | {step['name']}: querying cache")
                hit, info = CACHE.check(step["name"], recipe, abs_model_path, abs_prefix_path, parent_hash)
            if hit:
                print(f"{trial_name} | {step['name']}: cache HIT: SKIP")
                parent_hash = info
                continue

            print(f"{trial_name} | {step['name']}: cache MISS: TRAIN")

            tmp_yaml = SCRIPT_DIR / f"tmp_{trial_name}_{step['name']}.yml"
            save_yml({"output_dir": str(OUTPUT_DIR), "model_builder_config": mod_cfg["model_builder_config"], "steps": [step]}, tmp_yaml)

            try:
                process_config(tmp_yaml, [step["name"]])
            except Exception as e:
                print(f"ERROR {trial_name}: {step['name']}: {e}")
                traceback.print_exc()
                break

            finally:
                tmp_yaml.unlink(missing_ok=True)

            if DEBUG_EVAL:
                step_tags = tags_for_eval if step["name"] == "base.multi" else ["base"]
                _eval_step_mean_nse(step_name=step["name"],
                                    transfer_type=transfer_type,
                                    step_prefix=step["output_prefix"],
                                    tags=step_tags,
                                    trial_name=trial_name)

            parent_hash = None

        if multi_prefix_for_eval is None:
            print("Multi-scenario prefix not captured; skip evaluation for this trial.")
            continue

        trial_dir = SCRIPT_DIR / f"{RUN_ID}_{trial_name}_results"
        trial_dir.mkdir(exist_ok=True)
        summary = evaluate_and_plot(trial_dir=trial_dir, stations=stations_in_yaml, tags=tags_for_eval, multi_prefix=multi_prefix_for_eval, hyperparams=combo)

        if summary is None:
            continue

        row = {"trial_name": trial_name, **summary, **combo}
        mode = "a" if Path(MASTER_SUMMARY).exists() else "w"
        pd.DataFrame([row]).to_csv(MASTER_SUMMARY, mode=mode, header=not Path(MASTER_SUMMARY).exists(), index=False)

    if Path(MASTER_SUMMARY).exists():
        df = pd.read_csv(MASTER_SUMMARY)
        if "mean_nse_overall" in df.columns:
            df = df.sort_values("mean_nse_overall", ascending=False)
        print("Final Results:")
        print(df.head(20).to_string(index=False))
    else:
        print("No successful trials recorded.")


if __name__ == "__main__":
    main()
