"""Single gridsearch driver for both builders.

Usage: python gridsearch.py experiments/<spec>.py (set START_TRIAL=<n> to resume a run).

A spec file defines VERSION (run id; outputs go to runs/<VERSION>/, refused if it exists unless
START_TRIAL > 1), CONFIG (YAML filename in configs/), STEPS (step names to run in order; each
loads the previous run step's saved model, the first loads nothing), GRID ({key: [values]}, one
trial per combination) and optionally CONTRAST_SCALES ({station: scale}, used when
use_contrast_scales is True).

GRID keys: layers (feature_layers for MSTAGE, trunk/base layers for MSCEN), freeze (per run step,
number of leading layers frozen), schedule ({step_name: (init_lr, main_lr, init_epochs,
main_epochs)}), ndays, contrast_weight, repeat. MSCEN only: source_weight, target_weight,
per_scenario_branch, branch_layers, use_contrast_scales. MSTAGE only: transfer_type (override for
the last run step).

runs/<VERSION>/ holds master.csv, provenance.txt, spec.py and per trial Trial<n>/ with
config_<step>.yml, metrics.csv, models/, xvalid/ and plots/<head>/. The xvalid filename
conventions written by staged_learning and xvalid_multi are in metrics.py.
"""
import copy, importlib.util, itertools, json, os, shutil, socket, subprocess, sys, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from casanntra.staged_learning import process_config
from metrics import compute_metrics, load_and_merge, xvalid_pair, xvalid_pair_ms

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"
RECURRENT = ("GRU", "LSTM")


def canon_transfer(val):
    if val in (None, "None", "null", "", "NULL"):
        return None
    return str(val).lower()


def load_spec(path):
    spec = importlib.util.spec_from_file_location("experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("VERSION", "CONFIG", "STEPS", "GRID"):
        if not hasattr(module, name):
            raise ValueError(f"{path} must define {name}")
    return module


def builder_family(cfg):
    name = cfg["model_builder_config"]["builder_name"]
    if name == "MultiStageModelBuilder":
        return "mstage"
    if name == "MultiScenarioModelBuilder":
        return "mscen"
    raise ValueError(f"unsupported builder_name {name}")


def write_provenance(run_dir, spec_path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=SCRIPT_DIR).stdout.strip()
    diff = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=SCRIPT_DIR).stdout
    header = f"started {datetime.now():%Y-%m-%d %H:%M:%S}\nhost {socket.gethostname()}\npython {sys.executable}\nHEAD {head}\n"
    (run_dir / "provenance.txt").write_text(header + "\n" + diff)
    shutil.copy(spec_path, run_dir / "spec.py")


def freeze_layers(layers, freeze, run_idx):
    layers = copy.deepcopy(layers)
    n_frozen = freeze[run_idx] if run_idx < len(freeze) else 0
    for j, layer in enumerate(layers):
        layer["trainable"] = j >= n_frozen
    return layers


def prepare_step(step, combo, run_idx, is_last, trial_dir, last_saved, family, contrast_scales):
    name = Path(step["output_prefix"]).name
    step["output_prefix"] = str(trial_dir / "xvalid" / name)
    if step.get("save_model_fname") not in (None, "None"):
        step["save_model_fname"] = str(trial_dir / "models" / name)
    bargs = step.get("builder_args") or {}
    step["builder_args"] = bargs
    is_multi = family == "mscen" and bool(bargs.get("scenarios"))
    if step.get("load_model_fname") not in (None, "None"):
        step["load_model_fname"] = last_saved
        if last_saved is None and not is_multi:
            bargs["transfer_type"] = None

    init_lr, main_lr, init_epochs, main_epochs = combo["schedule"][step["name"]]
    step.update(init_train_rate=init_lr, main_train_rate=main_lr, init_epochs=init_epochs, main_epochs=main_epochs)
    bargs["ndays"] = combo["ndays"]

    if family == "mstage":
        if is_last and "transfer_type" in combo:
            bargs["transfer_type"] = combo["transfer_type"]
            if canon_transfer(combo["transfer_type"]) == "direct":
                for key in ("source_data_prefix", "source_input_mask_regex", "contrast_weight", "save_modified_orig_model_fname"):
                    bargs.pop(key, None)
        if "contrast_weight" in combo and canon_transfer(bargs.get("transfer_type")) in ("contrastive", "multi-direct"):
            bargs["contrast_weight"] = combo["contrast_weight"]
        bargs["feature_layers"] = freeze_layers(combo["layers"], combo["freeze"], run_idx)
        return canon_transfer(bargs.get("transfer_type"))

    recurrent_branch = combo.get("per_scenario_branch", False) and any(bl["type"].upper() in RECURRENT for bl in combo.get("branch_layers", []))
    if is_multi:
        bargs["per_scenario_branch"] = combo.get("per_scenario_branch", False)
        bargs["branch_layers"] = copy.deepcopy(combo.get("branch_layers", []))
        bargs["source_weight"] = combo.get("source_weight", 1.0)
        bargs["target_weight"] = combo.get("target_weight", 1.0)
        bargs["contrast_weight"] = combo["contrast_weight"]
        if combo.get("use_contrast_scales", False):
            if contrast_scales is None:
                raise ValueError("use_contrast_scales=True but the spec defines no CONTRAST_SCALES")
            bargs["contrast_scales"] = dict(contrast_scales)
        else:
            bargs.pop("contrast_scales", None)
        if recurrent_branch:
            bargs["include_source_branch"] = True
    layers = freeze_layers(combo["layers"], combo["freeze"], run_idx)
    if not (is_multi and recurrent_branch):
        layers[-1]["return_sequences"] = False
    if canon_transfer(bargs.get("transfer_type")) in (None, "direct"):
        bargs["base_layers"] = layers
    else:
        bargs["trunk_layers"] = layers
    return canon_transfer(bargs.get("transfer_type"))










def sort_cases(cases):
    return sorted(c for c in cases if not isinstance(c, str)) + sorted(c for c in cases if isinstance(c, str))


def plot_cases(df, station, label, out_dir, n_cases=7):
    pred = f"{station}_pred"
    cases = sort_cases([c for c in pd.unique(df["case"]) if pd.notnull(c)])[:n_cases]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 1, figsize=(8, 2.5 * len(cases)), constrained_layout=True, squeeze=False)
    for ax, case_id in zip(axes[:, 0], cases):
        sub = df[df["case"] == case_id]
        ax.plot(sub["datetime"], pd.to_numeric(sub[station], errors="coerce"), label="Ref")
        ax.plot(sub["datetime"], pd.to_numeric(sub[pred], errors="coerce"), label="ANN")
        ax.set_title(f"[{label}] station={station} case={case_id}")
    axes[0, 0].legend()
    plt.savefig(out_dir / f"{label}_{station}.png", dpi=150)
    plt.close(fig)


def station_rows(specs, stations, combo, trial_dir, label_col):
    rows = []
    for label, ref_csv, ann_csv in specs:
        df = load_and_merge(ref_csv, ann_csv)
        for station in stations:
            if station not in df.columns or f"{station}_pred" not in df.columns:
                continue
            met = compute_metrics(df[station], df[f"{station}_pred"])
            rows.append({**combo, label_col: label, "station": station, **{k: round(v, 4) for k, v in met.items()}})
            plot_cases(df, station, label, trial_dir / "plots" / str(label))
    if not rows:
        raise ValueError("no stations evaluated; station list does not match output columns")
    df_trial = pd.DataFrame(rows)
    df_trial["r2"] = df_trial["pearson_r"] ** 2
    df_trial.to_csv(trial_dir / "metrics.csv", index=False)
    return df_trial


def evaluate_mstage(run_steps, step_transfer, stations, combo, trial_dir):
    """mean_nse_overall = 0.5 * (base + target); base = secondary head if present, else the previous step."""
    target = run_steps[-1]
    tt = step_transfer[target["name"]]
    specs = [(target["name"], *xvalid_pair(target["output_prefix"], tt))]
    secondary = stage2 = None
    if tt in ("contrastive", "multi-direct"):
        secondary = f"{target['name']}-secondary"
        specs.append((secondary, *xvalid_pair(target["output_prefix"], tt, role="secondary")))
    if len(run_steps) > 1:
        prev = run_steps[-2]
        stage2 = prev["name"]
        specs.append((stage2, *xvalid_pair(prev["output_prefix"], step_transfer[stage2])))
    df = station_rows(specs, stations, combo, trial_dir, "model")
    df_target = df[df["model"] == target["name"]]
    df_base = df[df["model"] == secondary] if secondary else df.iloc[0:0]
    if not len(df_base) and stage2:
        df_base = df[df["model"] == stage2]
    mean_base = round(df_base["nse"].mean(), 4) if len(df_base) else np.nan
    mean_target = round(df_target["nse"].mean(), 4)
    overall = round(0.5 * (mean_base + mean_target), 4) if len(df_base) else mean_target
    df_all = pd.concat([df_target, df_base])
    return {"mean_nse_base": mean_base, "mean_nse_target": mean_target, "mean_nse_overall": overall, "mean_r2": round(df_all["r2"].mean(), 4)}


def evaluate_mscen(multi_step, tags, stations, combo, trial_dir):
    """mean_nse_overall = flat mean over all (tag, station) rows."""
    specs = [(tag, *xvalid_pair_ms(multi_step["output_prefix"], tag)) for tag in tags]
    df = station_rows(specs, stations, combo, trial_dir, "tag")
    summary = {f"mean_nse_{tag}": round(df[df["tag"] == tag]["nse"].mean(), 4) for tag in tags}
    summary["mean_nse_overall"] = round(df["nse"].mean(), 4)
    summary["mean_r2"] = round(df["r2"].mean(), 4)
    return summary


def append_master_row(master_csv, row, columns):
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in columns}], columns=columns)
    if master_csv.exists():
        existing = pd.read_csv(master_csv, nrows=0).columns.tolist()
        if existing != list(columns):
            raise ValueError(f"master CSV header mismatch in {master_csv}: {existing} vs {list(columns)}")
        df.to_csv(master_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(master_csv, mode="w", header=True, index=False)


def main(spec_path):
    exp = load_spec(spec_path)
    run_id = exp.VERSION
    contrast_scales = getattr(exp, "CONTRAST_SCALES", None)
    run_dir = RUNS_DIR / run_id
    master_csv = run_dir / "master.csv"
    start_trial = int(os.environ.get("START_TRIAL", "1"))
    if start_trial == 1 and run_dir.exists():
        raise SystemExit(f"{run_dir} exists: bump VERSION, or set START_TRIAL=<n> to resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(run_dir, spec_path)

    base_cfg = yaml.safe_load((SCRIPT_DIR / "configs" / exp.CONFIG).read_text())
    family = builder_family(base_cfg)
    stations = list(base_cfg["model_builder_config"]["args"]["output_names"])
    step_defs = {s["name"]: s for s in base_cfg["steps"]}
    missing = [s for s in exp.STEPS if s not in step_defs]
    if missing:
        raise ValueError(f"STEPS not in {exp.CONFIG}: {missing}")

    if family == "mscen":
        multi_def = next((s for s in base_cfg["steps"] if s["name"] in exp.STEPS and (s.get("builder_args") or {}).get("scenarios")), None)
        if multi_def is None:
            raise ValueError("MSCEN run needs a step with scenarios among STEPS")
        tags = ["base"] + [sc["id"] for sc in multi_def["builder_args"]["scenarios"]]
        summary_cols = [f"mean_nse_{tag}" for tag in tags] + ["mean_nse_overall", "mean_r2"]
    else:
        summary_cols = ["mean_nse_base", "mean_nse_target", "mean_nse_overall", "mean_r2"]
    grid_keys = list(exp.GRID)
    columns = ["trial_name", "status"] + summary_cols + grid_keys
    combos = [dict(zip(grid_keys, values)) for values in itertools.product(*[exp.GRID[k] for k in grid_keys])]
    print(f"{run_id} | {family} | steps={exp.STEPS} | {len(combos)} trial(s)")

    for t_idx, combo in enumerate(combos, start=1):
        if t_idx < start_trial:
            continue
        trial = f"Trial{t_idx}"
        print(f"\n========= {trial} =========")
        print(json.dumps(combo, indent=2, default=str))
        cfg = copy.deepcopy(base_cfg)
        cfg["model_builder_config"]["args"]["ndays"] = combo["ndays"]
        trial_dir = run_dir / trial
        for sub in ("models", "xvalid", "plots"):
            (trial_dir / sub).mkdir(parents=True, exist_ok=True)

        run_steps = [s for s in cfg["steps"] if s["name"] in exp.STEPS]
        step_transfer = {}
        failed_step = None
        last_saved = None
        for run_idx, step in enumerate(run_steps):
            step_transfer[step["name"]] = prepare_step(step, combo, run_idx, run_idx == len(run_steps) - 1, trial_dir, last_saved, family, contrast_scales)
            if step.get("save_model_fname") not in (None, "None"):
                last_saved = step["save_model_fname"]
            step_yaml = trial_dir / f"config_{step['name']}.yml"
            step_yaml.write_text(yaml.safe_dump({"output_dir": str(trial_dir), "model_builder_config": cfg["model_builder_config"], "steps": [step]}, sort_keys=False))
            try:
                process_config(step_yaml, [step["name"]])
            except Exception as e:
                print(f"ERROR | {trial} | {step['name']}: {e}")
                traceback.print_exc()
                failed_step = step["name"]
                break

        row = {"trial_name": trial, **{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in combo.items()}}
        if failed_step is not None:
            append_master_row(master_csv, {**row, "status": f"failed:{failed_step}"}, columns)
            continue
        try:
            if family == "mscen":
                multi_step = next(s for s in run_steps if s["builder_args"].get("scenarios"))
                summary = evaluate_mscen(multi_step, tags, stations, combo, trial_dir)
            else:
                summary = evaluate_mstage(run_steps, step_transfer, stations, combo, trial_dir)
        except Exception as e:
            print(f"ERROR | {trial} | evaluation failed: {e}")
            traceback.print_exc()
            append_master_row(master_csv, {**row, "status": f"eval_error:{type(e).__name__}"}, columns)
            continue
        append_master_row(master_csv, {**row, "status": "ok", **summary}, columns)

    if master_csv.exists():
        df = pd.read_csv(master_csv)
        print(f"\n========= FINAL SCOREBOARD ({run_id}) =========")
        print(df.sort_values("mean_nse_overall", ascending=False).head(20).to_string(index=False))
    else:
        print("No trials recorded.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python gridsearch.py experiments/<spec>.py")
    main(sys.argv[1])
