import os, json, copy, itertools, traceback, hashlib
from pathlib import Path
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from casanntra.staged_learning import process_config


# VERSION = "MSCEN_RMA_v1"  # 3-scenario run (base+suisun+cache+ft)
# VERSION = "MSCEN_RMA_v2"  # 1-scenario run (base+suisun only), DSM2 pretrained
# VERSION = "MSCEN_RMA_v3.2"  # No-DSM2 ablation: saturation test (1 scenario)
# VERSION = "MSCEN_RMA_v2.1"  # DSM2 pretrained, matched LR/epochs to v3.2
# VERSION = "MSCEN_RMA_v3.3"  # No-DSM2 ablation: all 3 scenarios from scratch
# VERSION = "MSCEN_RMA_v4"  # Architecture sweep: trunk size + per-scenario branch layers
# VERSION = "MSCEN_RMA_v5"  # Per-scenario GRU branch layers
# VERSION = "MSCEN_RMA_v6"  # Uniform 16/16/16 GRU branch (phase experiment)
# VERSION = "MSCEN_RMA_v7"  # Baseline [38,19] trunk + GRU(19) branch + cw sweep (BUGGED - lay2 not returning sequences)
# VERSION = "MSCEN_RMA_v8"  # GRU branch with FIXED return_sequences + cw sweep
# VERSION = "MSCEN_RMA_v8.1"  # Same as v8 + branch weights initialized from lay2
# VERSION = "MSCEN_RMA_v9a"  # [32,32]+GRU(32) branch, lay2 init, freeze x cw sweep
# VERSION = "MSCEN_RMA_v10"  # [32,16,16] 3-layer trunk, NO BRANCH, freeze x epochs x cw sweep
# VERSION = "MSCEN_RMA_v11"  # [32,16,16] 3-layer no-branch, cw sweet spot (2,3,5)
# VERSION = "MSCEN_RMA_v12_ctrscale"  # Contrast-scale A/B x matched 2-vs-3-layer trunk, 3 repeats (12 trials)
# VERSION = "MSCEN_RMA_v13_cwsweep"  # cw trade-off curve under contrast-scaled loss, 2-layer (8 trials)
VERSION = "MSCEN_RMA_v14_branch_ctrscale"  # Per-scenario GRU branch under contrast-scaled loss (Eli's dedicated-capacity test, 4 trials)

# Modular step control. Auto-chains load_model_fname when steps are skipped.
# 1-step: random init -> joint(base+scenarios)  [no-DSM2 ablation]
# STEPS_TO_RUN = ["base.multi"]
# 2-step: DSM2 -> joint(base+scenarios)
STEPS_TO_RUN = ["dsm2_base", "base.multi"]
# 3-step: DSM2 -> RMA base -> joint(base+suisun)
# STEPS_TO_RUN = ["dsm2_base", "dsm2.rma", "base.multi"]

RUN_ID = VERSION
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / "output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG_FILE = "transfer_config_multiscenario_rma.yml"
MASTER_SUMMARY = SCRIPT_DIR / f"gridsearch_{RUN_ID}_master_results.csv"

STATIONS = [
    "x2","mrz","pct","mal","god","vol","gzl","bdl","nsl2","cse",
    "emm2","tms","anh","jer","sal","frk","srv","bac","rsl","oh4"]

# v12: per-station typical contrast magnitude, std of (rma_suisun - rma_base) over
# aligned cases 1-7 (4078 rows, computed 2026-08-27), floored at 0.01 * absolute scale.
CONTRAST_SCALES = {
    "x2": 1.0, "mrz": 300.0, "pct": 1420.0, "mal": 243.0, "god": 578.0,
    "vol": 598.0, "gzl": 725.0, "bdl": 780.0, "nsl2": 898.0, "cse": 164.0,
    "emm2": 63.5, "tms": 42.6, "anh": 81.6, "jer": 35.2, "sal": 12.0,
    "frk": 25.0, "srv": 17.5, "bac": 17.8, "rsl": 16.2, "oh4": 14.3}

HYPERPARAM_SPACE = {
    # v14: per-scenario GRU branch under the contrast-scaled loss - the fair test of the
    # dedicated-capacity (phase) hypothesis. Equal widths [32,32]+GRU(32) so branch weight
    # seeding from lay2 works (v8.1/v9a mechanics). Anchors: no-branch cw=1 ctr .464 (v12),
    # cw=0.5 ctr .459 (v13).
    "trunk_layers": [
        [{"type": "GRU", "units": 32, "return_sequences": True, "name": "lay1", "trainable": True},
         {"type": "GRU", "units": 32, "return_sequences": True, "name": "lay2", "trainable": True}],
    ],

    "use_contrast_scales": [True],
    "repeat": [1, 2],

    "freeze_schedule": [[0, 0]],

    "ndays": [105],
    "dsm2_init_lr": [0.008], "dsm2_main_lr": [0.001],
    "dsm2_init_epochs": [10], "dsm2_main_epochs": [35],
    "base_init_lr": [0.003], "base_main_lr": [0.001],
    "base_init_epochs": [10], "base_main_epochs": [35],
    "multi_init_lr": [0.001], "multi_main_lr": [0.0005],
    "multi_init_epochs": [10], "multi_main_epochs": [105],
    "source_weight": [1.0],
    "target_weight": [1.0],
    "contrast_weight": [0.5, 1],

    "per_scenario_branch": [True],
    "branch_layers": [[{"type": "GRU", "units": 32, "name": "branch_1", "trainable": True}]],
}


# ============================================================
# EVALUATION UTILITIES
# ============================================================

def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
    if mask.sum() < 2:
        return {"mae": np.nan, "rmse": np.nan, "nse": np.nan, "pearson_r": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    nse = float(1.0 - np.sum((yt - yp) ** 2) / denom) if denom > 0 else np.nan
    r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else np.nan
    return {"mae": mae, "rmse": rmse, "nse": nse, "pearson_r": r}


def _first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(" / ".join(paths))


def load_and_merge_ms(prefix: str, tag: str):
    ref_csv = _first_existing(
        f"{prefix}_xvalid_ref_out_{tag}_unscaled.csv",
        f"{prefix}_xvalid_ref_{tag}_unscaled.csv",
        f"{prefix}_xvalid_ref_{tag}.csv")
    ann_csv = _first_existing(
        f"{prefix}_xvalid_out_{tag}_unscaled.csv",
        f"{prefix}_xvalid_{tag}_unscaled.csv",
        f"{prefix}_xvalid_{tag}.csv")

    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])
    df_ref["datetime"] = pd.to_datetime(df_ref["datetime"], errors="coerce")
    df_ann["datetime"] = pd.to_datetime(df_ann["datetime"], errors="coerce")

    df = pd.merge(df_ref, df_ann, on=["datetime", "case"], how="inner", suffixes=("", "_pred"))
    df = df.sort_values(["case", "datetime"]).reset_index(drop=True)
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), constrained_layout=True)
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
            rows.append({**hyperparams, "tag": tag, "station": st,
                         **{k: round(v, 4) for k, v in met.items()}})
            plot_timeseries_all_cases(df, st, tag, Path(trial_dir) / tag)

    if not rows:
        return None

    df_trial = pd.DataFrame(rows)
    df_trial["r2"] = df_trial["pearson_r"] ** 2
    Path(trial_dir).mkdir(exist_ok=True, parents=True)
    df_trial.to_csv(Path(trial_dir) / "trial_evaluation_metrics.csv", index=False)

    summary = {}
    for tag in tags:
        sub = df_trial[df_trial["tag"] == tag]
        if len(sub):
            summary[f"mean_nse_{tag}"] = round(sub["nse"].mean(), 4)
    summary["mean_nse_overall"] = round(df_trial["nse"].mean(), 4)
    summary["mean_r2"] = round(df_trial["r2"].mean(), 4)
    return summary


# ============================================================
# CONFIG HELPERS
# ============================================================

def load_yml(p):
    p = Path(p)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(p.read_text())


def save_yml(o, p):
    Path(p).write_text(yaml.safe_dump(o, sort_keys=False))


def _expand_outdir(p: str) -> str:
    if p in (None, "None"):
        return p
    return p.replace("{output_dir}", str(OUTPUT_DIR))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Resume support: set START_TRIAL > 1 to skip earlier trials; master CSV appended to.
    START_TRIAL = int(os.environ.get("START_TRIAL", "1"))
    if START_TRIAL == 1:
        MASTER_SUMMARY.unlink(missing_ok=True)
    else:
        print(f"RESUME MODE: starting from Trial{START_TRIAL}, preserving existing master CSV")

    base_cfg = load_yml(BASE_CONFIG_FILE)
    grid_keys = list(HYPERPARAM_SPACE.keys())
    all_combos = [dict(zip(grid_keys, combo))
                  for combo in itertools.product(*[HYPERPARAM_SPACE[k] for k in grid_keys])]
    stations_in_yaml = list(base_cfg["model_builder_config"]["args"]["output_names"].keys())

    # Build evaluation tags from YAML scenarios
    multi_step_def = next(s for s in base_cfg["steps"] if s["name"] == "base.multi")
    sc_list = multi_step_def.get("builder_args", {}).get("scenarios", []) or []
    tags_for_eval = ["base"] + [sc["id"] for sc in sc_list]

    print(f"\n{'=' * 60}")
    print(f"MSCEN RMA {VERSION} | steps={STEPS_TO_RUN} | {len(all_combos)} trial(s)")
    print(f"{'=' * 60}\n")

    for t_idx, combo in enumerate(all_combos, start=1):
        if t_idx < START_TRIAL:
            print(f"[skip] Trial{t_idx} (START_TRIAL={START_TRIAL})")
            continue
        trial_name = f"Trial{t_idx}"
        print(f"\n========= {trial_name} =========")
        print(json.dumps(combo, indent=2))
        mod_cfg = copy.deepcopy(base_cfg)
        mod_cfg["model_builder_config"]["args"]["ndays"] = combo["ndays"]
        fsched = combo["freeze_schedule"]

        trial_dir = SCRIPT_DIR / f"{RUN_ID}_{trial_name}_results"
        trial_dir.mkdir(exist_ok=True)

        multi_prefix_for_eval = None
        last_saved_model = None
        run_step_idx = 0

        for step in mod_cfg["steps"]:
            if step["name"] not in STEPS_TO_RUN:
                continue

            # Output prefix with RUN_ID and trial name
            orig_prefix = step["output_prefix"]
            base_name = Path(orig_prefix).name
            step_out = (OUTPUT_DIR / f"{base_name}_{RUN_ID}_{trial_name}").resolve().as_posix()
            step["output_prefix"] = step_out

            if step.get("save_model_fname") not in (None, "None"):
                step["save_model_fname"] = step_out

            # Auto-chain: use last saved model, or train from scratch if first step
            if step.get("load_model_fname") not in (None, "None"):
                if last_saved_model is not None:
                    step["load_model_fname"] = last_saved_model
                else:
                    step["load_model_fname"] = None
                    # For base.multi, preserve transfer_type so the multi-head
                    # contrastive architecture is built even without pretraining.
                    # Only collapse to direct for single-output steps.
                    if "builder_args" in step and step["name"] != "base.multi":
                        step["builder_args"]["transfer_type"] = None

            for k in ("save_model_fname", "load_model_fname", "output_prefix"):
                step[k] = _expand_outdir(step.get(k))

            if step.get("save_model_fname") not in (None, "None"):
                last_saved_model = step["save_model_fname"]

            bargs = step.get("builder_args", {}) or {}

            # Capture multi prefix for evaluation
            if step["name"] == "base.multi":
                multi_prefix_for_eval = step["output_prefix"]
                bargs["per_scenario_branch"] = combo["per_scenario_branch"]
                bargs["branch_layers"] = copy.deepcopy(combo["branch_layers"])
                bargs["source_weight"] = combo["source_weight"]
                bargs["target_weight"] = combo["target_weight"]
                bargs["contrast_weight"] = combo["contrast_weight"]
                # v12 A/B: scale contrast loss by per-station contrast magnitude
                if combo["use_contrast_scales"]:
                    bargs["contrast_scales"] = dict(CONTRAST_SCALES)
                else:
                    bargs.pop("contrast_scales", None)
                # GRU branches require include_source_branch=True (trunk returns sequences)
                if any(bl.get("type", "").upper() in ("GRU", "LSTM")
                       for bl in combo["branch_layers"]):
                    bargs["include_source_branch"] = True

            # LR/epoch assignment by step name
            if step["name"] == "dsm2_base":
                step.update(init_train_rate=combo["dsm2_init_lr"], main_train_rate=combo["dsm2_main_lr"],
                            init_epochs=combo["dsm2_init_epochs"], main_epochs=combo["dsm2_main_epochs"])
            elif step["name"] == "dsm2.rma":
                step.update(init_train_rate=combo["base_init_lr"], main_train_rate=combo["base_main_lr"],
                            init_epochs=combo["base_init_epochs"], main_epochs=combo["base_main_epochs"])
            elif step["name"] == "base.multi":
                step.update(init_train_rate=combo["multi_init_lr"], main_train_rate=combo["multi_main_lr"],
                            init_epochs=combo["multi_init_epochs"], main_epochs=combo["multi_main_epochs"])

            # Trunk layers with freeze schedule
            trunk_template = copy.deepcopy(combo["trunk_layers"])
            # GRU/LSTM branches need sequences from trunk; Dense/no branch needs final state
            has_recurrent_branch = (
                step["name"] == "base.multi"
                and combo.get("per_scenario_branch", False)
                and any(bl.get("type", "").upper() in ("GRU", "LSTM")
                        for bl in combo["branch_layers"])
            )
            if trunk_template and not has_recurrent_branch:
                trunk_template[-1]["return_sequences"] = False
            freezeN = fsched[run_step_idx] if run_step_idx < len(fsched) else 0
            for j, layer in enumerate(trunk_template):
                layer["trainable"] = bool(j >= freezeN)

            if bargs.get("transfer_type", None) in (None, "None", "direct", "Direct"):
                bargs["base_layers"] = copy.deepcopy(trunk_template)
            else:
                bargs["trunk_layers"] = copy.deepcopy(trunk_template)

            bargs["ndays"] = combo["ndays"]
            step["builder_args"] = bargs

            print(f"INFO  | {trial_name} | {step['name']} | "
                  f"transfer={bargs.get('transfer_type')} | "
                  f"cw={bargs.get('contrast_weight', 'N/A')} | TRAIN")

            step_yaml = trial_dir / f"config_{step['name']}.yml"
            save_yml({"output_dir": str(OUTPUT_DIR),
                       "model_builder_config": mod_cfg["model_builder_config"],
                       "steps": [step]}, step_yaml)

            try:
                process_config(step_yaml, [step["name"]])
            except Exception as e:
                print(f"ERROR | {trial_name} | {step['name']}: {e}")
                traceback.print_exc()
                break

            run_step_idx += 1

        if multi_prefix_for_eval is None:
            print(f"WARN  | {trial_name}: multi prefix not captured, skip evaluation")
            continue

        summary = evaluate_and_plot(trial_dir, stations_in_yaml, tags_for_eval, multi_prefix_for_eval, combo)
        if summary is None:
            continue

        row = {"trial_name": trial_name, **summary, **combo}
        mode = "a" if MASTER_SUMMARY.exists() else "w"
        pd.DataFrame([row]).to_csv(MASTER_SUMMARY, mode=mode,
                                   header=not MASTER_SUMMARY.exists(), index=False)

    if MASTER_SUMMARY.exists():
        df = pd.read_csv(MASTER_SUMMARY)
        print(f"\n========= FINAL SCOREBOARD ({VERSION}) =========")
        if "mean_nse_overall" in df.columns:
            df = df.sort_values("mean_nse_overall", ascending=False)
        print(df.head(20).to_string(index=False))
    else:
        print("No successful trials recorded.")


if __name__ == "__main__":
    main()
