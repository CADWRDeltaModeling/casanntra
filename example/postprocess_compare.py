"""
Post-processing to compare MultiStage runs (v1, v1.2) with per-station NSE.
Configurable knobs at the top:
- OUTPUT_DIR: where plots go.
- RUN_CONFIGS: master CSVs, run_ids, head/tag names.
- TOP_N: number of top trials per approach (by mean_nse_overall) if TRIAL_SELECTIONS is empty.
- SCENARIOS_TO_COMPARE: which scenarios to plot (e.g., ["base","suisun","slr"]).
- TRIAL_SELECTIONS: explicit trial names per approach (optional).
"""
from pathlib import Path
import glob
from typing import List, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "postprocess_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RUN_CONFIGS: Dict[str, Dict] = {
    "MSTAGE v1": {
        "master": "gridsearch_v1_MSTAGE_master_results.csv",
        "run_id": "v1_MSTAGE",
        "model_map": {"base": "base.suisun-secondary", "target": "base.suisun"},
        "tag_map": {"base": "base", "target": "suisun"},
    },
    "MSTAGE v1.2": {
        "master": "gridsearch_v1.2_MSTAGE_master_results.csv",
        "run_id": "v1.2_MSTAGE",
        "model_map": {"base": "base.suisun-secondary", "target": "base.suisun"},
        "tag_map": {"base": "base", "target": "suisun"},
    },
}

# pick this many top trials per approach when TRIAL_SELECTIONS is empty
TOP_N = 3

# Scenarios to compare (names must match keys in model_map/tag_map)
SCENARIOS_TO_COMPARE = ["base", "target"]

# Optional display labels for scenarios (e.g., map "target" -> "suisun")
SCENARIO_LABELS: Dict[str, str] = {
    "target": "suisun",
}

# Stations to ignore in the parallel line plots
STATIONS_TO_IGNORE = {"nsl2", "rsl"}

# Explicit trial selections (optional). If empty, picks top TOP_N by mean_nse_overall.
TRIAL_SELECTIONS: Dict[str, List[str]] = {
    # "MSTAGE v1": ["Trial1"],
    # "MSTAGE v1.2": ["Trial2"],
    }


def load_master(path: Path, approach: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "trial_name" not in df.columns:
        raise ValueError(f"{path} missing trial_name")
    df["approach"] = approach
    return df


def pick_top_trials(df: pd.DataFrame, metric: str = "mean_nse_overall", top_n: int = 3) -> pd.DataFrame:
    return df.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)


def find_trial_dirs(run_id: str, base_dir: Path) -> List[Path]:
    pattern = str(base_dir / f"{run_id}_Trial*results")
    return sorted(Path(p) for p in glob.glob(pattern))


def load_trial_long_metrics(trial_dir: Path,
                            model_map: Dict[str, str],
                            tag_map: Dict[str, str]) -> Optional[pd.DataFrame]:
    csv_path = trial_dir / "trial_evaluation_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "station" not in df.columns or "nse" not in df.columns:
        return None
    rows = []
    if "model" in df.columns:
        for scenario, model_name in model_map.items():
            if scenario not in SCENARIOS_TO_COMPARE:
                continue
            sub = df[df["model"] == model_name]
            for _, r in sub.iterrows():
                rows.append({"trial": trial_dir.name, "scenario": scenario, "station": r["station"], "nse": r["nse"]})
    elif "tag" in df.columns:
        for scenario, tag_val in tag_map.items():
            if scenario not in SCENARIOS_TO_COMPARE:
                continue
            sub = df[df["tag"] == tag_val]
            for _, r in sub.iterrows():
                rows.append({"trial": trial_dir.name, "scenario": scenario, "station": r["station"], "nse": r["nse"]})
    if not rows:
        return None
    return pd.DataFrame(rows)


def collect_station_metrics(run_id: str,
                            approach: str,
                            model_map: Dict[str, str],
                            tag_map: Dict[str, str],
                            base_dir: Path,
                            top_trials: Optional[List[str]] = None) -> pd.DataFrame:
    out = []
    for tdir in find_trial_dirs(run_id, base_dir):
        if top_trials and all(sel not in tdir.name for sel in top_trials):
            continue
        df_long = load_trial_long_metrics(tdir, model_map=model_map, tag_map=tag_map)
        if df_long is None:
            continue
        df_long["approach"] = approach
        out.append(df_long)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def parallel_line_plot(df: pd.DataFrame, scenario: str, title: str, outfile: Path):
    """
    Plot NSE across stations for each trial (lines) per approach for a single scenario.
    """
    if df.empty:
        print(f"[plot] no data to plot for scenario={scenario}")
        return
    label = SCENARIO_LABELS.get(scenario, scenario)
    stations = sorted(s for s in df["station"].unique() if s not in STATIONS_TO_IGNORE)
    x = range(len(stations))
    fig, ax = plt.subplots(figsize=(14, 5))
    for (trial, approach), sub in df[df["scenario"] == scenario].groupby(["trial", "approach"]):
        y = [sub[sub["station"] == s]["nse"].mean() for s in stations]
        label = f"{trial} ({approach[:7]})"
        ax.plot(x, y, marker="o", label=label, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stations, rotation=30, ha="right")
    ax.set_ylabel(f"{label} NSE")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {outfile}")


def plot_nse_by_freeze(df_master: pd.DataFrame, outfile: Path):
    if df_master.empty or "freeze_schedule" not in df_master.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    df_master["freeze_schedule"] = df_master["freeze_schedule"].astype(str)
    for approach, sub in df_master.groupby("approach"):
        ax.scatter(sub["freeze_schedule"], sub["mean_nse_overall"], label=approach, alpha=0.6)
    ax.set_xlabel("freeze_schedule")
    ax.set_ylabel("mean_nse_overall")
    ax.set_title("NSE vs freeze schedule")
    plt.xticks(rotation=20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {outfile}")


def main():
    frames = []
    for approach, cfg in RUN_CONFIGS.items():
        path = BASE_DIR / cfg["master"]
        if not path.exists():
            print(f"[warn] missing master: {path}")
            continue
        frames.append(load_master(path, approach))
    if not frames:
        print("[warn] no master files loaded")
        return
    df_master = pd.concat(frames, ignore_index=True)

    # Choose trials
    tops = []
    for approach, sub in df_master.groupby("approach"):
        if TRIAL_SELECTIONS.get(approach):
            qs = sub[sub["trial_name"].isin(TRIAL_SELECTIONS[approach])]
            tops.append(qs)
        else:
            tops.append(pick_top_trials(sub, metric="mean_nse_overall", top_n=TOP_N))
    df_top = pd.concat(tops, ignore_index=True)
    print("[info] Selected trials:")
    print(df_top[["approach", "trial_name", "mean_nse_overall", "freeze_schedule"]])

    # Collect station metrics
    rows = []
    for _, r in df_top.iterrows():
        cfg = RUN_CONFIGS[r["approach"]]
        df_trials = collect_station_metrics(run_id=cfg["run_id"],
                                            approach=r["approach"],
                                            model_map=cfg["model_map"],
                                            tag_map=cfg["tag_map"],
                                            base_dir=BASE_DIR,
                                            top_trials=[r["trial_name"]])
        rows.append(df_trials)
    collected = [d for d in rows if d is not None and not d.empty]
    if not collected:
        print("[warn] no station-level data collected")
        return
    df_station = pd.concat(collected, ignore_index=True)

    for scenario in SCENARIOS_TO_COMPARE:
        out_png = OUTPUT_DIR / f"parallel_lines_{scenario}.png"
        label = SCENARIO_LABELS.get(scenario, scenario)
        parallel_line_plot(df_station, scenario=scenario,
                           title=f"{label.capitalize()} NSE Across Stations — Top {TOP_N} Trials",
                           outfile=out_png)

    plot_nse_by_freeze(df_master, outfile=OUTPUT_DIR / "nse_vs_freeze.png")


if __name__ == "__main__":
    main()
