"""
Utilities for analyzing MSTAGE/MSCEN experiment results.
Modular, version-agnostic functions for loading, processing, and plotting.
"""

import ast
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONSTANTS: Color palette and line styles for consistent plotting
# =============================================================================

# 20 distinct colors - colorblind-friendly where possible
CMAP20 = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # muted brown
    "#e377c2",  # muted pink
    "#7f7f7f",  # dark grey
    "#2ca02c",  # dark green
    "#ff9896",  # salmon
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
]

# Line styles for distinguishing groups
LINESTYLES = [
    "-",          # solid
    "--",         # dashed
    ":",          # dotted
    "-.",         # dash-dot
    (0, (3, 1, 1, 1)),      # densely dash-dotted
    (0, (5, 2)),            # loosely dashed
    (0, (1, 1)),            # densely dotted
    (0, (3, 5, 1, 5)),      # dash-dot-dot
    (0, (5, 1)),            # dashed tight
    (0, (3, 1, 1, 1, 1, 1)), # dash-dot-dot-dot
]

# Marker styles for scatter points
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "<", ">"]


def make_style_map(groups: List[str], abbrevs: List[str] = None) -> Dict[str, dict]:
    """
    Generate a style map for a list of groups using CMAP20 and LINESTYLES.

    Args:
        groups: List of group names
        abbrevs: Optional list of abbreviations (default: first 3 chars uppercase)

    Returns:
        Dict mapping group name to {color, ls, abbrev}
    """
    if abbrevs is None:
        abbrevs = [g[:3].upper() for g in groups]

    return {
        g: {
            "color": CMAP20[i % len(CMAP20)],
            "ls": LINESTYLES[i % len(LINESTYLES)],
            "abbrev": abbrevs[i] if i < len(abbrevs) else g[:3].upper()
        }
        for i, g in enumerate(groups)
    }


RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def load_master(run_id: str, base_dir=RUNS_DIR) -> pd.DataFrame:
    df = pd.read_csv(Path(base_dir) / run_id / "master.csv")
    df["run_id"] = run_id
    return df


def load_trials(run_id: str, base_dir=RUNS_DIR) -> pd.DataFrame:
    base = Path(base_dir) / run_id
    pattern = "Trial*"
    trial_dirs = sorted(base.glob(pattern), key=lambda d: int(d.name[5:]))

    dfs = []
    for tdir in trial_dirs:
        trial_num = int(tdir.name[5:])

        fpath = tdir / "metrics.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} missing: trial {trial_num} of {run_id} did not finish evaluation")
        df = pd.read_csv(fpath)
        df["trial"] = trial_num
        df["run_id"] = run_id
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"no {pattern} directories under {base}")
    return pd.concat(dfs, ignore_index=True)


def assign_scenario(df_t: pd.DataFrame) -> pd.DataFrame:
    """Assign scenario column based on model (MSTAGE) or tag (MSCEN).

    MSTAGE trial files use 'model' column with values like 'base.suisun'.
    MSCEN trial files use 'tag' column with values like 'suisun'.
    """
    df_t = df_t.copy()
    if "model" in df_t.columns and df_t["model"].notna().any():
        # MSTAGE: extract scenario from model name (e.g., "base.suisun" → "suisun")
        df_t["scenario"] = df_t["model"].apply(
            lambda x: x.split(".")[-1].replace("-secondary", "") if pd.notna(x) and "." in str(x) else None
        )
    if "tag" in df_t.columns and df_t["tag"].notna().any():
        # MSCEN: use tag directly, excluding "base"
        mask = df_t["tag"].notna() & (df_t["tag"] != "base")
        df_t.loc[mask, "scenario"] = df_t.loc[mask, "tag"]
    return df_t


def is_scenario_row(row) -> bool:
    """Check if a trial row is for scenario prediction (not base/secondary head).

    MSTAGE: model starts with "base." but NOT "-secondary"
    MSCEN: tag is a scenario name (not "base")
    """
    if pd.notna(row.get("model")):
        m = str(row["model"])
        return m.startswith("base.") and "-secondary" not in m
    if pd.notna(row.get("tag")):
        return row["tag"] not in ["base", None]
    return False


def filter_scenario_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter trial dataframe to scenario-only rows (exclude base/secondary heads)."""
    return df[df.apply(is_scenario_row, axis=1)].copy()


def list_runs(base_dir: str = ".") -> List[str]:
    base = Path(base_dir)
    files = base.glob("gridsearch_*_master_results.csv")

    run_ids = []
    for f in files:
        name = f.name
        run_id = name.replace("gridsearch_", "").replace("_master_results.csv", "")
        run_ids.append(run_id)

    return sorted(run_ids)

def parse_run_id(run_id: str) -> dict:
   
    parts = run_id.split("_")

    if "MSTAGE" in parts:
        builder = "MSTAGE"
    elif "MSCEN" in parts:
        builder = "MSCEN"
    else:
        builder = None

    version = None
    for p in parts:
        if p.startswith("v") and any(c.isdigit() for c in p):
            version = p
            break

    scenario = None
    scenarios = ["suisun", "slr", "cache", "franks", "ft"]
    for p in parts:
        if p.lower() in scenarios:
            scenario = p.lower()
            break

    return {"builder": builder, "version": version, "scenario": scenario}


def _parse_literal(val):
    """Parse a list column: JSON (gridsearch.py) or Python repr (old drivers)."""
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return ast.literal_eval(val)


def expand_layers(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    df = df.copy()

    if col is None:
        found = [c for c in ("trunk_layers", "feature_layers", "layers") if c in df.columns]
        if not found:
            raise KeyError("no layer column (trunk_layers, feature_layers or layers) in frame")
        col = found[0]

    def parse_layers(val):
        if pd.isna(val):
            return {}
        result = {}
        for i, layer in enumerate(_parse_literal(val), start=1):
            result[f"layer{i}_type"] = layer.get("type")
            result[f"layer{i}_units"] = layer.get("units")
            result[f"layer{i}_trainable"] = layer.get("trainable")
        return result

    expanded = df[col].apply(parse_layers).apply(pd.Series)
    return pd.concat([df, expanded], axis=1)


def expand_freeze(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    df = df.copy()

    if col is None:
        found = [c for c in ("freeze_schedule", "freeze") if c in df.columns]
        if not found:
            raise KeyError("no freeze column (freeze_schedule or freeze) in frame")
        col = found[0]

    def parse_freeze(val):
        if pd.isna(val):
            return {}
        return {f"freeze_step{i+1}": v for i, v in enumerate(_parse_literal(val))}

    expanded = df[col].apply(parse_freeze).apply(pd.Series)
    return pd.concat([df, expanded], axis=1)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = expand_layers(df)
    df = expand_freeze(df)

    if "run_id" in df.columns:
        parsed = df["run_id"].apply(parse_run_id).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)

    return df

def plot_comparison(
    data: Dict[str, pd.DataFrame],
    metric: str,
    title: str = None,
    ylabel: str = "NSE",
    figsize: Tuple[int, int] = (10, 5),
    show_values: bool = True,
    ylim: Tuple[float, float] = None) -> plt.Figure:

    labels = list(data.keys())
    means = [df[metric].mean() for df in data.values()]
    stds = [df[metric].std() for df in data.values()]

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(labels))

    bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{metric} comparison")
    ax.grid(axis="y", alpha=0.3)

    if ylim:
        ax.set_ylim(ylim)

    if show_values:
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.annotate(f"{m:.4f}", (i, m + s + 0.002), ha="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_parallel_lines(
    df: pd.DataFrame,
    x_col: str = "station",
    y_col: str = "nse",
    group_col: str = "run_id",
    line_col: str = "trial",
    title: str = None,
    ylabel: str = "NSE",
    figsize: Tuple[int, int] = (14, 5),
    stations_to_ignore: List[str] = None,
    alpha: float = 0.7,
    show_legend: bool = True) -> plt.Figure:

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    stations_to_ignore = stations_to_ignore or []
    df_plot = df[~df[x_col].isin(stations_to_ignore)].copy()

    x_vals = sorted(df_plot[x_col].unique())
    x_idx = range(len(x_vals))

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette for groups
    groups = df_plot[group_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = dict(zip(groups, colors))

    for (group, line_id), sub in df_plot.groupby([group_col, line_col]):
        y = [sub[sub[x_col] == s][y_col].mean() if s in sub[x_col].values else np.nan
             for s in x_vals]
        label = f"{group} T{line_id}" if line_col == "trial" else f"{group}_{line_id}"
        ax.plot(x_idx, y, marker="o", label=label, alpha=alpha,
                color=color_map[group], linewidth=1.5)

    ax.set_xticks(list(x_idx))
    ax.set_xticklabels(x_vals, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} Across {x_col.capitalize()}s")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

    plt.tight_layout()
    return fig


def pct_change_table(df: pd.DataFrame, metrics: List[str], archs: List[str], baseline_arch: str, arch_col: str = "arch") -> pd.DataFrame:
    """Compute mean values and percent change from baseline for each architecture."""
    rows = []
    for arch in archs:
        row = {"Architecture": arch}
        for m in metrics:
            val = df[df[arch_col] == arch][m].mean()
            baseline = df[df[arch_col] == baseline_arch][m].mean()
            pct = ((val - baseline) / baseline) * 100
            row[m] = f"{val:.4f}"
            row[f"{m}_pct"] = f"{pct:+.2f}%" if arch != baseline_arch else "baseline"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_grouped_bars(
    df: pd.DataFrame,
    nse_col: str,
    group_col: str = "scenario",
    hue_col: str = "arch",
    title: str = None,
    groups: List[str] = None,
    hues: List[str] = None,
    colors: List[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    ylim: Tuple[float, float] = (0.8, 1.0),
) -> plt.Figure:
    """Bar chart with groups on x-axis, hue as colored bars.

    Args:
        df: DataFrame with data
        nse_col: Column for y-axis values
        group_col: Column for x-axis grouping (default: scenario)
        hue_col: Column for bar colors (default: arch)
        title: Plot title
        groups: List of group values (auto-detected if None)
        hues: List of hue values (auto-detected if None)
        colors: List of colors for hues
        figsize: Figure size
        ylim: Y-axis limits

    Returns:
        matplotlib Figure
    """
    groups = groups or sorted(df[group_col].dropna().unique())
    hues = hues or sorted(df[hue_col].dropna().unique())
    colors = colors or ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    width = 0.8 / len(hues)

    for i, (hue, color) in enumerate(zip(hues, colors)):
        means = []
        stds = []
        for grp in groups:
            subset = df[(df[hue_col] == hue) & (df[group_col] == grp)]
            means.append(subset[nse_col].mean() if len(subset) > 0 else np.nan)
            stds.append(subset[nse_col].std() if len(subset) > 0 else 0)

        offset = (i - len(hues)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=hue,
               color=color, capsize=3, alpha=0.85, edgecolor="black")

    ax.set_xlabel(group_col.capitalize())
    ax.set_ylabel("NSE")
    ax.set_title(title or f"{nse_col} by {group_col}")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(title=hue_col.capitalize())
    ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_parallel_topN(
    df_trials: pd.DataFrame,
    scenario: str,
    group_col: str = "arch",
    station_col: str = "station",
    nse_col: str = "nse",
    trial_col: str = "trial",
    top_n: int = 3,
    metric_col: str = None,
    is_min_better: bool = False,
    style_map: Dict[str, dict] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = None,
    stations_order: List[str] = None,
) -> plt.Figure:
    """
    Parallel line plot showing top N trials per group, with different line styles per group.

    Args:
        df_trials: Station-level trial data with columns for group, station, nse, trial
        scenario: Scenario to filter (e.g., "suisun")
        group_col: Column for grouping (e.g., "arch" for V4/V5/V6)
        station_col: Column with station names
        nse_col: Column with NSE values
        trial_col: Column with trial identifier
        top_n: Number of top trials to select per group
        metric_col: Column to rank by (default: mean of nse_col per trial)
        is_min_better: If True, lower metric is better (e.g., RMSE)
        style_map: Dict mapping group values to {color, ls, abbrev}
        figsize: Figure size
        title: Plot title
        stations_order: Optional ordered list of stations for x-axis

    Returns:
        matplotlib Figure
    """
    from matplotlib.lines import Line2D

    # Filter by scenario if present
    df = df_trials.copy()
    if "scenario" in df.columns and scenario:
        df = df[df["scenario"] == scenario]

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for scenario={scenario}", ha="center", va="center")
        return fig

    # Default style map using global CMAP20 and LINESTYLES
    if style_map is None:
        groups = list(df[group_col].unique())
        style_map = make_style_map(groups)

    # Compute mean NSE per trial per group
    trial_means = df.groupby([group_col, trial_col])[nse_col].mean().reset_index()
    trial_means.columns = [group_col, trial_col, "mean_nse"]

    # Select top N trials per group
    selected_trials = []
    for grp in df[group_col].unique():
        grp_trials = trial_means[trial_means[group_col] == grp]
        grp_sorted = grp_trials.sort_values("mean_nse", ascending=is_min_better)
        selected_trials.append(grp_sorted.head(top_n))

    selected = pd.concat(selected_trials, ignore_index=True)

    # Filter df to only selected trials
    df_plot = df.merge(selected[[group_col, trial_col]], on=[group_col, trial_col])

    # Determine station order
    if stations_order is None:
        stations_order = sorted(df_plot[station_col].unique())

    # Pivot to wide format: rows = (group, trial), columns = stations
    wide_data = []
    for (grp, trial), sub in df_plot.groupby([group_col, trial_col]):
        row = {"group": grp, "trial": trial}
        for _, r in sub.iterrows():
            row[r[station_col]] = r[nse_col]
        wide_data.append(row)

    df_wide = pd.DataFrame(wide_data)

    # Sort by mean NSE (best first)
    df_wide["_mean"] = df_wide[stations_order].mean(axis=1)
    df_wide = df_wide.sort_values("_mean", ascending=is_min_better).reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    xvals = np.arange(len(stations_order))

    for rank, (_, row) in enumerate(df_wide.iterrows(), start=1):
        grp = row["group"]
        sty = style_map.get(grp, {"color": "grey", "ls": "-", "abbrev": "?"})
        line_label = f"{rank}-{sty['abbrev']}"
        yvals = [row.get(s, np.nan) for s in stations_order]
        ax.plot(xvals, yvals, color=sty["color"], linestyle=sty["ls"],
                linewidth=2, alpha=0.8, label=line_label)

    # Mark best performer per station
    for j, station in enumerate(stations_order):
        if station not in df_wide.columns:
            continue
        vals = df_wide[station]
        if vals.notna().sum() == 0:
            continue
        idx_best = vals.idxmin() if is_min_better else vals.idxmax()
        best_row = df_wide.loc[idx_best]
        sty_best = style_map.get(best_row["group"], {"color": "black"})
        ax.scatter(j, best_row[station], s=140, marker="o",
                  edgecolor="k", linewidth=1.5, zorder=5, color=sty_best["color"])

    ax.set_xticks(xvals)
    ax.set_xticklabels(stations_order, rotation=30, ha="right")
    ax.set_ylabel("NSE")
    ax.set_title(title or f"Top {top_n} Trials per Group — {scenario.capitalize() if scenario else 'All'}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend 1: Trial ranks
    handles_trials, labels_trials = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles_trials, labels_trials, title=f"Top-{top_n} (rank-abbr)",
                    bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)
    ax.add_artist(leg1)

    # Legend 2: Group styles with counts
    counts = df_wide["group"].value_counts().to_dict()
    handles_grp, labels_grp = [], []
    for grp, sty in style_map.items():
        if grp not in counts:
            continue
        h = Line2D([0], [0], color=sty["color"], linestyle=sty["ls"], linewidth=3)
        handles_grp.append(h)
        labels_grp.append(f"{grp} ({counts[grp]})")

    ax.legend(handles_grp, labels_grp, title="Group (count)",
             bbox_to_anchor=(1.02, 0.5), loc="upper left", frameon=False, fontsize=9)

    plt.tight_layout()
    return fig
