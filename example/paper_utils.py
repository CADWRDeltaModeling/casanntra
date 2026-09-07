"""Utility functions for paper_findings.ipynb."""

import numpy as np
import pandas as pd
import metrics
from metrics import compute_metrics, nse


def load_and_merge(prefix, tag=None):
    """tag=None for single-head MSTAGE direct models."""
    prefix = str(prefix)
    ref_csv, ann_csv = metrics.xvalid_pair_ms(prefix, tag) if tag else metrics.xvalid_pair(prefix, "direct")
    return metrics.load_and_merge(ref_csv, ann_csv)










def build_contrast_df(df_base, df_scenario, stations):
    """Merge base and scenario frames, compute true/pred contrast for each station."""
    merged = pd.merge(
        df_base, df_scenario, on=["datetime", "case"], how="inner",
        suffixes=("_base", "_scen")
    )
    result = merged[["datetime", "case"]].copy()
    for st in stations:
        b_ref, b_pred = f"{st}_base", f"{st}_pred_base"
        s_ref, s_pred = f"{st}_scen", f"{st}_pred_scen"
        if all(c in merged.columns for c in [b_ref, b_pred, s_ref, s_pred]):
            result[f"{st}_true_contrast"] = merged[s_ref].values - merged[b_ref].values
            result[f"{st}_pred_contrast"] = merged[s_pred].values - merged[b_pred].values
            result[f"{st}_ref_base"] = merged[b_ref].values
            result[f"{st}_ref_scen"] = merged[s_ref].values
    return result


def signal_diagnostics(contrast_df, stations, scenario, label):
    """Compute signal magnitude diagnostics for each station."""
    rows = []
    for st in stations:
        tc = f"{st}_true_contrast"
        rb = f"{st}_ref_base"
        if tc not in contrast_df.columns or rb not in contrast_df.columns:
            continue
        true_c = contrast_df[tc].dropna()
        base_vals = contrast_df[rb].dropna()
        rows.append({
            "scenario": scenario, "label": label, "station": st,
            "contrast_mean": round(float(true_c.mean()), 2),
            "contrast_std": round(float(true_c.std()), 2),
            "contrast_abs_mean": round(float(true_c.abs().mean()), 2),
            "base_std": round(float(base_vals.std()), 2),
            "signal_ratio": round(float(true_c.std() / base_vals.std()), 4) if base_vals.std() > 0 else np.nan,
        })
    return rows


def error_correlation_analysis(base_df, scen_df, station, label):
    """Compute per-timestep error correlation between base and scenario predictions."""
    m = pd.merge(base_df, scen_df, on=["datetime", "case"], how="inner", suffixes=("_base", "_scen"))
    err_base = m[f"{station}_pred_base"] - m[f"{station}_base"]
    err_scen = m[f"{station}_pred_scen"] - m[f"{station}_scen"]
    ok = err_base.notna() & err_scen.notna()
    r = float(np.corrcoef(err_base[ok], err_scen[ok])[0, 1])
    return {"label": label, "station": station,
            "error_corr": round(r, 4),
            "err_base_std": round(float(err_base.std()), 2),
            "err_scen_std": round(float(err_scen.std()), 2)}
