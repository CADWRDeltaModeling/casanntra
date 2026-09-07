"""Shared evaluation helpers: metrics, xvalid filename conventions, ref/pred merge."""

from pathlib import Path
import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred):
    mask = (~pd.isnull(y_true)) & (~pd.isnull(y_pred))
    if mask.sum() < 2:
        return {"nse": np.nan, "pearson_r": np.nan, "mae": np.nan, "rmse": np.nan}
    yt, yp = np.array(y_true[mask], dtype=float), np.array(y_pred[mask], dtype=float)
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    nse_val = float(1.0 - np.sum((yt - yp) ** 2) / denom) if denom > 0 else np.nan
    r = float(np.corrcoef(yt, yp)[0, 1])
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"nse": nse_val, "pearson_r": r, "mae": mae, "rmse": rmse}


def nse(y_true, y_pred):
    return compute_metrics(y_true, y_pred)["nse"]


def xvalid_pair(prefix, transfer_type, role="primary"):
    """MSTAGE xvalid (ref, pred) filenames written by xvalid_fit_multi."""
    if role == "secondary":
        if transfer_type not in ("contrastive", "multi-direct"):
            raise ValueError(f"no secondary head for transfer_type={transfer_type!r}")
        return f"{prefix}_xvalid_ref_out_secondary_unscaled.csv", f"{prefix}_xvalid_1.csv"
    if transfer_type in ("contrastive", "multi-direct"):
        return f"{prefix}_xvalid_ref_out_unscaled.csv", f"{prefix}_xvalid_0.csv"
    return f"{prefix}_xvalid_ref_out_unscaled.csv", f"{prefix}_xvalid.csv"


def xvalid_pair_ms(prefix, tag):
    """MSCEN xvalid (ref, pred) filenames for one scenario tag."""
    return f"{prefix}_xvalid_ref_out_{tag}_unscaled.csv", f"{prefix}_xvalid_{tag}.csv"


def load_and_merge(ref_csv, ann_csv):
    missing = [p for p in (ref_csv, ann_csv) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"expected xvalid files missing: {missing}")
    df_ref = pd.read_csv(ref_csv, parse_dates=["datetime"])
    df_ann = pd.read_csv(ann_csv, parse_dates=["datetime"])
    df = pd.merge(df_ref, df_ann, on=["datetime", "case"], how="inner", suffixes=("", "_pred"))
    return df.sort_values(["case", "datetime"]).reset_index(drop=True)
