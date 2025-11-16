"""
Deep smoke-test that MultiStage (contrastive) and MultiScenario share identical
data plumbing and trunk/head wiring when configured for the same Suisun run.

This script is deterministic: it fabricates synthetic base/scenario datasets,
feeds them through each builder’s pool/lag/fold utilities, and compares every
intermediate result. It also builds both models and verifies that the shared
layers (preprocessing + GRU trunk) are configured identically, along with the
expected output shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import yaml

from casanntra.multi_stage_model_builder import MultiStageModelBuilder
from casanntra.multi_scenario_model_builder import MultiScenarioModelBuilder


# --------------------------------------------------------------------------- #
# Helper utilities


def _load_core_args(config_path: str) -> Tuple[List[str], Dict[str, float], int]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    args = cfg["model_builder_config"]["args"]
    return args["input_names"], args["output_names"], int(args.get("ndays", 105))


def _make_case_block(case_id: int, start_ts: pd.Timestamp, ndays: int, input_names: List[str], output_names: Dict[str, float], feature_offset: float) -> pd.DataFrame:
    """Create a deterministic slice of data for one case."""
    dates = pd.date_range(start_ts, periods=ndays, freq="D")
    data = {"datetime": dates, "case": case_id}
    for idx, feat in enumerate(input_names):
        data[feat] = feature_offset + idx + np.arange(ndays, dtype=float)
    for idx, out in enumerate(output_names):
        data[out] = feature_offset * 10 + idx + np.linspace(0.0, 1.0, ndays)
    return pd.DataFrame(data)


def _build_synthetic_frames(input_names: List[str], output_names: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fabricate deterministic base + scenario datasets:
    - Base has 2 cases across contiguous blocks in early 2021.
    - Scenario shares those blocks but also appends extra (non-overlapping) rows
      to mimic real-world mismatches.
    """
    base_parts = []
    scenario_parts = []
    start_dates = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]
    for case_id, start in enumerate(start_dates, start=1):
        base = _make_case_block(case_id, start, ndays=6, input_names=input_names, output_names=output_names, feature_offset=case_id * 1.0)
        base_parts.append(base)
        overlap = _make_case_block(case_id, start, ndays=6, input_names=input_names, output_names=output_names, feature_offset=case_id * 10.0)
        extra = _make_case_block(case_id, start + pd.Timedelta(days=6), ndays=2, input_names=input_names, output_names=output_names, feature_offset=case_id * 20.0)
        scenario_parts.append(pd.concat([overlap, extra], ignore_index=True))
    base_df = pd.concat(base_parts, ignore_index=True)
    scenario_df = pd.concat(scenario_parts, ignore_index=True)
    return base_df, scenario_df


def _normalize(df: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    return df.loc[:, ordered_cols].sort_values(["case", "datetime"]).reset_index(drop=True)


def _assert_same_dataframe(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: List[str], label: str):
    norm_a = _normalize(df_a, cols)
    norm_b = _normalize(df_b, cols)
    assert_frame_equal(norm_a, norm_b, check_dtype=False), f"{label} mismatch"


def _predict_shapes(ann, feed, out_keys):
    preds = ann.predict(feed)
    if not isinstance(preds, dict):
        if hasattr(ann, "output_names"):
            preds = {k: v for k, v in zip(ann.output_names, preds)}
        else:
            raise RuntimeError("Unexpected non-dict predictions and no output_names present")
    return {k: preds[k].shape for k in out_keys}


def _snapshot_layers(model, layer_names: List[str]) -> Dict[str, Dict]:
    return {name: model.get_layer(name).get_config() for name in layer_names}


# --------------------------------------------------------------------------- #
# Main test routine


def main():
    cfg_path = Path(__file__).resolve().parent / "transfer_config_multi.yaml"
    input_names, output_names, ndays = _load_core_args(str(cfg_path))
    outdim = len(output_names)

    feature_spec = [
        {"type": "GRU", "units": 16, "name": "lay1", "return_sequences": True, "trainable": True},
        {"type": "GRU", "units": 16, "name": "lay2", "return_sequences": False, "trainable": True},
    ]

    # Instantiate builders with matching arguments
    ms = MultiStageModelBuilder(input_names, output_names, ndays)
    ms.set_builder_args({"transfer_type": "contrastive", "feature_layers": feature_spec, "contrast_weight": 0.5})

    msc = MultiScenarioModelBuilder(input_names, output_names, ndays)
    msc.set_builder_args({
        "transfer_type": "contrastive",
        "trunk_layers": feature_spec,
        "per_scenario_branch": False,
        "branch_layers": [],
        "contrast_weight": 0.5,
        "scenarios": [{"id": "suisun"}],
        "include_source_branch": True,
    })

    # --- Data parity checks -------------------------------------------------
    base_df, scenario_df = _build_synthetic_frames(input_names, output_names)

    aligned_ms = ms.pool_and_align_cases([base_df, scenario_df])
    aligned_msc = msc.pool_and_align_cases([base_df, scenario_df])
    ms_base, ms_scenario = aligned_ms
    msc_base, msc_scenario = aligned_msc

    base_cols = ["datetime", "case", *input_names, *output_names]
    _assert_same_dataframe(ms_base, msc_base, base_cols, "Base alignment")
    _assert_same_dataframe(ms_scenario, msc_scenario, base_cols, "Scenario alignment")

    ms_lag = ms.calc_antecedent_preserve_cases(ms_base)
    msc_lag = msc.calc_antecedent_preserve_cases(msc_base)
    _assert_same_dataframe(ms_lag, msc_lag, ms_lag.columns.tolist(), "Antecedent inputs")

    ms_in, ms_out = ms.xvalid_time_folds(ms_base, target_fold_len="3d", split_in_out=True)
    msc_in, msc_out = msc.xvalid_time_folds(msc_base, target_fold_len="3d", split_in_out=True)
    _assert_same_dataframe(ms_in, msc_in, ms_in.columns.tolist(), "Folded inputs")
    _assert_same_dataframe(ms_out, msc_out, ms_out.columns.tolist(), "Folded outputs")

    print("OK: data pipeline parity (alignment, lags, folds)")

    # --- Architecture parity checks ----------------------------------------
    dummy_df = ms_base
    ms_inputs = ms.input_layers()
    msc_inputs = msc.input_layers()
    ms_model = ms.build_model(ms_inputs, dummy_df)
    msc_model = msc.build_model(msc_inputs, dummy_df)

    shared_layer_names = ["stacked", "lay1", "lay2"]
    ms_snapshot = _snapshot_layers(ms_model, shared_layer_names)
    msc_snapshot = _snapshot_layers(msc_model, shared_layer_names)
    assert ms_snapshot == msc_snapshot, "Shared layer configs differ"
    assert ms_model.count_params() == msc_model.count_params(), "Parameter counts differ"

    ms_heads = sorted([l.name for l in ms_model.layers if l.name in ("target_scaled", "source_scaled")])
    msc_heads = sorted([l.name for l in msc_model.layers if l.name in ("head_suisun_scaled", "head_base_scaled")])
    print("MultiStage heads:", ms_heads)
    print("MultiScenario heads:", msc_heads)

    ms_expect = ["out_target_unscaled", "out_source_unscaled", "out_contrast_unscaled"]
    msc_expect = ["out_suisun_unscaled", "out_base_unscaled", "out_suisun_contrast_unscaled"]

    N = 16
    feed = {name: np.random.randn(N, ndays).astype(np.float32) for name in input_names}
    ms_shapes = _predict_shapes(ms_model, feed, ms_expect)
    msc_shapes = _predict_shapes(msc_model, feed, msc_expect)
    for shape in ms_shapes.values():
        assert shape == (N, outdim), f"Unexpected MultiStage shape {shape}"
    for shape in msc_shapes.values():
        assert shape == (N, outdim), f"Unexpected MultiScenario shape {shape}"

    print("OK: architecture parity (shared trunk + output shapes)")


if __name__ == "__main__":
    main()
