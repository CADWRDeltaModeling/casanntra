#!/usr/bin/env python
# check_builder_architecture.py
# -----------------------------------------------------------
# Verifies
#   • layer order / type / units
#   • trainable flag (freezing)
#   • heads & outputs for contrastive mode
#   • prints   Expected | Actual | PASS/FAIL  for every check
# -----------------------------------------------------------

import sys, numpy as np, tensorflow as tf
from itertools import count

# ---------- import path (adjust if needed) -----------------
from casanntra.multi_stage_model_builder import MultiStageModelBuilder

# ---------- dummy IO spec (light-weight) -------------------
INPUTS  = ["feat1", "feat2"]
OUTPUTS = {"dummy": 100.0}

# ---------- YAML-like scenarios ----------------------------
SCENARIOS = {
    "A_vanilla": {
        "transfer_type": "direct",
        "feature_layers": [
            {"type": "LSTM", "units": 64},
            {"type": "LSTM", "units": 32},
        ],
    },
    "B_first_frozen": {
        "transfer_type": "direct",
        "feature_layers": [
            {"type": "LSTM", "units": 64, "trainable": False, "name": "enc1"},
            {"type": "LSTM", "units": 32},
        ],
    },
    "C_mixed_gru_dense": {
        "transfer_type": "direct",
        "feature_layers": [
            {"type": "GRU",   "units": 128, "trainable": False},
            {"type": "GRU",   "units": 64},
            {"type": "Dense", "units": 32},
        ],
    },
    "D_deep_gru_freeze3": {
        "transfer_type": "direct",
        "feature_layers": [
            {"type": "GRU", "units": 256, "trainable": False},
            {"type": "GRU", "units": 128, "trainable": False},
            {"type": "GRU", "units":  64, "trainable": False},
            {"type": "GRU", "units":  32},
        ],
    },
    "E_contrastive_two_heads": {
        "transfer_type": "contrastive",
        "contrast_weight": 0.8,
        "feature_layers": [
            {"type": "LSTM", "units": 64, "trainable": False},
            {"type": "LSTM", "units": 32, "trainable": False},
        ],
        "heads": {
            "target": [{"type": "Dense", "units": 18, "activation": "elu"}],
            "source": [{"type": "Dense", "units": 18, "activation": "elu"}],
        },
    },
}

TYPE_MAP = {
    "lstm":  tf.keras.layers.LSTM,
    "gru":   tf.keras.layers.GRU,
    "dense": tf.keras.layers.Dense,
}

# ---------- helpers ----------------------------------------
def _noop_prepro(self, inputs, _df):
    """Skip domain-specific preprocessing in unit test."""
    return list(inputs.values())

def _print_check(label, expected, actual):
    ok = expected == actual
    print(f"   {label:<25} expected={expected!r:>15} | actual={actual!r:<15} → {'PASS' if ok else 'FAIL'}")
    return ok

# ---------- run checks -------------------------------------
test_counter = count(1)
overall_success = True

for case, cfg in SCENARIOS.items():
    num = next(test_counter)
    print(f"\n────────  TEST {num}: {case}  ────────")

    # build
    mb = MultiStageModelBuilder(INPUTS, OUTPUTS, ndays=90)
    mb.set_builder_args(cfg)
    # monkeypatch equivalent: just overwrite the bound method
    mb.prepro_layers = _noop_prepro.__get__(mb)
    inputs = {n: tf.keras.Input(shape=(90,), name=n) for n in INPUTS}
    model  = mb.build_model(inputs, None)

    ok = True
    # -- feature layers ------------------------------------------------------
    for idx, spec in enumerate(cfg["feature_layers"]):
        lname = spec.get("name", f"feature_{idx+1}")
        layer = model.get_layer(lname)

        ok &= _print_check(f"layer{idx+1}-type",
                           TYPE_MAP[spec["type"].lower()].__name__,
                           layer.__class__.__name__)

        if hasattr(layer, "units"):
            ok &= _print_check(f"layer{idx+1}-units",
                               spec["units"],
                               layer.units)

        ok &= _print_check(f"layer{idx+1}-trainable",
                           spec.get("trainable", True),
                           layer.trainable)

    # -- output heads --------------------------------------------------------
    exp_heads = {"out_unscaled"} if cfg["transfer_type"] != "contrastive" else {
        "out_target_unscaled", "out_source_unscaled", "out_contrast_unscaled"}
    ok &= _print_check("output-heads-set", exp_heads, set(model.output_names))

    # -- parameter counts (informational) ------------------------------------
    tp = int(np.sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    fp = int(np.sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    print(f"   param-counts            trainable={tp:,}  frozen={fp:,}")

    print(f"✔︎  OVERALL {'PASS' if ok else 'FAIL'} for {case}")
    overall_success &= ok

# ---------- summary ---------------------------------------------------------
print("\n══════════════════════════════════════════════════════════════")
print("ALL TESTS", "PASSED" if overall_success else "FAILED")
sys.exit(0 if overall_success else 1)