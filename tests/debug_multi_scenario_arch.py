"""
Debug/inspection script for multi-scenario pipeline.

Builds the model for each stage in the YAML, compiles it (0 epochs),
draws the architecture, prints loss dictionaries & trainable flags,
and generates an additional, colored loss-graph with λ/μ weights and Σ.
"""

import argparse
from typing import Optional
import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

try:
    import pydot  
except Exception:
    pydot = None

from casanntra.staged_learning import read_config, model_builder_from_config
from casanntra.read_data import read_data  
from casanntra.single_or_list import single_or_list

try:
    from casanntra.multi_scenario_model_builder import MultiScenarioModelBuilder  
except Exception:
    pass


GLOBAL_OUTPUT_DIR = "."         
STRICT_LOSS_CHECK = False       

def _make_dummy_df(input_names, output_names, ndays, nrows=None, ncases=1):
    if nrows is None:
        nrows = max(ndays + 5, 20)
    base_time = datetime(2020, 1, 1)
    rows = []
    for case in range(1, ncases + 1):
        for i in range(nrows):
            row = {"datetime": base_time + timedelta(days=i), "case": case}
            for name in input_names:
                row[name] = np.random.rand() * 1.0
            for name in output_names:
                row[name] = np.random.rand() * 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _dict_inputs_from_names(input_names, ndays, batch):
    return {name: np.random.rand(batch, ndays).astype(np.float32) for name in input_names}


def _df_outputs(output_names, batch):
    cols = list(output_names)
    arr = np.random.rand(batch, len(cols)).astype(np.float32)
    return pd.DataFrame(arr, columns=cols)


def _layer_param_counts(layer: tf.keras.layers.Layer):
    t = sum(int(np.prod(w.shape)) for w in layer.trainable_weights)
    nt = sum(int(np.prod(w.shape)) for w in layer.non_trainable_weights)
    return t, nt


def write_layer_table(model: tf.keras.Model, csv_path: str):
    rows = []
    for l in model.layers:
        t, nt = _layer_param_counts(l)
        act = getattr(l, "activation", None)
        act_name = getattr(act, "__name__", None) if callable(act) else None
        rows.append({
            "name": l.name,
            "class": l.__class__.__name__,
            "trainable": l.trainable,
            "trainable_params": t,
            "non_trainable_params": nt,
            "activation": act_name
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[OK] wrote layer table: {csv_path}")


def safe_plot(model, out_png):
    try:
        tf.keras.utils.plot_model(model, to_file=out_png, show_shapes=True,
                                  expand_nested=True, dpi=120, rankdir="TB")
        print(f"[OK] wrote graphviz diagram: {out_png}")
    except Exception as e:
        print(f"[WARN] plot_model failed ({e}); writing text summary instead.")
        with open(out_png.replace(".png", ".txt"), "w") as f:
            model.summary(print_fn=lambda s: f.write(s + "\n"))
        print(f"[OK] wrote textual summary: {out_png.replace('.png', '.txt')}")


def save_dot(model: tf.keras.Model, out_dot: str):
    try:
        dot = tf.keras.utils.model_to_dot(model, show_shapes=True, expand_nested=True, dpi=120, rankdir="TB")
        dot.write(out_dot)
        print(f"[OK] wrote DOT graph: {out_dot}")
    except Exception as e:
        print(f"[WARN] model_to_dot failed ({e})")


def describe_compiled(model: tf.keras.Model):
    print("\n--- COMPILED LOSS / METRICS ---")
    try:
        print("loss :", model.loss)
    except Exception as e:
        print("loss : <unavailable>", e)
    lw = getattr(model, "loss_weights", None)
    if lw is None:
        try:
            lw = getattr(model.compiled_loss, "_loss_weights", None)
        except Exception:
            lw = None
    print("loss_weights:", lw if lw is not None else "<N/A>")
    try:
        print("metrics_names:", model.metrics_names)
    except Exception:
        try:
            print("metrics:", [m.name for m in model.metrics])
        except Exception:
            print("metrics: <N/A>")


def list_trainability(model: tf.keras.Model):
    trainable, frozen = [], []
    for layer in model.layers:
        (trainable if layer.trainable else frozen).append(layer.name)
    print("\n--- TRAINABLE LAYERS ---")
    for n in trainable:
        print("  ", n)
    print("\n--- FROZEN LAYERS ---")
    for n in frozen:
        print("  ", n)


def list_outputs(model: tf.keras.Model):
    print("\n--- OUTPUTS ---")
    if isinstance(model.outputs, (list, tuple)):
        for t in model.outputs:
            print(f"  {t.name}: {t.shape}")
    elif isinstance(model.outputs, dict):
        for k, t in model.outputs.items():
            print(f"  {k}: {t.shape}")
    else:
        t = model.outputs
        print(f"  {t.name}: {t.shape}")



COLOR_BASE = "#1f77b4"    
COLOR_TARGET = "#2ca02c"   
COLOR_CONTRAST = "#d62728"  
COLOR_SUM = "#9467bd"       

def loss_schema_from_builder(builder):
    """
    Extract the exact λ/μ from YAML (builder_args), including per-scenario overrides.
    Returns dict describing base & per-scenario terms.
    """
    args = builder.builder_args or {}
    scenarios = args.get("scenarios", [])

    source_w = float(args.get("source_weight", 1.0))
    default_target_w = float(args.get("target_weight", 1.0))
    default_contrast_w = float(args.get("contrast_weight", 0.5))

    sc_terms = []
    for sc in scenarios:
        sid = sc["id"]
        tw = float(sc.get("target_weight", default_target_w))
        cw = float(sc.get("contrast_weight", default_contrast_w))
        sc_terms.append({
            "id": sid,
            "out": f"out_{sid}_unscaled",
            "contrast_out": f"out_{sid}_contrast_unscaled",
            "lambda": tw,
            "mu": cw})

    schema = {
        "loss_fn": "ScaledMaskedMAE", 
        "base": {"out": "out_base_unscaled", "lambda": source_w},
        "scenarios": sc_terms,
        "formula_symbolic": "L = λ_base·L_base + Σ_i [ λ_i·L_i + μ_i·L_i^Δ ]"}
    return schema


def write_formula_txt(schema, out_txt):
    """
    Write a concrete loss formula with all scenario weights (λ_i, μ_i).
    """
    parts = [f"λ_base·L_base (λ_base={schema['base']['lambda']})"]
    for sc in schema["scenarios"]:
        parts.append(f"λ_{sc['id']}·L_{sc['id']} (λ_{sc['id']}={sc['lambda']})")
        parts.append(f"μ_{sc['id']}·L^Δ_{sc['id']} (μ_{sc['id']}={sc['mu']})")
    full = " + ".join(parts)
    with open(out_txt, "w") as f:
        f.write("Loss function (per-step, compiled):\n")
        f.write("  " + schema["formula_symbolic"] + "\n\n")
        f.write("Expanded with configured weights:\n")
        f.write("  L_total = " + full + "\n")
        f.write("\nLoss per-output uses: " + schema["loss_fn"] + "\n")
    print(f"[OK] wrote loss formula: {out_txt}")


def draw_loss_graph(schema, out_png):
    """
    Create a small Graphviz diagram showing outputs -> losses -> Σ with λ/μ labels.
    """
    if pydot is None:
        print("[WARN] pydot is not available; skipping loss graph.")
        return

    g = pydot.Dot("loss_graph", graph_type="digraph", rankdir="LR", fontsize="10")
    base_out = schema["base"]["out"]
    ln = f"L_base"
    n_out_base = pydot.Node(base_out, shape="box", style="filled", fillcolor="#e6f0ff", color=COLOR_BASE)
    n_L_base = pydot.Node(ln, shape="ellipse", color=COLOR_BASE, fontcolor=COLOR_BASE, label=f"{ln}\n({schema['loss_fn']})")
    g.add_node(n_out_base); g.add_node(n_L_base)
    g.add_edge(pydot.Edge(base_out, ln, color=COLOR_BASE))
    n_sum = pydot.Node("Σ (L_total)", shape="doublecircle", color=COLOR_SUM, fontcolor=COLOR_SUM, style="bold")
    g.add_node(n_sum)
    g.add_edge(pydot.Edge(ln, "Σ (L_total)", label=f"λ_base={schema['base']['lambda']}", color=COLOR_BASE))

    for sc in schema["scenarios"]:
        sid = sc["id"]
        out_s = sc["out"]
        Ls = f"L_{sid}"
        n_out_s = pydot.Node(out_s, shape="box", style="filled", fillcolor="#e9f7ec", color=COLOR_TARGET)
        n_Ls = pydot.Node(Ls, shape="ellipse", color=COLOR_TARGET, fontcolor=COLOR_TARGET, label=f"{Ls}\n({schema['loss_fn']})")
        g.add_node(n_out_s); g.add_node(n_Ls)
        g.add_edge(pydot.Edge(out_s, Ls, color=COLOR_TARGET))
        g.add_edge(pydot.Edge(Ls, "Σ (L_total)", label=f"λ_{sid}={sc['lambda']}", color=COLOR_TARGET))
        out_c = sc["contrast_out"]
        Lc = f"L^Δ_{sid}"
        n_out_c = pydot.Node(out_c, shape="box", style="filled", fillcolor="#fdecea", color=COLOR_CONTRAST)
        n_Lc = pydot.Node(Lc, shape="ellipse", color=COLOR_CONTRAST, fontcolor=COLOR_CONTRAST, label=f"{Lc}\n({schema['loss_fn']})")
        g.add_node(n_out_c); g.add_node(n_Lc)
        g.add_edge(pydot.Edge(out_c, Lc, color=COLOR_CONTRAST))
        g.add_edge(pydot.Edge(Lc, "Σ (L_total)", label=f"μ_{sid}={sc['mu']}", color=COLOR_CONTRAST))

    legend = pydot.Cluster(graph_name="cluster_legend", label="Legend", color="#cccccc", fontsize="10")
    legend.add_node(pydot.Node("Legend_Base", label="Base supervised", shape="plaintext", fontcolor=COLOR_BASE))
    legend.add_node(pydot.Node("Legend_Target", label="Scenario supervised", shape="plaintext", fontcolor=COLOR_TARGET))
    legend.add_node(pydot.Node("Legend_Contrast", label="Scenario contrast", shape="plaintext", fontcolor=COLOR_CONTRAST))
    g.add_subgraph(legend)

    try:
        g.write_png(out_png)
        print(f"[OK] wrote loss graph: {out_png}")
    except Exception as e:
        print(f"[WARN] writing loss graph failed ({e})")


def assert_multiscenario_contract(model: tf.keras.Model, scenarios: list, loss_weights: Optional[dict] = None):
    outnames = []
    if isinstance(model.outputs, (list, tuple)):
        for t in model.outputs:
            outnames.append(t.name.split(":")[0].split("/")[0])
    elif isinstance(model.outputs, dict):
        outnames = list(model.outputs.keys())
    else:
        outnames = [model.outputs.name.split(":")[0].split("/")[0]]

    outset = set(outnames)
    required = {"out_base_unscaled"}
    for sc in scenarios:
        required.add(f"out_{sc}_unscaled")
        required.add(f"out_{sc}_contrast_unscaled")

    missing = required - outset
    if missing:
        raise AssertionError(f"Missing required outputs: {sorted(missing)}")
    print("[OK] multi-scenario outputs present:", sorted(required))

    if not isinstance(loss_weights, dict):
        lw = getattr(model, "loss_weights", None)
        if isinstance(lw, dict):
            loss_weights = lw
        else:
            lw = getattr(model.compiled_loss, "_loss_weights", None)
            loss_weights = lw if isinstance(lw, dict) else {}

    for k, v in loss_weights.items():
        print(f"[INFO] compiled weight {k} = {v}")


def build_and_inspect_step(builder, step_cfg, out_dir):
    builder.set_builder_args(step_cfg.get("builder_args", {}))

    def _expand(val):
        return val.replace("{output_dir}", GLOBAL_OUTPUT_DIR) if isinstance(val, str) else val

    load_model_fname = _expand(step_cfg.get("load_model_fname", None))
    save_model_fname = _expand(step_cfg.get("save_model_fname", None))
    output_prefix   = _expand(step_cfg.get("output_prefix", None))  

    if load_model_fname and not os.path.exists(load_model_fname + ".h5"):
        print(f"[WARN] weights not found for inspection: {load_model_fname+'.h5'} — building from scratch.")
        load_model_fname = None

    builder.load_model_fname = load_model_fname
    input_names = builder.input_names
    output_names = list(builder.output_names)
    ndays = builder.ndays
    df_dummy = _make_dummy_df(input_names, output_names, ndays, nrows=ndays + 5, ncases=1)
    input_layers = builder.input_layers()
    model = builder.build_model(input_layers, df_dummy)
    batch = 8
    fit_in = _dict_inputs_from_names(input_names, ndays, batch)
    test_in = _dict_inputs_from_names(input_names, ndays, batch)

    is_multi = getattr(builder, "is_multi_scenario_step", lambda: False)()
    if is_multi:
        n_sc = len(builder.builder_args.get("scenarios", []))
        outs = [_df_outputs(output_names, batch) for _ in range(1 + n_sc)]
        fit_out, test_out = outs, [df.copy() for df in outs]
        init_epochs = 0
        main_epochs = 0
    else:
        fit_out = _df_outputs(output_names, batch)
        test_out = _df_outputs(output_names, batch)
        init_epochs = 0
        main_epochs = 0

    history, compiled_model = builder.fit_model(
        model,
        fit_in,
        fit_out,
        test_in,
        test_out,
        init_train_rate=step_cfg.get("init_train_rate", 0.001),
        init_epochs=init_epochs,
        main_train_rate=step_cfg.get("main_train_rate", 0.0005),
        main_epochs=main_epochs)

    list_outputs(compiled_model)
    list_trainability(compiled_model)
    describe_compiled(compiled_model)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{step_cfg['name']}")
    safe_plot(compiled_model, base + "_model.png")
    save_dot(compiled_model, base + "_model.dot")
    write_layer_table(compiled_model, base + "_layers.csv")

    if is_multi:
        sc_ids = [s["id"] for s in builder.builder_args.get("scenarios", [])]
        assert_multiscenario_contract(compiled_model, sc_ids, loss_weights=None)

        schema = loss_schema_from_builder(builder)  
        draw_loss_graph(schema, base + "_losses.png")
        write_formula_txt(schema, base + "_loss_formula.txt")

        if STRICT_LOSS_CHECK:
            strict_check_loss_weights(compiled_model, schema)

    with open(base + "_builder_args.json", "w") as f:
        json.dump(builder.builder_args, f, indent=2)

    print(f"\n[OK] Step '{step_cfg['name']}' inspection complete.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML transfer config")
    ap.add_argument("--steps", nargs="*", default=None,
                    help="Subset of steps to inspect. Default: all in config")
    ap.add_argument("--outdir", default="test_artifacts", help="Where to put diagrams")
    args = ap.parse_args()

    cfg = read_config(args.config)
    mbcfg = cfg["model_builder_config"]

    global GLOBAL_OUTPUT_DIR
    GLOBAL_OUTPUT_DIR = cfg.get("output_dir", ".")

    builder = model_builder_from_config(mbcfg) 

    steps = cfg["steps"]
    if args.steps:
        wanted = set(args.steps)
        steps = [s for s in steps if s["name"] in wanted]

    print("\n========== START INSPECTION ==========\n")
    for step in steps:
        print(f"--- BUILDING STEP: {step['name']} ---")
        build_and_inspect_step(builder, step, args.outdir)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()

def strict_check_loss_weights(compiled_model: tf.keras.Model, schema: dict) -> None:
    compiled = getattr(compiled_model, "loss_weights", None)
    if compiled is None:
        compiled = getattr(getattr(compiled_model, "compiled_loss", None), "_loss_weights", None)
    if not isinstance(compiled, dict):
        compiled = {}

    expected = {schema["base"]["out"]: schema["base"]["lambda"]}
    for sc in schema["scenarios"]:
        expected[sc["out"]] = sc["lambda"]
        expected[sc["contrast_out"]] = sc["mu"]

    missing = [k for k in expected if k not in compiled]
    mismatch = {k: (expected[k], compiled.get(k)) for k in expected if compiled.get(k) != expected[k]}

    if missing or mismatch:
        raise AssertionError(f"Loss weight mismatch.\nMissing keys: {missing}\nMismatched values: {mismatch}")

    print("[OK] compiled loss weights match YAML (base + scenarios + contrasts).")

