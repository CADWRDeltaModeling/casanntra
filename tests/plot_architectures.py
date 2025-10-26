import os
import argparse
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from casanntra.staged_learning import read_config, model_builder_from_config

try:
    from casanntra.multi_scenario_model_builder import MultiScenarioModelBuilder
except Exception:
    MultiScenarioModelBuilder = None


def _synthetic_input_df(input_names, rows=365, seed=13):
    """Create minimal DF so Normalization.adapt() can run without CSVs."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "datetime": pd.date_range("2000-01-01", periods=rows, freq="D"),
        "case": 1,
    })
    for name in input_names:
        df[name] = rng.standard_normal(rows).astype("float32")
    return df


def _safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _transfer_type(step):
    tt = _safe_get(step, "builder_args", "transfer_type")
    if tt is None or str(tt).lower() == "none":
        return "direct"
    return str(tt).lower()


def _normalize_builder_args(builder_args):
    if not builder_args:
        return {}
    snap = deepcopy(builder_args)
    if "transfer_type" in snap:
        tt = snap["transfer_type"]
        snap["transfer_type"] = "direct" if (tt is None or str(tt).lower() == "none") else str(tt).lower()
    return snap


def _diff_steps(prev, cur):
    """Human-friendly bullets of changes prev -> cur."""
    if prev is None:
        return ["(first step)"]

    bullets = []

    top_keys = [
        "name", "input_prefix", "input_mask_regex", "output_prefix",
        "load_model_fname", "save_model_fname", "pool_size", "pool_aggregation",
        "target_fold_length", "init_train_rate", "init_epochs", "main_train_rate", "main_epochs"
    ]
    for k in top_keys:
        pv, cv = prev.get(k), cur.get(k)
        if pv != cv:
            bullets.append(f"* {k}: {pv} → {cv}")

    pa = _normalize_builder_args(prev.get("builder_args"))
    ca = _normalize_builder_args(cur.get("builder_args"))

    ba_keys = [
        "transfer_type",
        "contrast_weight",
        "feature_layers",    
        "base_layers",    
        "trunk_layers",      
        "branch_layer",
        "source_data_prefix",
        "source_input_mask_regex",
        "scenarios"
    ]
    for k in ba_keys:
        pv, cv = pa.get(k), ca.get(k)
        if pv != cv:
            bullets.append(f"* builder_args.{k}: {pv} → {cv}")

    return bullets or ["(no parameter changes)"]


def _plot_single_model(builder, out_png, step_name=None, respect_load=False, dpi=220):
    """Build + render to PNG. We ignore load_model unless explicitly requested."""
    if not respect_load and hasattr(builder, "load_model_fname"):
        builder.load_model_fname = None

    df = _synthetic_input_df(builder.input_names, rows=365)
    inputs = builder.input_layers()
    model = builder.build_model(inputs, df)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plot_model(
        model,
        to_file=out_png,
        show_shapes=True,
        show_layer_names=True,
        rankdir="LR", 
        dpi=dpi,
    )
    return model


def _write_notes(out_notes_md, step, change_bullets):
    os.makedirs(os.path.dirname(out_notes_md), exist_ok=True)
    with open(out_notes_md, "w", encoding="utf-8") as f:
        f.write(f"# {step['name']}\n\n")
        f.write("**Highlights / Changes vs previous step**\n\n")
        for line in change_bullets:
            f.write(f"{line}\n")
        f.write("\n**Transfer type**: " + _transfer_type(step) + "\n")


def render_per_step(cfg, outdir, respect_load=False, dpi=220):
    """Render each YAML step with notes on what changed."""
    mb_cfg = cfg["model_builder_config"]
    builder = model_builder_from_config(mb_cfg)

    prev_step = None
    for i, step in enumerate(cfg["steps"], start=1):
        builder.set_builder_args(step.get("builder_args", {}))

        tag = f"{i:02d}-{step['name'].replace('/', '_')}"
        out_png = os.path.join(outdir, f"{tag}.png")
        out_notes = os.path.join(outdir, f"{tag}.notes.md")

        _plot_single_model(builder, out_png, step_name=step["name"], respect_load=respect_load, dpi=dpi)

        changes = _diff_steps(prev_step, step)
        _write_notes(out_notes, step, changes)
        prev_step = step

    print(f"[per-step] diagrams written to: {outdir}")


def render_direct(cfg, out_png, respect_load=False, dpi=220):
    """Render the first 'direct' step (or first step if none labeled direct)."""
    mb_cfg = cfg["model_builder_config"]
    builder = model_builder_from_config(mb_cfg)

    step = None
    for s in cfg["steps"]:
        if _transfer_type(s) == "direct":
            step = s
            break
    if step is None:
        step = cfg["steps"][0]

    builder.set_builder_args(step.get("builder_args", {}))
    _plot_single_model(builder, out_png, step_name=step["name"], respect_load=respect_load, dpi=dpi)
    print(f"[direct] {out_png}")


def render_contrastive(cfg, out_png, respect_load=False, dpi=220):
    """Render the first 'contrastive' step from the YAML."""
    mb_cfg = cfg["model_builder_config"]
    builder = model_builder_from_config(mb_cfg)

    step = None
    for s in cfg["steps"]:
        if _transfer_type(s) == "contrastive":
            step = s
            break
    if step is None:
        raise RuntimeError("No contrastive step found in YAML (builder_args.transfer_type: contrastive).")

    builder.set_builder_args(step.get("builder_args", {}))
    _plot_single_model(builder, out_png, step_name=step["name"], respect_load=respect_load, dpi=dpi)
    print(f"[contrastive] {out_png}")


def render_multiscenario(cfg, out_png, respect_load=False, dpi=220):
    """If the YAML uses MultiScenarioModelBuilder (or available in code), render a multi-scenario diagram."""
    try:
        mb_cfg = cfg["model_builder_config"]
        if mb_cfg["builder_name"] == "MultiScenarioModelBuilder":
            builder = model_builder_from_config(mb_cfg)
            step = None
            for s in cfg["steps"]:
                sc = _safe_get(s, "builder_args", "scenarios", default=None)
                if sc:
                    step = s
                    break
            if step is None:
                step = cfg["steps"][-1]
            builder.set_builder_args(step.get("builder_args", {}))
            _plot_single_model(builder, out_png, step_name=step["name"], respect_load=respect_load, dpi=dpi)
            print(f"[multi-scenario] {out_png}")
            return

        if MultiScenarioModelBuilder is None:
            raise RuntimeError("MultiScenarioModelBuilder is not importable in this environment.")
        base_args = mb_cfg["args"]
        ms_builder = MultiScenarioModelBuilder(**base_args)

        trunk = None
        for s in cfg["steps"]:
            trunk = _safe_get(s, "builder_args", "feature_layers") or _safe_get(s, "builder_args", "base_layers")
            if trunk:
                break
       
        scenarios = []
        for s in cfg["steps"]:
            if _transfer_type(s) == "contrastive":
                sid = s["name"].split(".", 1)[-1] if "." in s["name"] else s["name"]
                scenarios.append({
                    "id": sid,
                    "input_prefix": s.get("input_prefix"),
                    "input_mask_regex": s.get("input_mask_regex"),
                })
        if not scenarios:
            scenarios = [{"id": "suisun"}, {"id": "slr"}]

        ms_builder.set_builder_args({
            "trunk_layers": trunk,
            "branch_layer": {"type": "GRU", "units": 16, "name": "branch", "return_sequences": False},
            "include_source_branch": True,
            "head_activation": "elu",
            "init_targets_from_source": True,
            "scenarios": scenarios,
            "source_weight": 1.0,
            "target_weight": 1.0,
            "contrast_weight": 0.5,
        })

        _plot_single_model(ms_builder, out_png, step_name="multi_scenario", respect_load=respect_load, dpi=dpi)
        print(f"[multi-scenario] {out_png}")

    except Exception as e:
        raise RuntimeError(f"Unable to render multi-scenario: {e}") from e

def main():
    parser = argparse.ArgumentParser(description="Render Casanntra architectures from YAML.")
    parser.add_argument("--config", required=True, help="Path to YAML (e.g., transfer_config_schism_v4.yml)")
    parser.add_argument("--outdir", default="arch_png", help="Output dir for PNGs and notes")
    parser.add_argument("--render", nargs="+",
                        choices=["per-step", "direct", "contrastive", "multiscenario", "all"],
                        default=["all"],
                        help="Which diagrams to render")
    parser.add_argument("--respect-load-model", action="store_true",
                        help="If set, do not override load_model_fname (requires model files to exist).")
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    args = parser.parse_args()

    cfg = read_config(args.config)
    outdir = args.outdir
    todo = set(args.render)
    if "all" in todo:
        todo = {"per-step", "direct", "contrastive", "multiscenario"}

    if "per-step" in todo:
        render_per_step(cfg, outdir, respect_load=args.respect_load_model, dpi=args.dpi)

    if "direct" in todo:
        render_direct(cfg, os.path.join(outdir, "direct.png"), respect_load=args.respect_load_model, dpi=args.dpi)

    if "contrastive" in todo:
        try:
            render_contrastive(cfg, os.path.join(outdir, "contrastive.png"),
                               respect_load=args.respect_load_model, dpi=args.dpi)
        except RuntimeError as e:
            print(f"[contrastive] skipped: {e}")

    if "multiscenario" in todo:
        try:
            render_multiscenario(cfg, os.path.join(outdir, "multiscenario.png"),
                                 respect_load=args.respect_load_model, dpi=args.dpi)
        except RuntimeError as e:
            print(f"[multi-scenario] skipped: {e}")


if __name__ == "__main__":
    main()