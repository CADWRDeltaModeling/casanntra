import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
import pandas as pd
from casanntra.multi_scenario_model_builder import MultiScenarioModelBuilder
from casanntra.model_builder import ScaledMaskedMAE, ScaledMaskedMSE
from debug_common import make_synthetic_data, set_global_seeds, dump_history_to_csv, assert_heads_match, INPUT_NAMES, OUTPUT_NAMES, NDAYS, EPOCHS

OUTPUT_DIR = "./debug_output/multiscenario"
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# --- WRAPPERS ---
class DebugScaledMaskedMAE(ScaledMaskedMAE):
    def __init__(self, output_scales, name="scaled_mae", **kwargs):
        kwargs.pop('reduction', None)
        super().__init__(output_scales, name=name)

class DebugScaledMaskedMSE(ScaledMaskedMSE):
    def __init__(self, output_scales, name="scaled_mse", **kwargs):
        kwargs.pop('reduction', None)
        super().__init__(output_scales, name=name)
# ----------------

def run_step(step_name, transfer_type, load_fname, trunk_layers, scenarios_cfg=None):
    print(f"\n>> STEP: {step_name}")
    
    # 1. Setup
    set_global_seeds(100) 
    builder = MultiScenarioModelBuilder(INPUT_NAMES, OUTPUT_NAMES, ndays=NDAYS)
    builder.custom_objects["ScaledMaskedMAE"] = DebugScaledMaskedMAE
    builder.custom_objects["ScaledMaskedMSE"] = DebugScaledMaskedMSE
    
    builder.set_builder_args({
        "transfer_type": transfer_type,
        "trunk_layers": trunk_layers,
        "contrast_weight": 1.0,
        "source_weight": 1.0,
        "target_weight": 1.0,
        "per_scenario_branch": False,
        "branch_layers": [],
        "include_source_branch": False,
        "scenarios": scenarios_cfg or []
    })
    builder.load_model_fname = load_fname
    
    # 2. Data Generation
    # STRICT PRODUCTION PARITY: Seed 200 = Base, Seed 201 = Target
    
    if scenarios_cfg:
        set_global_seeds(200)
        df_base = make_synthetic_data(INPUT_NAMES, OUTPUT_NAMES) # Seed 200
        
        set_global_seeds(201)
        df_target = make_synthetic_data(INPUT_NAMES, OUTPUT_NAMES) # Seed 201
        
        dfs = [df_base, df_target]
    else:
        set_global_seeds(200)
        df_base = make_synthetic_data(INPUT_NAMES, OUTPUT_NAMES)
        dfs = [df_base]
            
    aligned = builder.pool_and_align_cases(dfs)
    
    df_in_list = []
    df_out_list = []
    for d in aligned:
        i, o = builder.xvalid_time_folds(d, target_fold_len='5d', split_in_out=True)
        df_in_list.append(i)
        df_out_list.append(o)
        
    # staged_learning logic: df_in = df_base_in (Index 0)
    df_in = df_in_list[0].iloc[:100]
    
    # Output Order: [Base, Target] (MultiScenario convention)
    df_out = [d.iloc[:100] for d in df_out_list]
    if len(df_out) == 1: df_out = df_out[0]

    # Adaptation Data: build_model uses df_in (Base) because that's what xvalid passes
    df_adapt = df_base.iloc[:50]

    # 3. Build
    set_global_seeds(300)
    input_layers = builder.input_layers()
    model = builder.build_model(input_layers, df_adapt)

    if transfer_type == "contrastive" and scenarios_cfg:
        sid = scenarios_cfg[0]['id']
        assert_heads_match(model, "head_base_scaled", f"head_{sid}_scaled")
    
    # 4. Fit
    set_global_seeds(400)
    print(f"   Training {step_name}...")
    
    inputs_lagged = builder.calc_antecedent_preserve_cases(df_in)
    if isinstance(df_out, list):
        outputs_trim = [d.loc[inputs_lagged.index, list(OUTPUT_NAMES)] for d in df_out]
    else:
        outputs_trim = df_out.loc[inputs_lagged.index, list(OUTPUT_NAMES)]
        
    idx = pd.IndexSlice
    fit_in_dict = {name: builder.df_by_feature_and_time(inputs_lagged).loc[:, idx[name, :]].droplevel("var", axis=1) for name in INPUT_NAMES}
    
    history, ann = builder.fit_model(model, fit_in_dict, outputs_trim, fit_in_dict, outputs_trim, 
                      init_train_rate=0.01, init_epochs=EPOCHS, main_train_rate=0.0, main_epochs=0)
    
    csv_path = f"{OUTPUT_DIR}/{step_name}_metrics.csv"
    dump_history_to_csv(history, csv_path)
    
    save_path = f"{OUTPUT_DIR}/{step_name}"
    ann.save_weights(save_path + ".weights.h5")
    ann.save(save_path + ".h5")
    return save_path

def run_full_pipeline():
    trunk = [{"type": "GRU", "units": 32, "name": "lay1", "return_sequences": True, "trainable": True},
             {"type": "GRU", "units": 16, "name": "lay2", "return_sequences": False, "trainable": True}]

    path1 = run_step("dsm2_base", "None", None, trunk)
    path2 = run_step("dsm2.schism", "direct", path1, trunk)
    path3 = run_step("base.multi", "contrastive", path2, trunk, scenarios_cfg=[{"id": "suisun"}])

if __name__ == "__main__":
    run_full_pipeline()
    print("\n✅ MultiScenario Done.")