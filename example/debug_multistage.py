import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
import pandas as pd
from casanntra.multi_stage_model_builder import MultiStageModelBuilder
from casanntra.model_builder import ScaledMaskedMAE, ScaledMaskedMSE
from debug_common import make_synthetic_data, set_global_seeds, dump_history_to_csv, assert_heads_match, INPUT_NAMES, OUTPUT_NAMES, NDAYS, EPOCHS

OUTPUT_DIR = "./debug_output/multistage"
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

def run_step(step_name, transfer_type, load_fname, feature_layers, contrast_weight=None):
    print(f"\n>> STEP: {step_name}")
    
    # 1. Setup
    set_global_seeds(100) 
    builder = MultiStageModelBuilder(INPUT_NAMES, OUTPUT_NAMES, ndays=NDAYS)
    builder.custom_objects["ScaledMaskedMAE"] = DebugScaledMaskedMAE
    builder.custom_objects["ScaledMaskedMSE"] = DebugScaledMaskedMSE
    
    builder.set_builder_args({
        "transfer_type": transfer_type,
        "feature_layers": feature_layers,
        "contrast_weight": contrast_weight
    })
    builder.load_model_fname = load_fname
    
    # 2. Data Generation
    # STRICT PRODUCTION PARITY: Seed 200 = Base, Seed 201 = Target
    
    if transfer_type == "contrastive":
        set_global_seeds(200)
        df_base = make_synthetic_data(INPUT_NAMES, OUTPUT_NAMES) # Seed 200 (Base)
        
        set_global_seeds(201)
        df_target = make_synthetic_data(INPUT_NAMES, OUTPUT_NAMES) # Seed 201 (Target)

        # Manual Alignment mimicking staged_learning
        # staged_learning.py: df_source_in ... df_in (Target). THEN df_in = df_source_in
        
        df_in_base, df_out_base = builder.xvalid_time_folds(df_base, target_fold_len='5d', split_in_out=True)
        _, df_out_target = builder.xvalid_time_folds(df_target, target_fold_len='5d', split_in_out=True)
        
        # PRODUCTION LOGIC: Input is Base. Output is [Target, Base].
        df_in = df_in_base.iloc[:100]
        df_out = [df_out_target.iloc[:100], df_out_base.iloc[:100]]
        
        # Adaptation Data: build_model uses df_in (Base)
        df_adapt = df_base.iloc[:50]
        
    else:
        # Direct/Pretrain uses Base (Seed 200)
        set_global_seeds(200)
        df_base = make_synthetic_data(INPUT_NAMES, OUTPUT_NAMES)
        df_in, df_out = builder.xvalid_time_folds(df_base, target_fold_len='5d', split_in_out=True)
        df_in = df_in.iloc[:100]
        df_out = df_out.iloc[:100]
        df_adapt = df_base.iloc[:50]

    # 3. Build
    set_global_seeds(300)
    input_layers = builder.input_layers()
    model = builder.build_model(input_layers, df_adapt)
    
    if transfer_type == "contrastive":
        assert_heads_match(model, "target_scaled", "source_scaled")

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
    layers = [{"type": "GRU", "units": 32, "name": "lay1", "return_sequences": True, "trainable": True},
              {"type": "GRU", "units": 16, "name": "lay2", "return_sequences": False, "trainable": True}]

    path1 = run_step("dsm2_base", "None", None, layers)
    path2 = run_step("dsm2.schism", "direct", path1, layers)
    path3 = run_step("base.suisun", "contrastive", path2, layers, contrast_weight=1.0)

if __name__ == "__main__":
    run_full_pipeline()
    print("\n✅ Baseline Done.")