import os
import pandas as pd
import yaml
from tensorflow.keras.models import load_model
from casanntra.model_builder import ModelBuilder, UnscaleLayer
from casanntra.staged_learning import read_config,model_builder_from_config
from casanntra.read_data import read_data


def initialize_builder(config_file, model_h5_path):
    """Initializes and returns the correct ModelBuilder subclass based on the YAML config."""
    config = read_config(config_file)
    model_name = os.path.splitext(os.path.basename(model_h5_path))[0]

    # ✅ Find the corresponding training step
    step = next((s for s in config["steps"] if s["save_model_fname"] == model_name), None)
    if step is None:
        raise ValueError(f"Error: No training step found for model '{model_name}' in the YAML config.")

    # ✅ Load the correct ModelBuilder subclass using staged_learning.py logic
    builder_config = config["model_builder_config"]
    builder = model_builder_from_config(builder_config)  # ✅ Now correctly initializes subclass

    return builder, config, model_name

def convert_h5_to_tf(builder, model_h5_path):
    """Converts an .h5 model to TensorFlow's SavedModel format with an UnscaleLayer."""
    
    save_as_tf = True
    # ✅ Register required custom objects
    custom_objects = builder.custom_objects
    #custom_objects["UnscaleLayer"] = UnscaleLayer  # ✅ Explicitly register UnscaleLayer

    # ✅ Load the original trained model
    print(custom_objects)
    model = load_model(model_h5_path, custom_objects=custom_objects)
    print(model.summary())

    # ✅ Wrap model with UnscaleLayer **before saving**
    wrapped_model = builder.wrap_with_unscale_layer2(model)
    wrapped_model.compile()

    # ✅ Verify that UnscaleLayer is present before saving
    print("🔍 Checking for UnscaleLayer in model:")
    print([layer.name for layer in wrapped_model.layers])  # ✅ Debug: Print all layers

    # ✅ Save the wrapped model in TensorFlow format
    if save_as_tf:
        tf_model_path = model_h5_path.replace(".h5", "_tf")
        wrapped_model.save(tf_model_path, save_format="tf", include_optimizer=False)  # ✅ Save model structure
        print(f"✅ Model converted and saved at: {tf_model_path}")
        return tf_model_path
    else:
        return wrapped_model


def process_and_predict(model, input_file, output_csv, builder, prediction_head="base"):
    """Processes input data, ensures correct prediction shape, and makes predictions."""
    
    # ✅ Read input data
    df_raw = read_data(input_file, input_mask_regex=None)


    # ✅ Compute antecedent lags
    df_in = builder.calc_antecedent_preserve_cases(df_raw)
    df_in['fold'] = 0
    timestamps = df_in.datetime    

    # ✅ Restructure inputs for ANN
    df_in = builder.df_by_feature_and_time(df_in).drop(["datetime", "case", "fold"], level="var", axis=1)
    df_in = {name: df_in.loc[:, (name, slice(None))].droplevel("var", axis=1) for name in builder.input_names}

    # ✅ Make predictions (returns a dictionary of named outputs)
    predictions = model.predict(df_in)


    # ✅ Handle dictionary outputs: Check if output is a dictionary or a list
    if isinstance(predictions, list):  
        print("⚠ WARNING: Predictions are a list, named outputs may not have persisted.")

        # ✅ Manually reconstruct dictionary from list using model output names
        predictions_dict = dict(zip(model.output_names, predictions))
    elif isinstance(predictions, dict):
        predictions_dict = predictions
    else:
        raise ValueError("Unexpected model output type. Expected a dictionary or list.")

    # ✅ Debugging: Check if UnscaleLayer was applied
    print("🔍 Raw Model Output (Before Selecting Head):", predictions_dict)

    # ✅ Handle dictionary outputs: Ensure we retrieve the correct output tensor
    print(prediction_head, predictions_dict)
    if prediction_head not in predictions_dict:
        raise ValueError(f"Invalid prediction head '{prediction_head}'. Choose from {list(predictions_dict.keys())}.")
        
    predictions = predictions_dict[prediction_head]  # ✅ Select the correct output

    # ✅ Debugging: Check min/max values
    print(f"🔍 Final Predictions ({prediction_head}): min={predictions.min()}, max={predictions.max()}")

    # ✅ Convert predictions to DataFrame
    df_pred = pd.DataFrame(predictions, columns=builder.output_list(),index=timestamps)
    df_pred.to_csv(output_csv, index=True)
    print(f"✅ Predictions ({prediction_head}) saved at: {output_csv}")


def convert_validate_model(config_file, model_h5_path, input_file,prediction_head):
    """Runs prediction for both the H5 model and TF model, producing output validation CSVs."""

    # ✅ Initialize builder once
    builder, config, model_name = initialize_builder(config_file, model_h5_path)

    # ✅ Convert the model to TF format
    tf_model_path = convert_h5_to_tf(builder, model_h5_path)

    # ✅ Load both models with registered custom objects
    custom_objects = builder.custom_objects  # ✅ Use the builder's registered objects
    #custom_objects['UnscaleLayer'] = UnscaleLayer
    h5_model = load_model(model_h5_path, custom_objects=custom_objects)
    h5_model.load_weights(model_h5_path.replace(".h5",".weights.h5"))
    print(h5_model.summary())

     #  Wrap the `.h5` model with `UnscaleLayer`
    #h5_model = builder.wrap_with_unscale_layer2(h5_model)   
    h5_model = builder.wrap_with_unscale_layer2(h5_model)   
    tf_model = load_model(tf_model_path, custom_objects=custom_objects)

    # ✅ Predict with both and save outputs
    h5_output_csv = model_h5_path.replace(".h5", "_h5inputcheck.csv")
    tf_output_csv = h5_output_csv.replace("h5input","tfinput")
    
    process_and_predict(h5_model, input_file, h5_output_csv, builder,prediction_head)
    process_and_predict(tf_model, input_file, tf_output_csv, builder,prediction_head)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a trained H5 model to TF format and validate with predictions.")
    parser.add_argument("config", type=str, help="Path to the YAML training config file.")
    parser.add_argument("model_h5", type=str, help="Path to the trained H5 model.")
    parser.add_argument("input_file", type=str, help="Path to an input file for model validation.")
    parser.add_argument("pred_head", type=str, help="Prediction output tensor for model validation.")
    args = parser.parse_args()

    # ✅ Run conversion and validation
    convert_validate_model(args.config, args.model_h5, args.input_file,prediction_head=args.pred_head)
