import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

from casanntra.model_builder import UnscaleLayer, StackLayer, ModifiedExponentialDecayLayer
from casanntra.staged_learning import read_config, model_builder_from_config
from casanntra.model_builder import masked_mae, masked_mse, ScaledMaskedMAE, ScaledMaskedMSE

def convert_model(h5_model_name, config_path):
    # Parse config and get output_dir
    config_data = read_config(config_path)
    output_dir = config_data.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    # Resolve full H5 model path
    if os.path.isabs(h5_model_name) or h5_model_name.startswith(output_dir):
        h5_model_path = h5_model_name
    else:
        h5_model_path = os.path.join(output_dir, h5_model_name)
    if not h5_model_path.endswith(".h5"):
        h5_model_path += ".h5"

    if not os.path.exists(h5_model_path):
        raise FileNotFoundError(f"❌ Could not find model: {h5_model_path}")

    builder = model_builder_from_config(config_data["model_builder_config"])
    output_scales = list(builder.output_names.values())

    # Register custom objects
    custom_objects = {
        "UnscaleLayer": UnscaleLayer,
        "StackLayer": StackLayer,
        "ModifiedExponentialDecayLayer": ModifiedExponentialDecayLayer,
        "masked_mae": masked_mae,
        "masked_mse": masked_mse,
        "scaled_masked_mae": ScaledMaskedMAE(output_scales),
        "scaled_masked_mse": ScaledMaskedMSE(output_scales),
    }

    # ✅ Load the model
    model = load_model(h5_model_path, custom_objects=custom_objects)

    # ✅ Optionally load weights
    weights_path = h5_model_path.replace(".h5", ".weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Weights loaded from: {weights_path}")
    else:
        print(f"No separate weights file found at {weights_path} (proceeding with compiled model).")

    print("🔍 Original model loaded:")
    model.summary()

    # Compile to preserve named outputs
    model.compile()

    # Save as TensorFlow SavedModel to output_dir
    base_name = os.path.splitext(os.path.basename(h5_model_path))[0]
    save_path = os.path.join(output_dir, base_name)
    model.save(save_path, save_format="tf")

    print(f"Converted model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an H5 model to TensorFlow SavedModel format using config-defined output_dir.")
    parser.add_argument("h5_model", help="Name or path of the .h5 model (relative name searched inside output_dir if not a path)")
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML config file (e.g. transfer_config.yml)")
    args = parser.parse_args()

    convert_model(args.h5_model, args.config)
