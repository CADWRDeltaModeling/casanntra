import os
import glob
import pandas as pd
import tensorflow as tf
import numpy as np
import yaml

def load_config(yaml_file):
    """Load and return the YAML configuration as a dictionary."""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def create_output_dataframe(data, config):
    """
    Creates a DataFrame using the output_names ordering from the YAML config.
    
    Parameters:
      data: A list of rows, where each row is a list or tuple representing a batch's predictions.
      config: The YAML configuration dictionary which should contain:
              model_builder_config -> args -> output_names.
    
    Returns:
      A Pandas DataFrame with columns ordered according to the keys in output_names.
    """
    # Extract output column names preserving order (Python 3.7+ maintains insertion order)
    output_names = list(config["model_builder_config"]["args"]["output_names"].keys())
    # Create DataFrame using these columns; default index is used as the batch number.
    df = pd.DataFrame(data, columns=output_names)
    return df





def load_debug_inputs():
    """
    Loads all CSV files with names matching 'debug_*.csv' from the current directory.
    The key is the input short name (filename stripped of "debug_" and ".csv")
    and the value is a numpy array of the input data.
    """
    input_dict = {}   #F:/ann_workspace/calsurrogate/bin/test/calsim/surrogate/data
    for filepath in glob.glob("F:/ann_workspace/calsurrogate/bin/test/calsim/surrogate/data/debug_*.csv"):
        base = os.path.basename(filepath)
        # Remove prefix "debug_" and suffix ".csv" to get the short input name.
        input_name = base[len("debug_"):-len(".csv")]
        # Read the CSV file with no header.
        df = pd.read_csv(filepath, header=None)
        input_dict[input_name] = df.values  # Convert DataFrame to numpy array.
    return input_dict

def main():
    # Replace this path with the location of your TensorFlow model.
    model_path = r"F:\ann_workspace\casanntra\example\schism_base.suisun_gru2_tf"      #schism_base.suisun_gru2_tf"
    model_path = r"F:\ann_workspace\casanntra\example\base.suisun_debug"  
    model = tf.keras.models.load_model(model_path)
    
    # Load the inputs from CSV files.
    inputs = load_debug_inputs()
    # Depending on your model's expected input signature, you may need to adjust the dictionary.
    # For example, if the model was built with named inputs, the keys here should match the names.
    predictions = model.predict(inputs)

    #["base"]
    config = load_config("../example/transfer_config.yml")

    tensor_out = "out_target_unscaled"
    tensor_out = "out_target"
    print(predictions)
    predictions = create_output_dataframe(predictions[tensor_out],config)
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
