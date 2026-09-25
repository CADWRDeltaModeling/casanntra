import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_input = pd.read_csv("../data/schism_base_1.csv", parse_dates=["datetime"]).set_index("datetime")
df_pred = pd.read_csv("schism_base.suisun_gru2_h5inputcheck.csv", parse_dates=["datetime"]).set_index("datetime")

# Ensure indices match
common_index = df_input.index.intersection(df_pred.index)
df_input = df_input.loc[common_index]
df_pred = df_pred.loc[common_index]

# Select a sample location (first column)
sample_location = "emm2"

# Create the scatter plot
fig,(ax0,ax1)=plt.subplots(2,figsize=(8, 6))
ax0.scatter(df_input[sample_location], df_pred[sample_location], alpha=0.5, label="Predictions vs Input")
ax0.plot([df_input[sample_location].min(), df_input[sample_location].max()], 
         [df_input[sample_location].min(), df_input[sample_location].max()], 
         linestyle="--", color="red", label="Ideal Fit")

ax0.set_xlabel("Input Data")
ax0.set_ylabel("Model Prediction")
ax0.set_title(f"Input vs Prediction at {sample_location}")
ax0.legend()
ax0.grid(True)

ax1.plot(df_input[sample_location])
ax1.plot(df_pred[sample_location])

# Show the plot
plt.show()
