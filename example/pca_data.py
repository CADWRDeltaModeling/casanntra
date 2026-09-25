from casanntra.read_data import read_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


def difference(x):
    return x.diff().bfill()


fpattern = "dsm2_base_*.csv"
df = read_data(fpattern)
# df['sf_dsub'] = df.groupby('case')['sf_tidal_filter'].transform(difference)
data_normalized = df[["sf_tidal_filter", "sf_tidal_energy", "case"]]
case_mean = data_normalized.groupby("case")["sf_tidal_filter"].transform("mean")  # 3.75
data_normalized.loc[:, "sf_tidal_filter"] -= case_mean
case_std = data_normalized.groupby("case")["sf_tidal_filter"].transform("std")  # 0.35

data_normalized.loc[:, "sf_tidal_filter"] /= case_std


mean_energy = (
    data_normalized.sf_tidal_energy.mean()
)  # Note global for energy, group for filter
std_energy = data_normalized.sf_tidal_energy.std()

print(f"mean_energy: {mean_energy}, std_energy: {std_energy}")

fig, (ax0, ax1) = plt.subplots(2, sharex=True, figsize=(8, 6))
ax0.plot(case_mean)
ax1.plot(case_std)
plt.show()

data_normalized.loc[:, "sf_tidal_energy"] -= mean_energy
data_normalized.loc[:, "sf_tidal_energy"] /= std_energy
data_normalized = data_normalized.drop("case", axis=1)

fig, ax = plt.subplots(1)
ax.scatter(data_normalized.sf_tidal_energy, data_normalized.sf_tidal_filter)
plt.show()

# Perform PCA
pca = PCA(n_components=2)
pca.fit(data_normalized)
print(pca.components_)
pca_loadings = pca.fit_transform(data_normalized)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print(data_normalized.columns)

pca_df = pd.DataFrame(
    index=data_normalized.index,
    data={"PC1": pca_loadings[:, 0], "PC2": pca_loadings[:, 1]},
)

# Plot the principal components over time
fig, (ax0, ax1) = plt.subplots(2, sharex=True, figsize=(8, 6))
ax0.plot(pca_df.index, pca_df["PC1"], label="PC1 (Principal Component 1)")
ax0.plot(pca_df.index, pca_df["PC2"], label="PC2 (Principal Component 2)")
ax0.set_title("PCA Loadings Over Time")
ax0.set_xlabel("Time")
ax0.set_ylabel("PCA Loadings")
ax0.legend()
ax0.grid()
ax1.plot(
    data_normalized.index,
    data_normalized.sf_tidal_filter.values,
    label="SF filter (norm)",
)
ax1.plot(
    data_normalized.index,
    data_normalized.sf_tidal_energy.values,
    label="SF Energy (norm)",
)
ax1.legend()
ax1.grid()
fig.tight_layout()
plt.show()


fig, ((ax0, ax0a), (ax1, ax1a), (ax2, ax2a)) = plt.subplots(3, 2)
ax0.hist(df.jer, bins=15, range=(0, 4000))
ax0a.ecdf(df.jer, label="CDF")
ax0a.grid()
ax0.set_title("JER")

ax1.hist(df.emm2, bins=15, range=(0, 6000))
ax1a.ecdf(df.emm2, label="CDF")
ax1a.grid()
ax1.set_title("EMM")

ax2.hist(df.bac, bins=15, range=(0, 3000.0))
ax2a.ecdf(df.bac, label="CDF")
ax2a.grid()
ax2.set_title("BAC")

plt.show()
