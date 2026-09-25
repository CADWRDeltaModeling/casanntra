from casanntra.read_data import read_data
import matplotlib.pyplot as plt

df0 = read_data("dsm2_base*", input_mask_regex=r"dsm2_base_10.*csv")
df1 = read_data("schism_base*", None)
df2 = read_data("rma_base*", None)


fig, axes = plt.subplots(2, sharex=True)

for ax, column_addressed in zip(axes, ["sf_tidal_filter", "sf_tidal_energy"]):
    sf0 = df0[["datetime", column_addressed]]
    print("dsm2", column_addressed, sf0[column_addressed].mean())

    sf1 = df1[["datetime", column_addressed]]
    print("schism", column_addressed, sf1[column_addressed].mean())

    sf2 = df1[["datetime", column_addressed]]
    print("rma", column_addressed, sf2[column_addressed].mean())
    ax.plot(sf0.datetime, sf0[column_addressed] * 2.0 - 2, c="0.2", label="DSM2")
    ax.plot(sf1.datetime, sf1[column_addressed], c="red", label="schism")
    ax.plot(sf2.datetime, sf2[column_addressed] + 0.1, c="blue", label="rma")
    ax.set_title(column_addressed)
    ax.legend()


plt.show()
