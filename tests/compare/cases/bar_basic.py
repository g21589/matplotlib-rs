import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(["a", "b", "c", "d"], [3, 7, 2, 5], color="tab:orange", edgecolor="k")
ax.set_ylabel("count")
fig.savefig("bar_basic.png")
