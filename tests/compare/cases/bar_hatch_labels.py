import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.arange(4)
b1 = ax.bar(x - 0.2, [3, 5, 2, 6], 0.4, label="a", hatch="//", edgecolor="k")
b2 = ax.bar(x + 0.2, [4, 2, 5, 3], 0.4, label="b", hatch="x.", color="w", edgecolor="tab:red")
ax.bar_label(b1, padding=3)
ax.bar_label(b2, fmt="%.1f", label_type="center")
ax.set_xticks(x, ["one", "two", "three", "four"])
ax.legend(loc="upper left", ncols=2)
fig.savefig("bar_hatch_labels.png")
