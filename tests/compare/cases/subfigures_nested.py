import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 5))
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.4], hspace=0.05)
top.set_facecolor("lavender")
ax = top.subplots(1, 3)
for i, a in enumerate(ax):
    a.plot(np.arange(5) * (i + 1), label=f"s{i}")
top.legend(loc="center right")
top.supylabel("top y")
left, right = bottom.subfigures(1, 2, wspace=0.1)
left.set_facecolor("0.85")
left.set_edgecolor("k")
left.set_linewidth(2)
la = left.add_subplot()
la.imshow(np.arange(12).reshape(3, 4))
left.suptitle("nested left", fontsize=10)
ra = right.subplots(2, 2, sharex=True)
for a in ra.flat:
    a.scatter([1, 2, 3], [3, 1, 2], s=10)
right.supxlabel("right x")
fig.text(0.5, 0.5, "figure text", ha="center", zorder=10, color="r")
fig.savefig("subfigures_nested.png")
fig.savefig("subfigures_nested_tight.png", bbox_inches="tight")
