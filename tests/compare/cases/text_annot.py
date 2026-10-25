import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 0])
ax.text(0.5, 0.5, "hello", fontsize=14, color="purple")
ax.text(1.5, 0.2, "rotated", rotation=45, ha="center")
ax.text(0.1, 0.9, "boxed", transform=ax.transAxes, bbox=dict(facecolor="yellow", alpha=0.5))
ax.annotate("peak", xy=(1, 1), xytext=(1.5, 0.8), arrowprops=dict(arrowstyle="->"))
fig.text(0.02, 0.02, "figure text")
fig.savefig("text_annot.png")
