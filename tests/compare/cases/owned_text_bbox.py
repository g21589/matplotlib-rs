import matplotlib.pyplot as plt

# bbox patches on texts the axes own (titles, axis labels, tick labels),
# in every builtin box style, rotated with the y label and tick labels
styles = ["square", "circle", "ellipse", "rarrow", "larrow", "darrow", "round",
          "round4", "sawtooth", "roundtooth"]
fig, axs = plt.subplots(2, 5, figsize=(12, 5), layout="constrained")
for ax, st in zip(axs.flat, styles):
    ax.set_title(st, bbox=dict(boxstyle=st, fc="yellow", alpha=0.6))
    ax.set_ylabel("ylabel", bbox=dict(boxstyle=st + ",pad=0.4", fc="red", ec="k"))
    ax.set_xticks([0.2, 0.8], ["tick a", "tick b"], rotation=20,
                  bbox=dict(boxstyle=st, fc="c", lw=0.5))
    ax.set_xlabel("x label", bbox=dict(facecolor="0.8", pad=6))
fig.suptitle("suptitle", bbox=dict(boxstyle="round4", fc="w"))
fig.savefig("owned_text_bbox.png")
