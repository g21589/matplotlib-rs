import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(np.arange(5), np.arange(5) ** 2, "o-")
fig.savefig("savefig_dpi150.png", dpi=150)
fig.savefig("savefig_transparent.png", transparent=True)
fig.savefig("savefig_face.png", facecolor="lightyellow", edgecolor="k")
fig.savefig("savefig_tight_pad.png", bbox_inches="tight", pad_inches=0.5)
