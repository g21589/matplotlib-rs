import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 60)
X, Y = np.meshgrid(x, x)
Z = np.sin(X) * np.cos(Y)
fig, ax = plt.subplots()
cs = ax.contour(X, Y, Z, levels=7, cmap="RdBu")
ax.clabel(cs, inline=True, fontsize=8)
cf = ax.contourf(X, Y, Z, levels=[-1, -0.5, 0, 0.5, 1], alpha=0.3, hatches=["/", None, "\\", "."])
fig.colorbar(cf, orientation="horizontal", extend="both")
fig.savefig("contour_labels.png")
