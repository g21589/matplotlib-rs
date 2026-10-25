import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.linspace(0, 10, 200)
ax.plot(x, np.sin(x))
axin = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
axin.plot(x, np.cos(x), "r")
axin.set_xlim(2, 4)
ax.indicate_inset_zoom(axin, edgecolor="black")
sec = ax.secondary_xaxis("top", functions=(lambda v: v * 2, lambda v: v / 2))
sec.set_xlabel("doubled")
fig.savefig("inset_secondary.png")

# limits changing after the secondary axis exists; log parent scale
fig2, ax2 = plt.subplots()
sec2 = ax2.secondary_yaxis("right", functions=(np.deg2rad, np.rad2deg))
ax2.plot(x * 30, x * 20)
ax2.set_xscale("log")
fig2.savefig("inset_secondary_late.png")
