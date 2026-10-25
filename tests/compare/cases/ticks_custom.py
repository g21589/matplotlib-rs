import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

fig, ax = plt.subplots()
ax.plot(np.arange(20), np.arange(20) ** 2)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.tick_params(axis="x", direction="in", length=8, colors="r")
ax.set_xlim(-1, 21)
ax.set_xticks([0, 5, 10, 15, 20], ["zero", "five", "ten", "15", "20"])
fig.savefig("ticks_custom.png")
