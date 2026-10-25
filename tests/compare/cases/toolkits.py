import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
import mpl_toolkits.axisartist as AA

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(2, 2, 1)
im = ax.imshow(np.arange(100).reshape(10, 10))
div = make_axes_locatable(ax)
cax = div.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
top = div.append_axes("top", size="25%", pad=0.1, sharex=ax)
top.plot(np.arange(10), np.arange(10) ** 0.5)

grid = ImageGrid(fig, 222, nrows_ncols=(2, 2), axes_pad=0.1, cbar_mode="single")
for a in grid:
    imm = a.imshow(np.random.default_rng(0).random((5, 5)), vmin=0, vmax=1)
grid.cbar_axes[0].colorbar(imm)

ax3 = fig.add_subplot(2, 2, 3)
x = np.linspace(0, 10, 400)
ax3.plot(x, np.sin(x) * np.exp(-x / 5))
axins = zoomed_inset_axes(ax3, 3, loc="upper right")
axins.plot(x, np.sin(x) * np.exp(-x / 5))
axins.set_xlim(1, 2)
axins.set_ylim(0.6, 0.9)
mark_inset(ax3, axins, loc1=2, loc2=4, fc="none", ec="0.5")
inset_axes(ax3, width="30%", height="20%", loc="lower left").plot([1, 2, 3])

ax4 = fig.add_subplot(2, 2, 4, axes_class=AA.Axes)
ax4.axis["right"].set_visible(False)
ax4.axis["top"].set_visible(False)
ax4.axis["y=0"] = ax4.new_floating_axis(nth_coord=0, value=0)
ax4.plot(x, np.sin(x))
fig.savefig("toolkits.png")
