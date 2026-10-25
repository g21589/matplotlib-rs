import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# rotated flat and gouraud meshes, at a dpi where cell edges hit pixel
# boundaries; rasterization below a zorder; gids and urls on artists
d = np.arange(100).reshape(10, 10)
x, y = np.meshgrid(np.arange(11), np.arange(11))
theta = 0.25 * np.pi
xx = x * np.cos(theta) - y * np.sin(theta)
yy = x * np.sin(theta) + y * np.cos(theta)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained")
ax1.set_aspect(1)
ax1.pcolormesh(xx, yy, d)
ax2.set_aspect(1)
ax2.pcolormesh(xx, yy, d, rasterized=True)
xc, yc = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 8))
ax3.pcolormesh(xc, yc, np.hypot(xc, yc), shading="gouraud", cmap="RdBu")
ax3.text(0.5, 0.5, "Text", alpha=0.2, va="center", ha="center", size=30,
         transform=ax3.transAxes)
ax4.set_aspect(1)
ax4.pcolormesh(xx, yy, d, zorder=-10)
ax4.text(0.5, 0.5, "Text", alpha=0.2, va="center", ha="center", size=30,
         transform=ax4.transAxes)
ax4.set_rasterization_zorder(0)
l1, = ax4.plot([0, 5], [0, 5], "r-", lw=3)
shadow, = ax4.plot([0, 5], [0, 5])
shadow.update_from(l1)
shadow.set_color("0.2")
shadow.set_zorder(l1.get_zorder() - 0.5)
shadow.set_transform(mtransforms.offset_copy(l1.get_transform(), fig, x=2.0, y=-3.0,
                                             units="points"))
shadow.set_gid("shadow")
l1.set_url("https://matplotlib.org/")
# (a text url needs autohinted metrics for the PDF link box: not yet)
ax4.set_title("linked")
fig.savefig("mesh_raster_links.png", dpi=150)
