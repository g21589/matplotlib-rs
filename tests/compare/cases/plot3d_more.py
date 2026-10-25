import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(2, 3, 1, projection="3d")
X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color="k", lw=0.5)
ax.view_init(elev=35, azim=-60)
ax = fig.add_subplot(2, 3, 2, projection="3d")
xs = np.arange(5)
for k, c in enumerate(["r", "g", "b"]):
    ax.bar(xs, np.arange(1, 6) * (k + 1), zs=k, zdir="y", color=c, alpha=0.8)
ax = fig.add_subplot(2, 3, 3, projection="3d")
ax.contour(X, Y, Z, levels=8, cmap="coolwarm")
ax.contourf(X, Y, Z, zdir="z", offset=-2, cmap="coolwarm", levels=8)
ax.set_zlim(-2, 1)
ax = fig.add_subplot(2, 3, 4, projection="3d")
v = np.zeros((4, 4, 4), bool)
v[0:2, 0:2, 0:2] = True
v[2:, 2:, 2:] = True
ax.voxels(v, facecolors="tab:orange", edgecolor="k")
ax = fig.add_subplot(2, 3, 5, projection="3d")
ax.bar3d([0, 1, 2], [0, 1, 0], [0, 0, 0], 0.5, 0.5, [1, 2, 3], shade=True)
ax.text(1, 1, 3.5, "bars", color="r")
ax.set_box_aspect((1, 1, 0.6))
ax = fig.add_subplot(2, 3, 6, projection="3d")
ax.quiver(X[::5, ::5], Y[::5, ::5], Z[::5, ::5], 0.2, 0.2, 0.5, length=1, normalize=True)
ax.set_axis_off()
fig.savefig("plot3d_more.png")
