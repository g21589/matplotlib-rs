import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
X, Y = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
Z = np.exp(-X**2 - Y**2)
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_title("surface")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
t = np.linspace(0, 6 * np.pi, 200)
ax2.plot(np.cos(t), np.sin(t), t / 10, label="helix")
ax2.scatter(np.cos(t[::10]), np.sin(t[::10]), t[::10] / 10, c=t[::10], cmap="plasma")
ax2.legend()
fig.savefig("plot3d_basic.png")
