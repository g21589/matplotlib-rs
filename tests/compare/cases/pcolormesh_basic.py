import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 25)
y = np.linspace(-1, 1, 15)
X, Y = np.meshgrid(x, y)
Z = np.exp(-X**2 - Y**2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.5))
pc = a1.pcolormesh(X, Y, Z, shading="auto", cmap="plasma")
fig.colorbar(pc, ax=a1)
a2.pcolor(Z[:6, :8], edgecolors="k", linewidths=0.5)
fig.savefig("pcolormesh_basic.png")
