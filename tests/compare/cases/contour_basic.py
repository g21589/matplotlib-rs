import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
z = np.exp(-x ** 2 - y ** 2)
fig, ax = plt.subplots()
cs = ax.contour(x, y, z, levels=6)
ax.clabel(cs)
fig.savefig("contour_basic.png")
