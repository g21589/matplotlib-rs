import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:15j, -3:3:15j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
a1.quiver(X, Y, U, V, np.hypot(U, V))
a2.streamplot(X, Y, U, V, density=1.0, color=U, cmap="autumn")
fig.savefig("quiver_stream.png")
