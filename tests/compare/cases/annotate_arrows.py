import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.arange(0.0, 5.0, 0.01)
ax.plot(t, np.cos(2 * np.pi * t), lw=2)
ax.annotate("local max", xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor="black", shrink=0.05))
ax.annotate("fancy", xy=(1, -1), xytext=(0.2, -1.6),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.3"))
ax.annotate("box", xy=(4, 1), xytext=(3.5, -1.5),
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="fancy", fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
ax.set_ylim(-2, 2)
fig.savefig("annotate_arrows.png")
