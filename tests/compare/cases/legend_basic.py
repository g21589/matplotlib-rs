import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), "--", label="cos(x)")
ax.plot(x, np.sin(2 * x), marker="s", markevery=10, label="sin(2x)")
ax.legend()
fig.savefig("legend_basic.png")
