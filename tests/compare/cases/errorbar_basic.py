import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
fig, ax = plt.subplots()
ax.errorbar(x, x ** 0.5, yerr=0.2, xerr=0.1, fmt="o", capsize=3)
fig.savefig("errorbar_basic.png")
