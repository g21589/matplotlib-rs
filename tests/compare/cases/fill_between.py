import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots()
ax.fill_between(x, np.sin(x), np.sin(x) + 1, alpha=0.4)
ax.plot(x, np.sin(x))
ax.axhline(0.5, color="r")
ax.axvspan(2, 3, color="g", alpha=0.2)
fig.savefig("fill_between.png")
