import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 50)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), "b")
ax2 = ax.twinx()
ax2.plot(x, np.exp(x / 3), "r")
ax2.set_ylabel("exp")
fig.savefig("twinx_basic.png")
