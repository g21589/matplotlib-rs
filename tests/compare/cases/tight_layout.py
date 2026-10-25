import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3)
for ax in axs.flat:
    ax.plot(np.arange(10))
    ax.set_title("title")
    ax.set_ylabel("ylabel")
fig.tight_layout()
fig.savefig("tight_layout.png")
