import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
t = np.linspace(0, 2 * np.pi, 100)
axs[0].plot(np.cos(t), np.sin(t))
axs[0].set_aspect("equal")
axs[1].plot(np.cos(t), 2 * np.sin(t))
axs[1].axis("equal")
axs[2].plot(t, np.sin(t))
axs[2].axis("off")
fig.savefig("aspect_axis_off.png")
