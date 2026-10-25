import numpy as np
import matplotlib.pyplot as plt

x = np.logspace(0, 4, 50)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
ax1.loglog(x, x ** 1.5)
ax2.semilogy(np.arange(50), np.exp(np.arange(50) / 5))
fig.savefig("log_scale.png")
