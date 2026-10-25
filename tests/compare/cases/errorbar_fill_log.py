import numpy as np
import matplotlib.pyplot as plt

x = np.logspace(0, 3, 15)
y = x ** 1.5
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
a1.errorbar(x, y, yerr=y * 0.3, xerr=x * 0.1, fmt="s", capsize=4, ecolor="gray")
a1.set_xscale("log")
a1.set_yscale("log")
a2.semilogy(np.arange(20), np.exp(np.arange(20) / 3))
a2.fill_between(np.arange(20), np.exp(np.arange(20) / 3) * 0.5, np.exp(np.arange(20) / 3) * 2, alpha=0.3)
a2.set_yscale("symlog")
fig.savefig("errorbar_fill_log.png")
