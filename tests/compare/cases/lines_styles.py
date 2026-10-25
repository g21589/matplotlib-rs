import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), "r-", label="sin", linewidth=2)
ax.plot(x, np.cos(x), "b--", label="cos")
ax.plot(x[::10], np.sin(x[::10]) * 0.5, "go", label="pts", markersize=4)
ax.plot(x, 0.3 * np.sin(3 * x), ls=":", color="k", alpha=0.6)
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Styles")
ax.grid(True)
fig.savefig("lines_styles.png", dpi=80)
