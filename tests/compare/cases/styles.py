import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 50)
for style in ["ggplot", "seaborn-v0_8-darkgrid", "dark_background", "bmh", "fivethirtyeight"]:
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(4, 3))
        for k in range(4):
            ax.plot(x, np.sin(x + k), label=str(k))
        ax.legend()
        ax.set_title(style)
        fig.savefig(f"styles_{style}.png")
        plt.close(fig)
