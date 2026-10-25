import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
z = rng.random((12, 17))
fig, axs = plt.subplots(2, 3, figsize=(9, 6))
axs[0, 0].imshow(z, interpolation="bilinear", origin="lower", cmap="viridis")
axs[0, 1].imshow(z, interpolation="bicubic", extent=(0, 5, 0, 3), aspect="auto", cmap="magma")
m = np.ma.masked_where(z > 0.8, z)
axs[0, 2].imshow(m, interpolation="nearest", cmap="coolwarm", vmin=0.2, vmax=0.9)
rgb = rng.random((8, 8, 3))
axs[1, 0].imshow(rgb)
rgba = rng.random((8, 8, 4))
axs[1, 1].imshow(rgba, interpolation="hanning", alpha=0.6)
big = np.sin(np.linspace(0, 20, 400))[:, None] * np.cos(np.linspace(0, 20, 300))[None, :]
axs[1, 2].imshow(big, cmap="gray")
fig.savefig("imshow_variants.png")
