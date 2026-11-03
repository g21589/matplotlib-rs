"""Large images mapped by colormaps: the 'rgba' interpolation stage (data
shrunk more than 3 times, or asked for), which matplotlib_rs resamples
through the colormap's lut, and the cases it leaves to numpy."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

rng = np.random.default_rng(7)
z = rng.normal(size=(600, 800))
z[rng.random(z.shape) < 0.01] = np.nan
z[rng.random(z.shape) < 0.002] = np.inf
z[rng.random(z.shape) < 0.002] = -np.inf
masked = np.ma.masked_where(np.abs(z) > 2.2, z)
extremes = plt.get_cmap("viridis").with_extremes(bad=(1, 0, 0, 0.5), under="w", over=(0, 0, 0, 0.3))
translucent = colors.ListedColormap(rng.random((7, 4)), name="translucent")

fig, axs = plt.subplots(4, 4, figsize=(12, 10))
axs = axs.ravel()
axs[0].imshow(z)
axs[1].imshow(z, cmap=extremes, vmin=-1, vmax=1.5, origin="lower")
axs[2].imshow(masked, cmap="coolwarm", interpolation="bilinear")
axs[3].imshow(z.astype(np.float32), cmap="magma", interpolation="nearest")
axs[4].imshow(np.abs(z) + 1e-3, norm=colors.LogNorm(), cmap="plasma")
axs[5].imshow(z, norm=colors.Normalize(-1, 1, clip=True), cmap=extremes)
axs[6].imshow(np.full((300, 300), 2.5))
axs[7].imshow(z, alpha=0.4, cmap=translucent, extent=(5, 0, 0, 3))
axs[8].imshow(z, alpha=rng.random(z.shape), cmap="gray")
axs[9].imshow(np.clip(np.nan_to_num(z * 40), -500, 500).astype(np.int16), cmap="tab20")
axs[10].imshow(z[:140, :130], interpolation_stage="rgba", interpolation="bicubic", cmap=extremes)
axs[11].imshow(z, norm=colors.TwoSlopeNorm(0.2, -2, 3), cmap="RdBu", interpolation="lanczos")
axs[12].imshow(z, cmap=translucent, interpolation="antialiased", interpolation_stage="rgba")
axs[13].imshow(z, cmap=plt.get_cmap("viridis", 5), vmin=0, vmax=0.5, origin="lower", aspect="auto")
axs[14].imshow(masked.astype(np.float32), cmap=extremes, interpolation="hanning")
axs[15].imshow(z.T, cmap="cividis", vmin=-0.0, vmax=1e-300)
fig.savefig("imshow_large.png")
fig.savefig("imshow_large_dpi.png", dpi=47)
