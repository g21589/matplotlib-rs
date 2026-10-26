"""Boxplot bulk creation, non-default styles, scales and fallback paths."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections import register_projection

rng = np.random.default_rng(21)
data = rng.lognormal(size=(31, 4))

variants = [
    {}, {"patch_artist": True}, {"patch_artist": True, "notch": True},
    {"orientation": "horizontal", "patch_artist": True},
    {"showmeans": True}, {"showmeans": True, "meanline": True},
    {"showbox": False}, {"showcaps": False, "showfliers": False},
    {"whis": (5, 95), "patch_artist": True},
    {"patch_artist": True, "boxprops": {"hatch": "//", "alpha": .5}},
    {"widths": [.1, .2, .3, .4], "capwidths": .2, "manage_ticks": False},
    {"medianprops": {"color": None}},
]
for page in range(3):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for ax, kw in zip(axs.flat, variants[page * 4:page * 4 + 4]):
        out = ax.boxplot(data, **kw)
        out["medians"][0].set_color("purple")
        out["medians"][0].set_linewidth(2)
        if out["fliers"]:
            out["fliers"][0].remove()
        ax.relim()
        ax.autoscale_view()
    fig.savefig(f"boxplot_variants_{page}.png")
    plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs[0, 0].set_yscale("log")
axs[0, 0].boxplot(data, patch_artist=True)
axs[0, 1].boxplot([[], [1, 2, 3], [np.nan, 1, 2]])
axs[1, 0].boxplot(np.ones((20, 3)), autorange=True, notch=True)
axs[1, 1].boxplot(data / 20, positions=[.2, .4, .6, .8], widths=.08,
                 patch_artist=True, boxprops={"transform": axs[1, 1].transAxes})
fig.savefig("boxplot_scales_empty_transform.png")
plt.close(fig)


class CustomAxes(Axes):
    name = "boxplot_custom"
    line_calls = 0

    def plot(self, *args, **kwargs):
        self.line_calls += 1
        return super().plot(*args, **kwargs)


register_projection(CustomAxes)
fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(projection="boxplot_custom")
ax.boxplot(data, patch_artist=True)
assert ax.line_calls == 24  # preserve custom per-line plot hooks
fig.savefig("boxplot_custom_axes.png")
plt.close(fig)
