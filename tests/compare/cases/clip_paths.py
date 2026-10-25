import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, RegularPolygon
from matplotlib.path import Path
from matplotlib.transforms import TransformedPatchPath

# clip paths given as patches, (path, transform) pairs and patch paths, on
# native lines, patches and texts, and on images; logit masking.  (Each
# artist gets its own clip Path object: the PS backend keys clip paths by
# (path, id(temporary affine)), which depends on CPython reusing addresses.)
fig, axs = plt.subplots(2, 2, figsize=(7, 6))
ax = axs[0, 0]
track = Rectangle((0, .25), 1, .5, transform=ax.transAxes, facecolor="lightgrey")
ax.add_patch(track)
ax.axvline(0.5, 0, 1, color="red", lw=5, clip_path=TransformedPatchPath(track))
track2 = Polygon([(0, .25), (1, .25), (1, .75), (0, .75)], transform=ax.transAxes,
                 fc="none", ec="none")
ax.add_patch(track2)
ax.plot([0.2, 0.8], [0.0, 1.0], "b", lw=4, clip_path=TransformedPatchPath(track2))
circ = RegularPolygon((0.3, 0.6), 12, radius=0.25, transform=ax.transAxes, fc="none", ec="g")
ax.add_patch(circ)
t = ax.text(0.3, 0.6, "clipped text", size=18, ha="center", transform=ax.transAxes)
t.set_clip_path(circ)

ax = axs[0, 1]
x = np.linspace(0, 10, 200)
star = Path.unit_regular_star(5, 0.5)
for k in range(5):
    (ln,) = ax.plot(x, np.sin(x + k) * 2, lw=3)
    ln.set_clip_path(Path(star.vertices, star.codes), ax.transData + ax.transData.inverted()
                     + plt.matplotlib.transforms.Affine2D().scale(2.5).translate(5, 0)
                     + ax.transData)
poly = RegularPolygon((5, 0), 6, radius=2.5, fc="y", alpha=0.4)
ax.add_patch(poly)
e = Ellipse((5, 0), 6, 3, fc="c")
ax.add_patch(e)
e.set_clip_path(poly)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 3)

ax = axs[1, 0]
img = np.add.outer(np.linspace(0, 1, 120), np.linspace(0, 1, 160))
im = ax.imshow(img, cmap="viridis")
patch = Circle((80, 60), radius=50, transform=ax.transData)
im.set_clip_path(patch)
ax.axis("off")

ax = axs[1, 1]
ax.plot([0.1, 0.5, 0.9, 1.0], [0.1, 0.5, 0.9, 1.0], "o-")
ax.set_yscale("logit")
fig.savefig("clip_paths.png")
