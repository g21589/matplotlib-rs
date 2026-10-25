import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# blitting as interactive code does it: save the background, draw animated
# artists over it with draw_artist, restore (part of) it
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200)
ax.plot(x, np.sin(x), "k-")
(ln,) = ax.plot(x, np.cos(x), "r-", lw=3, animated=True)
txt = ax.text(0.5, 0.5, "frame", transform=ax.transAxes, animated=True, size=20)
fig.canvas.draw()
bg = fig.canvas.copy_from_bbox(ax.bbox)
x1, y1, x2, y2 = bg.get_extents()
assert np.asarray(bg).shape == (y2 - y1, x2 - x1, 4)


def save(name):
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(name)


for k in range(3):
    fig.canvas.restore_region(bg)
    ln.set_ydata(np.cos(x + k))
    txt.set_text(f"frame {k}")
    ax.draw_artist(ln)
    ax.draw_artist(txt)
    fig.canvas.blit(ax.bbox)
save("blit_regions_frames.png")
# part of the background, moved
fig.canvas.restore_region(bg, bbox=(x1 + 50, y1 + 20, x2 - 100, y2 - 60), xy=(x1 - 30, y1 + 10))
save("blit_regions_moved.png")
# a region restored partly outside the canvas (matplotlib leaves the part
# of a region copied from outside the canvas uninitialized: not tested)
r2 = fig.canvas.copy_from_bbox((10.5, 300.2, 120.7, 460))
fig.canvas.restore_region(r2)
r2.set_x(-40)
r2.set_y(400)
fig.canvas.restore_region(r2)
r2.set_x(560)
r2.set_y(-30)
fig.canvas.restore_region(r2)
save("blit_regions_outside.png")
