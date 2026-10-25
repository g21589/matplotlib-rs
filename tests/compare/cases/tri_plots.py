import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

rng = np.random.default_rng(9)
x = rng.random(60)
y = rng.random(60)
z = np.sin(4 * x) * np.cos(3 * y)
tri = mtri.Triangulation(x, y)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.5))
a1.triplot(tri, "bo-", lw=0.5, ms=2)
a2.tripcolor(tri, z, shading="flat")
a3.tricontourf(tri, z, levels=8)
a3.tricontour(tri, z, levels=8, colors="k", linewidths=0.5)
fig.savefig("tri_plots.png")
