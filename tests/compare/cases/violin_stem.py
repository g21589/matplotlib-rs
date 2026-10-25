import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10, 3.5))
a1.violinplot([rng.normal(0, s, 100) for s in (1, 2, 3)], showmeans=True)
a2.stem(np.arange(10), np.sin(np.arange(10)))
a3.broken_barh([(1, 2), (4, 1)], (0, 1), facecolors="tab:blue")
a3.broken_barh([(0.5, 3)], (1.5, 1), facecolors=("tab:red",))
fig.savefig("violin_stem.png")
