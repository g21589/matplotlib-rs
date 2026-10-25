import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained", figsize=(8, 4))
sf1, sf2 = fig.subfigures(1, 2)
sf1.set_facecolor("0.9")
ax = sf1.subplots()
ax.plot([1, 2, 3])
sf1.suptitle("left")
axs = sf2.subplots(2, 1)
axs[0].bar([1, 2], [3, 4])
axs[1].scatter([1, 2, 3], [3, 1, 2])
sf2.suptitle("right")
fig.savefig("subfigures.png")
