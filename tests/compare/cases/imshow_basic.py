import numpy as np
import matplotlib.pyplot as plt

z = np.arange(100).reshape(10, 10)
fig, ax = plt.subplots()
im = ax.imshow(z)
fig.colorbar(im)
fig.savefig("imshow_basic.png")
