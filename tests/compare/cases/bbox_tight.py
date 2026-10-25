import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 1, 2])
ax.set_title("tight bbox")
fig.savefig("bbox_tight.png", bbox_inches="tight")
