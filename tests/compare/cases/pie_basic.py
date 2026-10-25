import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.pie([30, 20, 50], labels=["a", "b", "c"], autopct="%1.1f%%")
fig.savefig("pie_basic.png")
