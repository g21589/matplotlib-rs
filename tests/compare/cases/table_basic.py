import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.axis("off")
data = [[66386, 174296, 75131], [58230, 381139, 78045], [89135, 80552, 152558]]
tab = ax.table(cellText=data, rowLabels=["a", "b", "c"], colLabels=["x", "y", "z"], loc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(10)
fig.savefig("table_basic.png")
