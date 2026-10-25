import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
people = ["Tom", "Dick", "Harry", "Slim", "Jim"]
y = np.arange(len(people))
perf = np.array([3, 7, 5, 9, 2.0])
ax.barh(y, perf, xerr=0.5, align="center", color="tab:green")
ax.barh(y, perf / 2, left=perf, color="tab:olive")
ax.set_yticks(y, labels=people)
ax.invert_yaxis()
ax.set_xlabel("Performance")
fig.savefig("barh_stacked.png")
