import datetime
import matplotlib.pyplot as plt

d0 = datetime.datetime(2024, 1, 1)
days = [d0 + datetime.timedelta(days=i) for i in range(60)]
fig, ax = plt.subplots()
ax.plot(days, range(60))
fig.autofmt_xdate()
fig.savefig("dates_basic.png")
