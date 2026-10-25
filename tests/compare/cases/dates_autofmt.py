import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

base = datetime.datetime(2024, 1, 1)
dates = [base + datetime.timedelta(hours=6 * i) for i in range(100)]
fig, ax = plt.subplots()
ax.plot(dates, np.cumsum(np.random.default_rng(2).normal(size=100)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate()
fig.savefig("dates_autofmt.png")
