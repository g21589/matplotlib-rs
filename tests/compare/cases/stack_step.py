import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
fig, (a, b) = plt.subplots(2, 1)
a.stackplot(x, x, x ** 0.5, np.ones(10))
b.step(x, x % 3, where="mid")
b.stem(x, np.sin(x))
fig.savefig("stack_step.png")
