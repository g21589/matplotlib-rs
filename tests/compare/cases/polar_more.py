import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)
fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(2, 3, 1, projection="polar")
theta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
ax1.bar(theta, rng.random(12) * 10, width=np.pi / 8, bottom=1, alpha=0.6,
        color=plt.cm.viridis(rng.random(12)))
ax2 = fig.add_subplot(2, 3, 2, projection="polar")
ax2.scatter(rng.random(50) * 2 * np.pi, rng.random(50), c=rng.random(50), s=40, cmap="hsv")
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax3 = fig.add_subplot(2, 3, 3, projection="polar")
ax3.plot(theta, np.abs(np.sin(theta)) + 0.2, "o-", label="sin")
ax3.set_thetamin(0)
ax3.set_thetamax(180)
ax3.set_rorigin(-0.5)
ax3.legend(loc="lower left")
for i, proj in enumerate(["aitoff", "hammer", "mollweide"]):
    ax = fig.add_subplot(2, 3, 4 + i, projection=proj)
    ax.plot(np.linspace(-2.5, 2.5, 30), np.sin(np.linspace(-2.5, 2.5, 30)))
    ax.grid(True)
    ax.set_title(proj)
fig.savefig("polar_more.png")
