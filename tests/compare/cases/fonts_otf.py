import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# OpenType/CFF fonts: text, ticks, legend, rotation, weights, mathtext
mpl.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
x = np.linspace(0, 10, 200)
axs[0].plot(x, np.sin(x), label="sin(x) — Inter")
axs[0].plot(x, np.cos(x), label="cos(x) Ångström")
axs[0].set_title("Inter Regular, AVATAR Tj fi")
axs[0].set_xlabel("time (s)", fontweight="bold")
axs[0].set_ylabel("amplitude", style="italic")
axs[0].legend(title="Legend", loc="lower left")
axs[0].text(5, 0.5, r"$\alpha^2 + \beta_i$ in math", ha="center")
axs[0].tick_params(axis="x", labelrotation=35)
for i, (w, fam) in enumerate([(100, "Inter"), (300, "Inter Display"), (600, "Inter"),
                              (900, "Inter Display")]):
    axs[1].text(0.05, 0.9 - 0.15 * i, f"weight {w}: Hamburgefonstiv 0123",
                fontweight=w, family=fam, fontsize=11 + i)
axs[1].text(0.05, 0.3, "สวัสดีครับ Loma", family="Loma", fontsize=14)
axs[1].text(0.05, 0.15, "Unifont pixels ABC", family="Unifont", fontsize=16)
axs[1].text(0.95, 0.05, "rotated", rotation=30, ha="right", family="Inter", style="italic")
axs[1].set_axis_off()
fig.suptitle("OpenType CFF fonts", fontsize=16, fontweight="semibold")
fig.savefig("fonts_otf.png", dpi=110)
