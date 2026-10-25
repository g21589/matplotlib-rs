import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(5, 3))
ax.set_title("Inter + fallback: ∫ ∮ ≈ ∞ ⊕ ★ ☃ 漢字 สวัสดี")
ax.text(0.05, 0.5, "Missing everywhere: \U0001F600 ☃ ∑x²", fontsize=14)
ax.set_xlabel("θ ∈ [0, 2π] ⟶ ⇒")
fig.savefig("fallback.png")
