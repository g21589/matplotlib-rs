import os, sys
import tempfile
os.environ["MATPLOTLIBRC"] = os.path.join(os.path.dirname(__file__), "..", "mplrc")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_rs import _mplrs
import rcconv

rcconv.setup(_mplrs, matplotlib)

fig = plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
fig.canvas.draw()
ref = np.asarray(fig.canvas.buffer_rgba()).copy()

f = _mplrs.Figure()
ax = f.axes(f.add_subplot(1, 1, 0, 0))
line = _mplrs.Line2D(np.array([1.0, 2, 3]), np.array([1.0, 4, 9]))
line.set("color", matplotlib.colors.to_rgba("C0"))
ax.add_line(line)
ax.request_autoscale_view()
out = f.render()
print("shapes", ref.shape, out.shape)
diff = np.any(ref != out, axis=2)
print("differing pixels:", int(diff.sum()))
if diff.any():
    ys, xs = np.nonzero(diff)
    print("bbox of diffs: x", xs.min(), xs.max(), "y", ys.min(), ys.max())
    from PIL import Image
    output = tempfile.mkdtemp(prefix="mplrs-model-")
    Image.fromarray(ref).save(os.path.join(output, "ref.png"))
    Image.fromarray(out).save(os.path.join(output, "out.png"))
    print("diagnostic images:", output)
sys.exit(1 if diff.any() else 0)
