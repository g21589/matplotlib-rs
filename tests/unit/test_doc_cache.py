"""The precomputed Artist docs (_doc_cache.py) must equal live computation."""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
code = r'''
import sys
sys.path.insert(0, "python")
import matplotlib_rs
matplotlib_rs.install()
import matplotlib.pyplot, mpl_toolkits.mplot3d, mpl_toolkits.axisartist, matplotlib.tri
import matplotlib.artist as martist
from matplotlib_rs import _doc_cache
bad = []
for (mod, qual), (doc, setters) in _doc_cache.DOCS.items():
    m = sys.modules.get(mod)
    if m is None:
        continue
    cls = m
    for part in qual.split("."):
        cls = getattr(cls, part, None)
    if cls is None:
        bad.append((mod, qual, "missing"))
        continue
    if martist.ArtistInspector(cls).get_setters() != setters:
        bad.append((mod, qual, "setters"))
    import os
    os.environ["MPLRS_NO_DOC_CACHE"] = "1"
    if martist.kwdoc(cls) != doc:
        bad.append((mod, qual, "kwdoc"))
    del os.environ["MPLRS_NO_DOC_CACHE"]
print(len(_doc_cache.DOCS), "classes;", "stale:", bad[:10])
sys.exit(1 if bad else 0)
'''
r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(here, "..", ".."))
sys.exit(r.returncode)
