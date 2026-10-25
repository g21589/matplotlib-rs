"""Run one matplotlib script with matplotlib or matplotlib_rs.

Usage: python runner.py {mpl,rs} script.py outdir

The script is run unmodified (``import matplotlib`` resolves to
matplotlib_rs for ``rs``) with ``outdir`` as working directory, so the files
it saves land there.  Prints the wall time of the script in seconds.
"""
import os
import runpy
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
# COMPARE_RC selects another matplotlibrc directory (e.g. the system one)
os.environ["MATPLOTLIBRC"] = os.environ.get("COMPARE_RC") or os.path.join(here, "..", "mplrc")
os.environ.setdefault("MPLBACKEND", "Agg")

impl, script, outdir = sys.argv[1:4]
script = os.path.abspath(script)
if impl == "rs":
    sys.path.insert(0, os.path.join(here, "..", "..", "python"))
    import matplotlib_rs
    matplotlib_rs.install()
# COMPARE_VECTOR=svg,pdf,eps: every PNG saved is also saved in these formats
# (reproducibly: fixed SVG hash salt and timestamps)
vector = [e for e in os.environ.get("COMPARE_VECTOR", "").split(",") if e]
if vector:
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    import matplotlib
    import matplotlib.figure as mfigure
    matplotlib.rcParams["svg.hashsalt"] = "compare"
    orig_savefig = mfigure.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        ret = orig_savefig(self, fname, *args, **kwargs)
        if isinstance(fname, str) and fname.endswith(".png") and "format" not in kwargs:
            kw = {k: v for k, v in kwargs.items() if k not in ("pil_kwargs", "metadata")}
            for ext in vector:
                orig_savefig(self, fname[:-4] + "." + ext, *args, **kw)
        return ret

    mfigure.Figure.savefig = savefig
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
t0 = time.perf_counter()
runpy.run_path(script, run_name="__main__")
print(f"time {time.perf_counter() - t0:.6f}")
