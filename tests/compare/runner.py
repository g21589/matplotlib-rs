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
os.environ["MATPLOTLIBRC"] = os.path.join(here, "..", "mplrc")
os.environ.setdefault("MPLBACKEND", "Agg")

impl, script, outdir = sys.argv[1:4]
script = os.path.abspath(script)
if impl == "rs":
    sys.path.insert(0, os.path.join(here, "..", "..", "python"))
    import matplotlib_rs
    matplotlib_rs.install()
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
t0 = time.perf_counter()
runpy.run_path(script, run_name="__main__")
print(f"time {time.perf_counter() - t0:.6f}")
