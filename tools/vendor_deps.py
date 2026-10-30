"""Vendor the pure-Python dependencies of matplotlib into matplotlib_rs._vendor.

Usage: python tools/vendor_deps.py

matplotlib_rs needs only NumPy at run time.  cycler, pyparsing and
python-dateutil (with six, which dateutil needs) are copied from the
installed packages at the exact versions in VERSIONS; absolute imports of the
vendored packages are rewritten to point into ``matplotlib_rs._vendor`` (each
rewrite rule must match, so an upstream change is noticed).  Their licenses
are copied to licenses/.
"""
import importlib.metadata
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DST = ROOT / "python" / "matplotlib_rs" / "_vendor"
VENDOR = "matplotlib_rs._vendor"

VERSIONS = {"cycler": "0.12.1", "pyparsing": "3.3.3", "python-dateutil": "3.0.0",
            "six": "1.17.0"}
# Subpackages without run-time use (railroad diagrams need jinja2, AI notes,
# name conversion tool, tz database rebuild script).
SKIP = {"pyparsing/diagram", "pyparsing/ai", "pyparsing/tools", "pyparsing/py.typed",
        "dateutil/zoneinfo/rebuild.py"}

# (package, regex, replacement, expected number of matches)
REWRITES = [
    ("dateutil", r"(?m)^(\s*)import six$", rf"\1from {VENDOR} import six", 7),
    ("dateutil", r"(?m)^(\s*)from six((?:\.moves)?) import ", rf"\1from {VENDOR}.six\2 import ", 11),
    ("dateutil", r"(?m)^(\s*)from dateutil((?:\.\w+)*) import ", rf"\1from {VENDOR}.dateutil\2 import ", 13),
    ("pyparsing", r"(?m)^(\s+)from pyparsing import Word$", r"\1from .core import Word", 1),
]
HEADER = "# Vendored by tools/vendor_deps.py from {dist} {version} (see licenses/{license}).\n"
LICENSES = {"cycler": "LICENSE_CYCLER", "pyparsing": "LICENSE_PYPARSING",
            "python-dateutil": "LICENSE_DATEUTIL", "six": "LICENSE_SIX"}


def license_file(dist):
    files = [f for f in importlib.metadata.distribution(dist).files
             if f.name.upper().startswith("LICENSE") and ".dist-info" in str(f)]
    if len(files) != 1:
        raise SystemExit(f"{dist}: expected one license file, found {files}")
    return pathlib.Path(files[0].locate())


def sources(dist, top):
    root = pathlib.Path(importlib.metadata.distribution(dist).locate_file(top))
    if root.is_file():
        return root.parent, [root]
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    base = root.parent
    keep = [p for p in files
            if not any(str(p.relative_to(base)).startswith(s) for s in SKIP)]
    return base, keep


def main():
    for dist, version in VERSIONS.items():
        found = importlib.metadata.version(dist)
        if found != version:
            raise SystemExit(f"{dist} {version} is required, {found} is installed")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    counts = {i: 0 for i in range(len(REWRITES))}
    for dist, top, target in [("cycler", "cycler/__init__.py", "cycler.py"),
                              ("six", "six.py", "six.py"),
                              ("pyparsing", "pyparsing", None),
                              ("python-dateutil", "dateutil", None)]:
        base, files = sources(dist, top)
        header = HEADER.format(dist=dist, version=VERSIONS[dist], license=LICENSES[dist])
        for src in files:
            rel = pathlib.Path(target) if target else src.relative_to(base)
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix != ".py":
                shutil.copyfile(src, out)
                continue
            text = src.read_text(encoding="utf-8")
            for i, (pkg, pattern, repl, _) in enumerate(REWRITES):
                if rel.parts[0] == pkg:
                    text, n = re.subn(pattern, repl, text)
                    counts[i] += n
            out.write_text(header + text, encoding="utf-8")
        shutil.copyfile(license_file(dist), ROOT / "licenses" / LICENSES[dist])
    for i, (pkg, pattern, _, expected) in enumerate(REWRITES):
        if counts[i] != expected:
            raise SystemExit(f"{pkg}: {pattern!r} matched {counts[i]} times, expected {expected}")
    # (cycler's and pyparsing's remaining matches are docstring examples)
    leftover = [f"{p}:{n}" for p in [DST / "six.py", *(DST / "dateutil").rglob("*.py")]
                for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                if re.match(r"\s*(from|import)\s+(six|dateutil)\b", line)]
    if leftover:
        raise SystemExit(f"unrewritten imports: {leftover}")
    (DST / "__init__.py").write_text(
        '"""Pure-Python packages used by matplotlib, vendored so that matplotlib_rs needs\n'
        'only NumPy at run time (regenerate with tools/vendor_deps.py).\n\n'
        + "".join(f"- {dist} {version}\n" for dist, version in VERSIONS.items())
        + '"""\n', encoding="utf-8")
    print(f"vendored {', '.join(f'{d} {v}' for d, v in VERSIONS.items())} into {DST}")


if __name__ == "__main__":
    sys.exit(main())
