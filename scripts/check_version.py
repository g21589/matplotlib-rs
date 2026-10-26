"""Check the Python, Cargo and optional release tag versions before publishing."""
import argparse
from pathlib import Path
import re
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, e.g. v0.0.2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    cargo = tomllib.loads((root / "Cargo.toml").read_text())["package"]
    lock = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    locked = next(p["version"] for p in lock if p["name"] == cargo["name"])
    version = project["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"Expected a stable X.Y.Z version, got {version!r}")
    if len({version, cargo["version"], locked}) != 1:
        raise SystemExit("pyproject.toml, Cargo.toml and Cargo.lock versions differ")
    if args.tag is not None:
        if args.tag != f"v{version}":
            raise SystemExit(f"Tag {args.tag!r} does not match v{version}")
        if not (root / f"docs/releases/{version}.md").is_file():
            raise SystemExit(f"Missing docs/releases/{version}.md")
    print(version)


if __name__ == "__main__":
    main()
