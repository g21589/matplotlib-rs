"""Run a script with matplotlib replaced by matplotlib_rs.

Usage: python -m matplotlib_rs script.py [args...]
       python -m matplotlib_rs -m module [args...]
"""
import runpy
import sys


def main():
    import matplotlib_rs
    matplotlib_rs.install()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "-m":
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
