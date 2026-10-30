#!/bin/sh
# Build the extension in release mode and place it next to the Python sources
# so that `PYTHONPATH=python` picks it up (no virtualenv needed).
# PYTHON selects the interpreter to build for (default: python3).
set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
PYO3_PYTHON=${PYO3_PYTHON:-$PYTHON} cargo build --release "$@"
EXT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
cp target/release/lib_mplrs.so "python/matplotlib_rs/_mplrs$EXT"
