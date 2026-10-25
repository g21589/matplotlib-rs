#!/bin/sh
# Build the extension in release mode and place it next to the Python sources
# so that `PYTHONPATH=python` picks it up (no virtualenv needed).
set -e
cd "$(dirname "$0")/.."
cargo build --release "$@"
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
cp target/release/lib_mplrs.so "python/matplotlib_rs/_mplrs$EXT"
