#!/bin/sh
# Build the Rust extension and install it into the Python package.
set -e
cd "$(dirname "$0")/.."
cargo build --release
SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
cp target/release/lib_mplrs.so "python/matplotlib_rs/_mplrs$SUFFIX"
