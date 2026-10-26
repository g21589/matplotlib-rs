#!/bin/sh
# PyO3 extension modules omit libpython linkage; Linux test executables need it.
set -eu
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
python_libdir=$("$PYTHON" -c '
import pathlib, sys, sysconfig
name = sysconfig.get_config_var("LDLIBRARY")
paths = [pathlib.Path(sys.base_prefix) / "lib", pathlib.Path(sysconfig.get_config_var("LIBDIR"))]
print(next(p for p in paths if (p / name).is_file()))
')
python_libname=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY").removeprefix("lib").removesuffix(".so"))')
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-L native=$python_libdir -l $python_libname -C link-arg=-Wl,-rpath,$python_libdir" \
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/rust-tests}" cargo test --locked "$@"
