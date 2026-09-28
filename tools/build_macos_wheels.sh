#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# macOS wheels without CI, built as [tool.cibuildwheel.macos] in pyproject.toml builds
# them: conda-forge's OpenMP runtime, macOS 11 and later, libomp bundled by delocate.
#
#   tools/build_macos_wheels.sh PYTHON [PYTHON ...]
#
# builds one wheel per Python executable given (e.g. python3.10 ... python3.14 of
# this machine's architecture) into dist/ (OUT=... for another folder).
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: tools/build_macos_wheels.sh PYTHON [PYTHON ...]" >&2
    exit 1
fi
root=$(cd "$(dirname "$0")/.." && pwd)
out=${OUT:-$root/dist}
arch=$(uname -m)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

bash "$root/tools/ci/install_libomp_macos.sh" "$work/libomp" "$arch"
export MACOSX_DEPLOYMENT_TARGET=11.0
for py in "$@"; do
    "$py" -m pip wheel --no-deps --wheel-dir "$work/raw" -Ccmake.define.OpenMP_ROOT="$work/libomp" "$root"
done

"$1" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet delocate
mkdir -p "$out"
"$work/venv/bin/delocate-wheel" --require-archs "$arch" --wheel-dir "$out" "$work"/raw/*.whl
ls -1 "$out"/*macosx*"$arch"*.whl
