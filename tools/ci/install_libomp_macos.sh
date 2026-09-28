#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The OpenMP runtime for macOS wheels: conda-forge's llvm-openmp, installed into
# PREFIX (include/omp.h, lib/libomp.dylib). Apple's clang has none of its own, and
# Homebrew's libomp is built for the running macOS only, while conda-forge's runs on
# macOS 11 and later, as the wheels do (MACOSX_DEPLOYMENT_TARGET=11.0). delocate
# copies it into the wheel.
#
# Usage: install_libomp_macos.sh PREFIX [arm64|x86_64]   (default: this machine's)
set -euo pipefail

prefix=${1:?usage: install_libomp_macos.sh PREFIX [arm64|x86_64]}
arch=${2:-$(uname -m)}
case "$arch" in
    arm64) platform=osx-arm64 ;;
    x86_64) platform=osx-64 ;;
    *) echo "unknown architecture: $arch" >&2; exit 1 ;;
esac

if [ -f "$prefix/include/omp.h" ] && [ -f "$prefix/lib/libomp.dylib" ]; then
    exit 0
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
host=$([ "$(uname -m)" = arm64 ] && echo osx-arm64 || echo osx-64)
curl -fsSL "https://micro.mamba.pm/api/micromamba/$host/latest" | tar -xj -C "$tmp" bin/micromamba
MAMBA_ROOT_PREFIX="$tmp/root" "$tmp/bin/micromamba" create --yes --quiet --prefix "$prefix" \
    --platform "$platform" --channel conda-forge --override-channels llvm-openmp
test -f "$prefix/lib/libomp.dylib"
