# SPDX-License-Identifier: GPL-3.0-or-later
"""The compiled circuit functions: the AVX2 build when the CPU has AVX2 and FMA."""

from __future__ import annotations

import os
from functools import lru_cache

from .. import _core


@lru_cache(maxsize=None)
def core():
    """landschaft._circuit_avx2 where it runs (unless LANDSCHAFT_CIRCUIT_AVX2=0), else landschaft._core."""
    if os.environ.get("LANDSCHAFT_CIRCUIT_AVX2", "1") != "0" and _core.cpu_supports_avx2_fma():
        try:
            from .. import _circuit_avx2
        except ImportError:
            return _core
        return _circuit_avx2
    return _core
