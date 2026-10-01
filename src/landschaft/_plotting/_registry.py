# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots made for particular results (registered by class name)."""

from __future__ import annotations

from collections.abc import Callable

_PLOTTERS: dict[str, Callable] = {}


def register(*names: str):
    """Register a function (obj, opts, title) -> list of panels for the classes named."""

    def deco(fn):
        for n in names:
            _PLOTTERS[n] = fn
        return fn

    return deco


def panels(obj, opts, title):
    """The registered plot of obj's class (or of a base class), or None."""
    from . import _results  # noqa: F401  (registers the plots of the results)

    for cls in type(obj).__mro__:
        fn = _PLOTTERS.get(cls.__name__)
        if fn is not None:
            return fn(obj, opts, title)
    return None
