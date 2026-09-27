# SPDX-License-Identifier: GPL-3.0-or-later
"""Griffe extension of the documentation: Sphinx roles in docstrings become
Markdown.

The docstrings cross-reference with Sphinx roles (``:func:`mspa```,
``:class:`~landschaft.RasterCollection```). This turns each role into an
autorefs link to the documented object (``[`mspa`][landschaft.mspa]``) when
the target is part of the public API, and into inline code otherwise.
"""

from __future__ import annotations

import re

import griffe

ROLE = re.compile(r":(?:py:)?(func|class|meth|attr|data|mod|obj|exc):`(~?)([^`]+)`")
# Functions that share their name with their module are documented under their full path.
ALIASES = {
    "landschaft.omniscape": "landschaft.circuit.omniscape.omniscape",
    "landschaft.circuit.omniscape": "landschaft.circuit.omniscape.omniscape",
    "landschaft.circuit.advanced": "landschaft.circuit.advanced.advanced",
}
_PUBLIC: set[str] | None = None


def _public() -> set[str]:
    """Qualified names of the documented objects (functions, classes, methods)."""
    global _PUBLIC
    if _PUBLIC is None:
        import inspect

        import landschaft
        import landschaft.circuit
        import landschaft.conn

        names = set()
        for mod in (landschaft, landschaft.conn, landschaft.circuit):
            for name in getattr(mod, "__all__", []):
                obj = getattr(mod, name)
                if not callable(obj):
                    continue
                names.add(f"{mod.__name__}.{name}")
                if inspect.isclass(obj):
                    names.update(f"{mod.__name__}.{name}.{attr}" for attr in vars(obj) if not attr.startswith("_"))
        _PUBLIC = names
    return _PUBLIC


def _resolve(target: str, context: str) -> str | None:
    public = _public()
    target = target.rstrip("()")
    candidates = [target]
    if not target.startswith("landschaft"):
        candidates += [f"landschaft.{target}", f"landschaft.conn.{target}", f"landschaft.circuit.{target}"]
        if context:
            candidates.insert(0, f"{context}.{target}")
    found = next((c for c in candidates if c in public), None)
    return ALIASES.get(found, found)


def convert(text: str, context: str = "") -> str:
    def repl(m: re.Match) -> str:
        role, short, target = m.groups()
        label = target.rstrip("()").split(".")[-1] if short or "." in target else target.rstrip("()")
        if role in ("func", "meth"):
            label += "()" if not label.endswith("()") else ""
        resolved = _resolve(target, context)
        return f"[`{label}`][{resolved}]" if resolved else f"`{label}`"

    return ROLE.sub(repl, text)


class SphinxRoles(griffe.Extension):
    """Convert Sphinx roles of every docstring (and its sections) to Markdown,
    and show the ``n_threads`` keyword that ``@threaded`` adds at run time."""

    def on_instance(self, *, obj: griffe.Object, **kwargs) -> None:  # noqa: ARG002
        if obj.is_function and any(str(d.value).endswith("threaded") for d in obj.decorators):
            if "n_threads" not in obj.parameters:
                params = list(obj.parameters)
                new = griffe.Parameter("n_threads", annotation="int | None", kind=griffe.ParameterKind.keyword_only,
                                       default="None")
                at = next((i for i, p in enumerate(params) if p.kind is griffe.ParameterKind.var_keyword), len(params))
                params.insert(at, new)
                obj.parameters = griffe.Parameters(*params)
        if obj.docstring is None:
            return
        context = obj.parent.path if obj.is_function and obj.parent is not None and obj.parent.is_class else ""
        if obj.is_class:
            context = obj.path
        obj.docstring.value = convert(obj.docstring.value, context)
