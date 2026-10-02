# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the Python examples of the documentation pages.

Usage::

    python tools/check_docs.py [page.md ...]
    python tools/check_docs.py --api [name filter]

Every fenced ``python`` block of every page under ``docs/`` (or of the pages
named) runs in order, the blocks of one page sharing a namespace, in a
temporary working directory (files the examples write go there). A block
preceded by an ``<!-- no-run -->`` comment is skipped. Exits with status 1 if
an example fails. With ``--api`` the ``Examples`` sections of the docstrings
of the public API (functions, classes and their methods) run instead.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)\s*(?P<lang>[\w+-]*)[^\n]*$")


def blocks(text: str):
    """(first line number, code) of the runnable python blocks of a page."""
    lines = text.splitlines()
    i = 0
    skip_next = False
    while i < len(lines):
        line = lines[i]
        if line.strip() == "<!-- no-run -->":
            skip_next = True
            i += 1
            continue
        m = FENCE.match(line)
        if not m:
            if line.strip():
                skip_next = skip_next and line.strip().startswith("<!--")
            i += 1
            continue
        indent, fence, lang = m.group("indent"), m.group("fence"), m.group("lang")
        body = []
        j = i + 1
        while j < len(lines) and not (lines[j].strip().startswith(fence) and lines[j].strip().strip(fence[0]) == ""):
            body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j].lstrip())
            j += 1
        if lang in ("python", "py") and not skip_next:
            yield i + 1, textwrap.dedent("\n".join(body))
        skip_next = False
        i = j + 1


def run_page(path: Path) -> list[str]:
    errors = []
    namespace: dict = {"__name__": "__docs__"}
    cwd = os.getcwd()
    # ignore_cleanup_errors: lazy (dask) arrays can keep files open on Windows
    with tempfile.TemporaryDirectory(prefix="landschaft_docs_", ignore_cleanup_errors=True) as tmp:
        os.chdir(tmp)
        try:
            for line, code in blocks(path.read_text(encoding="utf-8")):
                try:
                    exec(compile(code, f"{path.relative_to(ROOT)}:{line}", "exec"), namespace)
                except Exception:  # noqa: BLE001 - reported per block
                    errors.append(f"{path.relative_to(ROOT)}:{line}\n{traceback.format_exc(limit=4)}")
                    break  # later blocks usually depend on this one
        finally:
            import gc

            import matplotlib.pyplot as plt

            plt.close("all")
            namespace.clear()
            gc.collect()
            os.chdir(cwd)
    return errors


def public_objects():
    """(qualified name, object) of the public API: landschaft, landschaft.conn,
    landschaft.circuit, landschaft.nlm, landschaft.bio, landschaft.cost and the public
    methods of their classes."""
    import inspect

    import landschaft
    import landschaft.circuit
    import landschaft.conn
    import landschaft.cost
    import landschaft.bio
    import landschaft.nlm
    import landschaft.prioritize
    import landschaft.sdm
    import landschaft.surface

    seen = set()
    for mod in (landschaft, landschaft.conn, landschaft.circuit, landschaft.nlm, landschaft.bio, landschaft.cost,
                landschaft.surface, landschaft.prioritize, landschaft.sdm):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if id(obj) in seen or not callable(obj):
                continue
            seen.add(id(obj))
            yield f"{mod.__name__}.{name}", obj
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_") and attr != "__init__":
                        continue
                    member = getattr(member, "__func__", member)
                    if callable(member) or isinstance(member, property):
                        yield f"{mod.__name__}.{name}.{attr}", member


def run_api(filter_text: str = "") -> list[str]:
    """Run the Examples section of every public docstring as a doctest: the
    outputs shown are compared (whitespace normalised, ``...`` matches
    anything); ``# doctest: +SKIP`` lines are not run."""
    import doctest
    import io

    errors = []
    parser = doctest.DocTestParser()
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    n = 0
    for qualname, obj in public_objects():
        if filter_text and filter_text not in qualname:
            continue
        doc = getattr(obj, "__doc__", None) or ""
        if "Examples" not in doc:
            continue
        section = doc[doc.index("Examples"):]
        test = parser.get_doctest(section, {"__name__": "__api__"}, qualname, qualname, 0)
        if not test.examples:
            continue
        n += 1
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="landschaft_api_", ignore_cleanup_errors=True) as tmp:
            os.chdir(tmp)
            out = io.StringIO()
            runner = doctest.DocTestRunner(optionflags=flags)
            try:
                result = runner.run(test, out=out.write)
                if result.failed:
                    errors.append(qualname + "\n" + out.getvalue())
            finally:
                import gc

                import matplotlib.pyplot as plt

                plt.close("all")
                test.globs.clear()
                gc.collect()
                os.chdir(cwd)
    print(f"{n} docstring examples run, {len(errors)} failed")
    return errors


def main() -> None:
    os.environ.setdefault("LANDSCHAFT_PLOT_SHOW", "0")  # the examples' ls.plot must not open windows
    if sys.argv[1:2] == ["--api"]:
        errors = run_api(sys.argv[2] if len(sys.argv) > 2 else "")
        for e in errors:
            print("\n" + e)
        sys.exit(1 if errors else 0)
    pages = [Path(p).resolve() for p in sys.argv[1:]] or sorted(
        p for p in DOCS.rglob("*.md") if "overrides" not in p.parts)
    failed = []
    for page in pages:
        t = time.perf_counter()
        errs = run_page(page)
        n = sum(1 for _ in blocks(page.read_text(encoding="utf-8")))
        if n:
            status = "FAILED" if errs else "ok"
            print(f"{status:6s} {page.relative_to(ROOT)} ({n} blocks, {time.perf_counter() - t:.1f} s)")
        failed += errs
    for e in failed:
        print("\n" + e)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
