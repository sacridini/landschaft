# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the Python examples of the documentation pages.

Usage::

    python tools/check_docs.py [page.md ...]

Every fenced ``python`` block of every page under ``docs/`` (or of the pages
named) runs in order, the blocks of one page sharing a namespace, in a
temporary working directory (files the examples write go there). A block
preceded by an ``<!-- no-run -->`` comment is skipped. Exits with status 1 if
an example fails.
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


def main() -> None:
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
