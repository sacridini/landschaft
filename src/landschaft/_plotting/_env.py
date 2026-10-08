# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a plot is shown: inline in a notebook, or in a window of its own."""

from __future__ import annotations

import os
import sys
import tempfile
import webbrowser
from pathlib import Path


def in_notebook() -> bool:
    """A Jupyter kernel (Jupyter, JupyterLab, VS Code, Colab): plots show inline."""
    if "google.colab" in sys.modules:
        return True
    ip = sys.modules.get("IPython")
    if ip is None:
        return False
    try:
        shell = ip.get_ipython()
    except Exception:  # noqa: BLE001
        return False
    return shell is not None and type(shell).__name__ in ("ZMQInteractiveShell", "Shell")


def auto_show() -> bool:
    """Open windows by themselves (outside notebooks), unless LANDSCHAFT_PLOT_SHOW=0."""
    return os.environ.get("LANDSCHAFT_PLOT_SHOW", "1") not in ("0", "false", "no") and not in_notebook()


def plotly_url() -> str | bool:
    """plotly's script as a file of the installed package (a window's page loads it from disk,
    without the network), else in the page."""
    try:
        import plotly

        js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    except ImportError:  # pragma: no cover - depends on the environment
        return True
    return js.as_uri() if js.exists() else True


def open_window(html_page: str, title: str = "landschaft", js_api=None) -> Path:
    """A page in a window of its own (pywebview), else in the web browser; the file written.
    ``js_api``: an object whose methods the page calls (pywebview's JavaScript API)."""
    path = Path(tempfile.mkdtemp(prefix="landschaft_plot_")) / "plot.html"
    path.write_text(html_page, encoding="utf-8")
    try:
        import webview
    except ImportError:
        webbrowser.open(path.as_uri())
        return path
    try:
        webview.create_window(title, url=path.as_uri(), width=1280, height=860, js_api=js_api)
        webview.start()
    except Exception:  # noqa: BLE001 - no GUI (a server, missing WebView2...): the browser
        webbrowser.open(path.as_uri())
    return path
