"""Generate the landschaft "Raster Ridge" logo set.

Everything is parametric: edit PALETTE, the RIDGES heights or the SUN block and rerun.

    python tools/generate_ridge_logo.py --out assets/logos --font path/to/Poppins-Medium.ttf

Outputs
    landschaft_ridge_icon.svg                 icon (app-icon badge)
    landschaft_ridge_icon_dark.svg            icon with a hairline border for dark backgrounds
    landschaft_ridge_logo_{light,dark}.svg    horizontal lockup, text outlined
    landschaft_ridge_stacked_{light,dark}.svg stacked lockup (README), text outlined
    editable/landschaft_ridge_editable.svg    layered master: named groups, live text, palette

Text is converted to outlines when fontTools and the Poppins font are available, so the
SVGs render identically everywhere. Without them the lockups fall back to live text.
"""
from __future__ import annotations

import argparse
import os

# ------------------------------------------------------------------ design
PALETTE = {
    "sky": "#0f172a",        # badge / night sky (slate-900)
    "ridge_back": "#15803d",  # green-700
    "ridge_mid": "#22c55e",   # green-500
    "ridge_front": "#a3e635", # lime-400
    "sun": "#a3e635",
    "ink_light": "#0f172a",   # wordmark on light backgrounds
    "ink_dark": "#f8fafc",    # wordmark on dark backgrounds
    "border_dark": "#1e293b", # hairline around the badge on dark backgrounds
}
SIZE = 128        # icon canvas
PX = 8            # raster cell size -> 16 x 16 grid
RADIUS = 28       # badge corner radius
# column heights in cells, left -> right (16 columns)
RIDGES = [
    ("ridge_back", [5, 6, 7, 8, 9, 10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8]),
    ("ridge_mid", [3, 4, 4, 5, 6, 5, 5, 6, 7, 8, 9, 9, 8, 8, 7, 7]),
    ("ridge_front", [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9]),  # rises like a benchmark
]
SUN = (24, 24, 16, 2)       # x, y, size, corner radius
GRID_OPACITY = 0.35
GRID_WIDTH = 0.8
WORD = "LANDSCHAFT"
TRACKING = 7                # extra space between letters at font size 46
FONT_SIZE = 46


# ------------------------------------------------------------------ helpers
def fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def ridge_path(heights: list[int]) -> str:
    """Stepped silhouette as one closed path (easy to recolour in any editor)."""
    d = [f"M0,{SIZE}"]
    for c, h in enumerate(heights):
        d.append(f"V{SIZE - h * PX}H{(c + 1) * PX}")
    d.append(f"V{SIZE}Z")
    return "".join(d)


def grid_path() -> str:
    n = SIZE // PX
    v = "".join(f"M{c * PX},0V{SIZE}" for c in range(1, n))
    h = "".join(f"M0,{r * PX}H{SIZE}" for r in range(1, n))
    return v + h


def icon_group(uid: str, dark: bool = False, labels: bool = False) -> str:
    """The badge. `labels` adds Inkscape/Figma layer names."""
    def lab(name):
        return f' id="{uid}-{name}" inkscape:label="{name}" inkscape:groupmode="layer"' if labels else ""
    x, y, s, r = SUN
    out = [f'<defs><clipPath id="{uid}-clip"><rect width="{SIZE}" height="{SIZE}" rx="{RADIUS}"/></clipPath></defs>',
           f'<g{lab("sky")}><rect width="{SIZE}" height="{SIZE}" rx="{RADIUS}" fill="{PALETTE["sky"]}"/></g>',
           f'<g clip-path="url(#{uid}-clip)"{lab("landscape")}>']
    for key, heights in RIDGES:
        out.append(f'<path{lab(key)} d="{ridge_path(heights)}" fill="{PALETTE[key]}"/>' if labels
                   else f'<path d="{ridge_path(heights)}" fill="{PALETTE[key]}"/>')
    out.append(f'<path{lab("grid")} d="{grid_path()}" fill="none" stroke="{PALETTE["sky"]}" '
               f'stroke-width="{GRID_WIDTH}" opacity="{GRID_OPACITY}"/>' if labels else
               f'<path d="{grid_path()}" fill="none" stroke="{PALETTE["sky"]}" '
               f'stroke-width="{GRID_WIDTH}" opacity="{GRID_OPACITY}"/>')
    out.append("</g>")
    out.append(f'<rect{lab("sun")} x="{x}" y="{y}" width="{s}" height="{s}" rx="{r}" fill="{PALETTE["sun"]}"/>'
               if labels else f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="{r}" fill="{PALETTE["sun"]}"/>')
    if dark:
        out.append(f'<rect x=".75" y=".75" width="{SIZE - 1.5}" height="{SIZE - 1.5}" rx="{RADIUS - .75}" '
                   f'fill="none" stroke="{PALETTE["border_dark"]}" stroke-width="1.5"/>')
    return "\n".join(out)


class Wordmark:
    """Outlines text with fontTools; falls back to live <text>."""

    def __init__(self, font_path: str | None):
        self.font = None
        if font_path and os.path.exists(font_path):
            try:
                from fontTools.ttLib import TTFont
                self.font = TTFont(font_path)
            except ImportError:
                pass

    def measure(self, text, size, tracking):
        if not self.font:
            return len(text) * size * 0.68 + tracking * (len(text) - 1)
        s = size / self.font["head"].unitsPerEm
        cmap, hmtx = self.font.getBestCmap(), self.font["hmtx"]
        return sum(hmtx[cmap[ord(ch)]][0] * s for ch in text) + tracking * (len(text) - 1)

    def render(self, text, size, x, y, tracking, fill):
        if not self.font:
            return (f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="Poppins, sans-serif" font-weight="500" '
                    f'font-size="{size}" letter-spacing="{tracking}" fill="{fill}">{text}</text>')
        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.transformPen import TransformPen
        gs, cmap, hmtx = self.font.getGlyphSet(), self.font.getBestCmap(), self.font["hmtx"]
        s = size / self.font["head"].unitsPerEm
        d, cx = [], x
        for ch in text:
            g = cmap[ord(ch)]
            pen = SVGPathPen(gs, ntos=fmt)
            gs[g].draw(TransformPen(pen, (s, 0, 0, -s, cx, y)))
            d.append(pen.getCommands())
            cx += hmtx[g][0] * s + tracking
        return f'<path d="{" ".join(d)}" fill="{fill}"/>'


def svg(w, h, body, title="landschaft", extra_ns=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg"{extra_ns} viewBox="0 0 {fmt(w)} {fmt(h)}" '
            f'width="{fmt(w)}" height="{fmt(h)}" role="img" aria-label="{title}">\n<title>{title}</title>\n'
            f'{body}\n</svg>\n')


# ------------------------------------------------------------------ outputs
def horizontal(wm: Wordmark, dark: bool, pad=16, gap=28):
    ink = PALETTE["ink_dark" if dark else "ink_light"]
    tw = wm.measure(WORD, FONT_SIZE, TRACKING)
    tx = pad + SIZE + gap
    base = pad + SIZE / 2 + FONT_SIZE * 0.36   # cap height of Poppins ~0.7em -> optical centre
    W, H = tx + tw + pad, SIZE + 2 * pad
    uid = f"lsr-h{'d' if dark else 'l'}"
    body = (f'<g transform="translate({pad},{pad})">\n{icon_group(uid, dark)}\n</g>\n'
            + wm.render(WORD, FONT_SIZE, tx, base, TRACKING, ink))
    return svg(W, H, body)


def stacked(wm: Wordmark, dark: bool, pad=24, icon=160, gap=30):
    ink = PALETTE["ink_dark" if dark else "ink_light"]
    tw = wm.measure(WORD, FONT_SIZE, TRACKING)
    W = max(icon, tw) + 2 * pad
    H = pad + icon + gap + FONT_SIZE * 0.72 + pad
    k = icon / SIZE
    uid = f"lsr-s{'d' if dark else 'l'}"
    body = (f'<g transform="translate({fmt((W - icon) / 2)},{pad}) scale({fmt(k)})">\n{icon_group(uid, dark)}\n</g>\n'
            + wm.render(WORD, FONT_SIZE, (W - tw) / 2, pad + icon + gap + FONT_SIZE * 0.72, TRACKING, ink))
    return svg(W, H, body)


def editable():
    """Layered master for Inkscape / Figma / Illustrator: named layers, live text, swatches."""
    ns = ' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'
    W, H = 900, 560
    swatches = []
    for i, (name, col) in enumerate(PALETTE.items()):
        x = 40 + i * 104
        swatches.append(f'<rect x="{x}" y="470" width="88" height="40" rx="6" fill="{col}" stroke="#cbd5e1"/>'
                        f'<text x="{x}" y="528" font-family="Poppins, sans-serif" font-size="10" fill="#475569">{name}</text>'
                        f'<text x="{x}" y="542" font-family="monospace" font-size="10" fill="#475569">{col}</text>')
    body = f'''<rect width="{W}" height="{H}" fill="#ffffff"/>
<g id="icon" inkscape:label="icon" inkscape:groupmode="layer" transform="translate(40,40)">
{icon_group("lsr-ed", labels=True)}
</g>
<g id="icon-large" inkscape:label="icon (2x preview)" inkscape:groupmode="layer" transform="translate(40,200) scale(1.6)">
{icon_group("lsr-ed2")}
</g>
<g id="wordmark-light" inkscape:label="wordmark (light bg) - live text, font Poppins Medium" inkscape:groupmode="layer">
<text x="196" y="121" font-family="Poppins, sans-serif" font-weight="500" font-size="{FONT_SIZE}" letter-spacing="{TRACKING}" fill="{PALETTE["ink_light"]}">{WORD}</text>
</g>
<g id="wordmark-dark" inkscape:label="wordmark (dark bg) - live text" inkscape:groupmode="layer">
<rect x="300" y="220" width="560" height="160" rx="16" fill="{PALETTE["sky"]}"/>
<text x="330" y="317" font-family="Poppins, sans-serif" font-weight="500" font-size="{FONT_SIZE}" letter-spacing="{TRACKING}" fill="{PALETTE["ink_dark"]}">{WORD}</text>
</g>
<g id="palette" inkscape:label="palette" inkscape:groupmode="layer">
<text x="40" y="455" font-family="Poppins, sans-serif" font-size="12" font-weight="500" fill="#0f172a">Palette - grid 16 x 16 cells of 8 px, badge 128 px, radius 28</text>
{"".join(swatches)}
</g>'''
    return svg(W, H, body, title="landschaft - Raster Ridge (editable master)", extra_ns=ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets/logos")
    ap.add_argument("--font", default=None, help="Poppins-Medium.ttf (for outlined text)")
    a = ap.parse_args()
    wm = Wordmark(a.font)
    os.makedirs(os.path.join(a.out, "editable"), exist_ok=True)
    files = {
        "landschaft_ridge_icon.svg": svg(SIZE, SIZE, icon_group("lsr-i")),
        "landschaft_ridge_icon_dark.svg": svg(SIZE, SIZE, icon_group("lsr-id", dark=True)),
        "landschaft_ridge_logo_light.svg": horizontal(wm, False),
        "landschaft_ridge_logo_dark.svg": horizontal(wm, True),
        "landschaft_ridge_stacked_light.svg": stacked(wm, False),
        "landschaft_ridge_stacked_dark.svg": stacked(wm, True),
        "editable/landschaft_ridge_editable.svg": editable(),
    }
    for name, content in files.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(content)
        print("wrote", os.path.join(a.out, name))
    if not wm.font:
        print("note: fontTools/Poppins not found - lockups use live text (needs Poppins installed to render right)")


if __name__ == "__main__":
    main()
