# SPDX-License-Identifier: GPL-3.0-or-later
"""Element names as Biodiverse builds them.

Biodiverse identifies groups and labels by strings: the axis values joined with
``:`` by Text::CSV_XS (quote character ``'``), numbers written the way Perl
stringifies a double (``%.15g``). Sorting these strings gives the order in
which landschaft numbers groups and labels (the order the reference runs use).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np

JOIN_CHAR = ":"
QUOTE_CHAR = "'"

_BINARY = re.compile(r"[\x00-\x08\x0a-\x1f\x7f-\xa0]")


def perl_number(x: float) -> str:
    """A double as Perl stringifies it (``%.15g``; ``Inf``/``NaN``)."""
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    s = format(x, ".15g")
    if s == "-0":
        return "0"
    return s


def perl_numify(s: str) -> float:
    """The number Perl reads from a string that looks like a number."""
    return float(s)


def csv_field(value: str, sep: str = JOIN_CHAR, quote: str = QUOTE_CHAR) -> str:
    """One field as Text::CSV_XS ``combine`` writes it for Biodiverse.

    Biodiverse asks for ``quote_space => 0`` but filters that option out, so
    CSV_XS's default applies: fields with spaces or tabs are quoted too.
    """
    if value == "":
        return ""
    if sep in value or quote in value or " " in value or "\t" in value or _BINARY.search(value):
        return quote + value.replace(quote, quote + quote) + quote
    return value


def join_element(values: Sequence[str], sep: str = JOIN_CHAR, quote: str = QUOTE_CHAR) -> str:
    """Element name from its axis values (``list2csv``)."""
    return sep.join(csv_field(v, sep, quote) for v in values)


def dequote_element(name: str, quote: str = QUOTE_CHAR) -> str:
    """Biodiverse ``dequote_element``: strip one level of quotes from simple names."""
    q = re.escape(quote)
    if re.fullmatch(rf"{q}[^{q}\s]+{q}", name):
        return name[1:-1]
    return name


def split_element(name: str, sep: str = JOIN_CHAR, quote: str = QUOTE_CHAR) -> list[str]:
    """Axis values of an element name (``csv2list``)."""
    out: list[str] = []
    i = 0
    n = len(name)
    while True:
        if i < n and name[i] == quote:
            j = i + 1
            buf = []
            while j < n:
                if name[j] == quote:
                    if j + 1 < n and name[j + 1] == quote:
                        buf.append(quote)
                        j += 2
                        continue
                    break
                buf.append(name[j])
                j += 1
            out.append("".join(buf))
            i = j + 1
        else:
            j = name.find(sep, i)
            if j < 0:
                j = n
            out.append(name[i:j])
            i = j
        if i >= n:
            break
        if name[i] == sep:
            i += 1
            if i == n:
                out.append("")
                break
    return out


def group_name(values: Sequence[float | str]) -> str:
    """Group name from its (snapped) coordinates, as the text import builds it."""
    parts = [perl_number(v) if not isinstance(v, str) else v for v in values]
    name = join_element(parts)
    if len(parts) == 1:
        name = dequote_element(name)
    return name


def label_name(values: Sequence[str]) -> str:
    """Label name from the label columns of a record."""
    name = join_element(["" if v is None else str(v) for v in values])
    if len(values) <= 1:
        name = dequote_element(name)
    return name


def sort_names(names: Iterable[str]) -> list[str]:
    """Perl's default ``sort`` (string comparison by code point)."""
    return sorted(names)


def looks_like_number(s: str) -> bool:
    """Perl's ``looks_like_number`` for strings (decimal, exponent, Inf/NaN)."""
    s = s.strip() if s[:1].isspace() else s
    if re.fullmatch(r"[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|0[xX][0-9a-fA-F]+)\s*", s):
        return not s.lower().startswith(("0x", "+0x", "-0x"))
    return bool(re.fullmatch(r"[+-]?(?:inf(?:inity)?|nan)", s, flags=re.IGNORECASE))
