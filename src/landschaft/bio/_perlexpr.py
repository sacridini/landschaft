# SPDX-License-Identifier: GPL-3.0-or-later
"""A parser for the Perl expressions of Biodiverse's spatial conditions.

Biodiverse evaluates its spatial conditions as Perl code (``$D <= 100000``,
``sp_circle (radius => 3) && $nbr_x > 0``). This module reads the subset of
Perl those conditions use into a small AST: numbers, strings, regexes, array
references, the condition variables, the usual operators with Perl's
precedence, ``my`` variables, a few maths functions and calls to the ``sp_*``
functions with named arguments.

AST nodes are tuples: ``("num", v)``, ``("str", s)``, ``("regex", pattern,
flags)``, ``("list", [items])``, ``("var", name)``, ``("idx", name, expr)``,
``("aref", name)``, ``("un", op, expr)``, ``("bin", op, a, b)``,
``("tern", c, a, b)``, ``("call", name, args)`` (``args`` a list of
``(key, expr)`` pairs for ``sp_*`` calls, of ``(None, expr)`` otherwise),
``("seq", [exprs])`` and ``("my", name, expr)``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = ["PerlSyntaxError", "parse"]


class PerlSyntaxError(ValueError):
    """The condition is not in the supported subset of Perl."""


@dataclass
class Tok:
    kind: str  # num str regex var arrvar ident op eof
    value: object
    pos: int


_OPS = sorted(
    [
        "<=>", "**", "=>", "==", "!=", "<=", ">=", "&&", "||", "//", "=~", "!~", "->",
        "<", ">", "+", "-", "*", "/", "%", "!", "?", ":", "(", ")", "[", "]", "{", "}", ",", ".", "\\", ";", "=",
    ],
    key=len,
    reverse=True,
)
_NUM = re.compile(r"(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:::\w+)*")
_VAR = re.compile(r"\$([A-Za-z_]\w*)")
_ARRVAR = re.compile(r"@([A-Za-z_]\w*)")
_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}
_WORD_OPS = {"and", "or", "not", "xor", "eq", "ne", "lt", "gt", "le", "ge", "cmp", "x"}


def _delimited(src: str, i: int) -> tuple[str, int]:
    """Text between the delimiter at ``i`` and its closing pair; end index after it."""
    open_ = src[i]
    close = _BRACKETS.get(open_, open_)
    depth = 1
    j = i + 1
    out = []
    while j < len(src):
        ch = src[j]
        if ch == "\\" and j + 1 < len(src):
            out.append(src[j:j + 2])
            j += 2
            continue
        if open_ != close and ch == open_:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                return "".join(out), j + 1
        out.append(ch)
        j += 1
    raise PerlSyntaxError(f"unterminated delimiter {open_!r} at {i}")


def _unescape_single(s: str) -> str:
    return re.sub(r"\\([\\'])", r"\1", s)


def _unescape_double(s: str) -> str:
    if "$" in s or "@" in s:
        raise PerlSyntaxError("interpolated strings are not supported")
    return bytes(s, "utf-8").decode("unicode_escape")


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            j = src.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        prev = toks[-1] if toks else None
        # a slash starts a regex after =~ / !~ (and where a term is expected)
        if c == "/" and prev is not None and prev.kind == "op" and prev.value in ("=~", "!~", "(", ",", "=>"):
            body, j = _delimited(src, i)
            m = re.match(r"[msixpn]*", src[j:])
            toks.append(Tok("regex", (body, m.group(0)), i))
            i = j + m.end()
            continue
        m = _NUM.match(src, i)
        if m and (c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit())):
            toks.append(Tok("num", float(m.group(0).replace("_", "")), i))
            i = m.end()
            continue
        if c == "'":
            body, j = _delimited(src, i)
            toks.append(Tok("str", _unescape_single(body), i))
            i = j
            continue
        if c == '"':
            body, j = _delimited(src, i)
            toks.append(Tok("str", _unescape_double(body), i))
            i = j
            continue
        if c == "$":
            m = _VAR.match(src, i)
            if not m:
                raise PerlSyntaxError(f"unexpected '$' at {i}")
            toks.append(Tok("var", m.group(1), i))
            i = m.end()
            continue
        if c == "@":
            m = _ARRVAR.match(src, i)
            if not m:
                raise PerlSyntaxError(f"unexpected '@' at {i}")
            toks.append(Tok("arrvar", m.group(1), i))
            i = m.end()
            continue
        m = _IDENT.match(src, i)
        if m:
            word = m.group(0)
            j = m.end()
            # quote-like operators
            if word in ("q", "qq", "qr", "m", "qw") and j < n:
                k = j
                while k < n and src[k].isspace():
                    k += 1
                if k < n and not src[k].isalnum() and src[k] not in ",;=)" and not (src[k] == "=" and src[k:k + 2] == "=>"):
                    body, e = _delimited(src, k)
                    if word == "q":
                        toks.append(Tok("str", _unescape_single(body), i))
                        i = e
                    elif word == "qq":
                        toks.append(Tok("str", _unescape_double(body), i))
                        i = e
                    elif word == "qw":
                        toks.append(Tok("qw", body.split(), i))
                        i = e
                    else:
                        fm = re.match(r"[msixpn]*", src[e:])
                        toks.append(Tok("regex", (body, fm.group(0)), i))
                        i = e + fm.end()
                    continue
            toks.append(Tok("ident", word, i))
            i = j
            continue
        for op in _OPS:
            if src.startswith(op, i):
                toks.append(Tok("op", op, i))
                i += len(op)
                break
        else:
            raise PerlSyntaxError(f"unexpected character {c!r} at {i}")
    toks.append(Tok("eof", None, n))
    return toks


# binding powers (higher binds tighter), Perl's precedence table
_BINARY = {
    "or": (1, "left"), "xor": (1, "left"),
    "and": (2, "left"),
    "?": (6, "right"),
    "||": (7, "left"), "//": (7, "left"),
    "&&": (8, "left"),
    "==": (11, "none"), "!=": (11, "none"), "<=>": (11, "none"), "eq": (11, "none"), "ne": (11, "none"), "cmp": (11, "none"),
    "<": (12, "none"), ">": (12, "none"), "<=": (12, "none"), ">=": (12, "none"),
    "lt": (12, "none"), "gt": (12, "none"), "le": (12, "none"), "ge": (12, "none"),
    "+": (14, "left"), "-": (14, "left"), ".": (14, "left"),
    "*": (15, "left"), "/": (15, "left"), "%": (15, "left"), "x": (15, "left"),
    "=~": (16, "left"), "!~": (16, "left"),
    "**": (18, "right"),
}
_NAMED_UNARY = {"abs", "sqrt", "sin", "cos", "log", "exp", "int", "defined", "length", "lc", "uc", "deg2rad", "rad2deg"}
_LIST_FUNCS = {"atan2", "min", "max"}
_CONSTANTS = {
    "Math::Trig::pi": math.pi, "Math::Trig::pi2": 2 * math.pi, "Math::Trig::pip2": math.pi / 2,
    "Math::Trig::pip4": math.pi / 4, "Math::Trig::pi4": 4 * math.pi, "pi": math.pi, "pi2": 2 * math.pi,
    "pip2": math.pi / 2, "pip4": math.pi / 4,
}


class _Parser:
    def __init__(self, toks: list[Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self, k: int = 0) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect_op(self, op: str) -> None:
        t = self.next()
        if t.kind != "op" or t.value != op:
            raise PerlSyntaxError(f"expected {op!r} at {t.pos}, got {t.value!r}")

    def at_op(self, *ops: str) -> bool:
        t = self.peek()
        return t.kind == "op" and t.value in ops

    # -- statements -------------------------------------------------------------------
    def program(self):
        stmts = []
        while self.peek().kind != "eof":
            if self.at_op(";"):
                self.next()
                continue
            stmts.append(self.statement())
            if not self.at_op(";") and self.peek().kind != "eof":
                raise PerlSyntaxError(f"unexpected {self.peek().value!r} at {self.peek().pos}")
        if not stmts:
            return ("num", 0.0)
        return stmts[0] if len(stmts) == 1 else ("seq", stmts)

    def statement(self):
        t = self.peek()
        if t.kind == "ident" and t.value == "my":
            self.next()
            v = self.next()
            if v.kind != "var":
                raise PerlSyntaxError(f"expected a scalar after 'my' at {v.pos}")
            self.expect_op("=")
            return ("my", v.value, self.expr(0))
        if t.kind == "ident" and t.value == "return":
            self.next()
        return self.expr(0)

    # -- expressions --------------------------------------------------------------------
    def expr(self, min_bp: int):
        left = self.prefix()
        while True:
            t = self.peek()
            op = t.value if t.kind in ("op", "ident") else None
            if op not in _BINARY or (t.kind == "ident" and op not in _WORD_OPS):
                break
            bp, assoc = _BINARY[op]
            if bp < min_bp:
                break
            self.next()
            if op == "?":
                a = self.expr(0)
                self.expect_op(":")
                b = self.expr(bp)
                left = ("tern", left, a, b)
                continue
            if op in ("=~", "!~"):
                rhs = self.next()
                if rhs.kind != "regex":
                    raise PerlSyntaxError(f"expected a regex after {op} at {rhs.pos}")
                left = ("bin", op, left, ("regex", *rhs.value))
                continue
            right = self.expr(bp + 1 if assoc in ("left", "none") else bp)
            left = ("bin", op, left, right)
        return left

    def prefix(self):
        t = self.next()
        if t.kind == "num":
            return ("num", t.value)
        if t.kind == "str":
            return ("str", t.value)
        if t.kind == "regex":
            return ("regex", *t.value)
        if t.kind == "qw":
            return ("list", [("str", w) for w in t.value])
        if t.kind == "var":
            name = t.value
            if name == "self" and self.at_op("->"):
                self.next()
                m = self.next()
                if m.kind != "ident":
                    raise PerlSyntaxError(f"expected a method name at {m.pos}")
                return self.call(m.value)
            if self.at_op("["):
                self.next()
                idx = self.expr(0)
                self.expect_op("]")
                return ("idx", name, idx)
            return ("var", name)
        if t.kind == "arrvar":
            return ("array", t.value)
        if t.kind == "ident":
            word = t.value
            if self.at_op("=>"):
                return ("str", word)  # bareword hash key
            if word == "not":
                return ("un", "!", self.expr(3))
            if word in _CONSTANTS and not self.at_op("("):
                return ("num", _CONSTANTS[word])
            if word in ("undef",):
                if self.at_op("("):
                    self.next()
                    self.expect_op(")")
                return ("undef",)
            return self.call(word)
        if t.kind == "op":
            op = t.value
            if op == "(":
                if self.at_op(")"):
                    self.next()
                    return ("list", [])
                items = self.list_items(")")
                return items[0] if len(items) == 1 else ("list", items)
            if op == "[":
                return ("list", self.list_items("]"))
            if op == "{":
                return ("hash", self.list_items("}"))
            if op == "!":
                return ("un", "!", self.expr(17))
            if op == "-":
                return ("un", "-", self.expr(17))
            if op == "+":
                return self.expr(17)
            if op == "\\":
                inner = self.expr(17)
                if inner[0] == "array":
                    return ("aref", inner[1])
                return inner
        raise PerlSyntaxError(f"unexpected {t.value!r} at {t.pos}")

    def list_items(self, close: str) -> list:
        items = []
        while not self.at_op(close):
            item = self.expr(5)
            if items and isinstance(items[-1], tuple) and items[-1][0] == "__key":
                items[-1] = ("pair", items[-1][1], item)
            else:
                items.append(item)
            if self.at_op("=>"):
                self.next()
                key = items.pop()
                items.append(("__key", key))
                continue
            if self.at_op(","):
                self.next()
                continue
            if not self.at_op(close):
                raise PerlSyntaxError(f"expected ',' or {close!r} at {self.peek().pos}")
        self.expect_op(close)
        return items

    def call(self, name: str):
        if self.at_op("("):
            self.next()
            items = self.list_items(")")
        elif name in _NAMED_UNARY:
            items = [self.expr(13)]
        else:
            raise PerlSyntaxError(f"unknown bareword {name!r}")
        return ("call", name, items)


def parse(src: str):
    """AST of a condition (see the module docstring)."""
    return _Parser(tokenize(src)).program()
