# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the original Biodiverse (Perl) for reference results.

Biodiverse iterates hashes in Perl's randomised order: sums such as the PD
(``sum values %$nodes_in_path``) change in the last bit from run to run. The
harness can run an *overlay* of the Biodiverse library in which every
``keys``/``values``/``each`` over a hash goes through helper functions that
return the keys in sorted (string) order. landschaft numbers labels, groups and
tree nodes in that same order, so the overlay results are the bit-for-bit
target; the unpatched library agrees to ~1e-12.

Environment:

- ``LANDSCHAFT_BIODIVERSE``: Biodiverse clone (default
  ``~/Documents/dev/biodiverse``);
- ``LANDSCHAFT_STRAWBERRY``: Strawberry Perl (default ``C:/Strawberry``);
- ``LANDSCHAFT_BIO_PERL5``: private CPAN library with Biodiverse's dependencies
  (default ``%LOCALAPPDATA%/landschaft-bio-perl5``, filled with
  ``cpanm -L <dir> --notest ...``; see IDEAS.md).

The overlay lives in ``<LANDSCHAFT_BIO_PERL5>/overlay`` and is rebuilt when the
Biodiverse sources change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bio_reference.pl"
LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
BIODIVERSE = Path(os.environ.get("LANDSCHAFT_BIODIVERSE", Path.home() / "Documents" / "dev" / "biodiverse"))
STRAWBERRY = Path(os.environ.get("LANDSCHAFT_STRAWBERRY", "C:/Strawberry"))
PERL5 = Path(os.environ.get("LANDSCHAFT_BIO_PERL5", LOCAL / "landschaft-bio-perl5"))
OVERLAY = PERL5 / "overlay"

# Perl helpers: sorted keys, context-aware like the builtins.
LSK_MODULE = r"""package LSK;
use strict;
use warnings;
use Scalar::Util qw /refaddr/;

my %iterators;

sub k {
    my $h = $_[0];
    delete $iterators{refaddr $h};
    return wantarray ? (sort keys %$h) : scalar keys %$h;
}

sub v {
    my $h = $_[0];
    delete $iterators{refaddr $h};
    return wantarray ? @{$h}{sort keys %$h} : scalar keys %$h;
}

sub kv {
    my $h = $_[0];
    delete $iterators{refaddr $h};
    return map { ($_, $h->{$_}) } sort keys %$h;
}

sub e {
    my $h = $_[0];
    my $addr = refaddr $h;
    my $state = $iterators{$addr} //= [[sort keys %$h], 0];
    if ($state->[1] >= @{$state->[0]}) {
        delete $iterators{$addr};
        return;
    }
    my $k = $state->[0][$state->[1]++];
    return wantarray ? ($k, $h->{$k}) : $k;
}

1;
"""

_OP = re.compile(r"(?<![\w$@%>:])(keys|values|each)(\s+)(?=%)")
_FOREACH_VALUES = re.compile(r"\b(for|foreach)(\s*(?:my\s*)?(?:\$\w+)?\s*\(\s*)values\s+(?=%)")
# whole hashes flattened in list context: for my ($k, $v) (%h), pairs %h, pairmap {...} %h
_HASH_LIST = re.compile(r"(\b(?:for|foreach)\s+my\s*\([^()]*\)\s*\(\s*|\bpairs\s*\(?\s*|\bpairmap\s*\{[^{}]*\}\s*)(?=%)")


def _hash_expr(src: str, i: int) -> int | None:
    """End of the hash expression starting with ``%`` at ``i`` (or None)."""
    assert src[i] == "%"
    j = i + 1
    if j < len(src) and src[j] == "{":
        depth = 0
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return None
    if j < len(src) and src[j] == "$":
        j += 1
    m = re.compile(r"[A-Za-z_]\w*(?:::\w+)*").match(src, j)
    if not m:
        return None
    end = m.end()
    # %$h{...} / %h{...} are key/value slices: leave them alone
    if end < len(src) and src[end] in "{[":
        return None
    return end


def _rewrite_hash_lists(src: str) -> tuple[str, int]:
    out = []
    count = 0
    i = 0
    for m in _HASH_LIST.finditer(src):
        h0 = m.end()
        if h0 < i:
            continue
        # "% $h" (with a space) is valid Perl too
        k = h0 + 1
        while k < len(src) and src[k] == " ":
            k += 1
        compact = "%" + src[k:k + 400]
        h1 = _hash_expr(compact, 0)
        if h1 is None or h1 >= len(compact):
            continue
        out.append(src[i:h0])
        out.append("LSK::kv(\\" + compact[:h1] + ")")
        i = k + h1 - 1
        count += 1
    out.append(src[i:])
    return "".join(out), count


def rewrite_perl(src: str) -> tuple[str, int]:
    """Route hash iteration through the LSK helpers; returns (code, count)."""
    src, count = _rewrite_hash_lists(src)
    out = []
    i = 0
    while True:
        m_each = _OP.search(src, i)
        m_for = _FOREACH_VALUES.search(src, i)
        # foreach (values %h) keeps aliasing: sorted hash slice instead of a copy
        if m_for and (m_each is None or m_for.start() <= m_each.start()):
            h0 = m_for.end()
            h1 = _hash_expr(src, h0)
            if h1 is None:
                out.append(src[i:m_for.end()])
                i = m_for.end()
                continue
            expr = src[h0:h1]
            out.append(src[i:m_for.start()])
            out.append(f"{m_for.group(1)}{m_for.group(2)}@{{\\{expr}}}{{sort keys {expr}}}")
            i = h1
            count += 1
            continue
        if m_each is None:
            out.append(src[i:])
            break
        h0 = m_each.end()
        h1 = _hash_expr(src, h0)
        # keys %h = N presizes the hash: lvalue, keep
        if h1 is None or re.match(r"\s*=(?![=~>])", src[h1:h1 + 4]):
            out.append(src[i:h0])
            i = h0
            continue
        expr = src[h0:h1]
        fn = {"keys": "k", "values": "v", "each": "e"}[m_each.group(1)]
        out.append(src[i:m_each.start()])
        out.append(f"LSK::{fn}(\\{expr})")
        i = h1
        count += 1
    return "".join(out), count


def _source_digest() -> str:
    h = hashlib.sha256()
    h.update(LSK_MODULE.encode())
    h.update(Path(__file__).read_bytes())
    for p in sorted((BIODIVERSE / "lib").rglob("*.pm")):
        h.update(str(p.relative_to(BIODIVERSE)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def build_overlay(force: bool = False) -> Path:
    """Copy the Biodiverse library with sorted hash iteration; returns its lib dir."""
    lib = OVERLAY / "lib"
    stamp = OVERLAY / "digest.txt"
    digest = _source_digest()
    if not force and stamp.exists() and stamp.read_text() == digest:
        return lib
    if OVERLAY.exists():
        shutil.rmtree(OVERLAY)
    shutil.copytree(BIODIVERSE / "lib", lib)
    total = 0
    for p in lib.rglob("*.pm"):
        if "GUI" in p.relative_to(lib).parts:
            continue
        src = p.read_text(encoding="utf-8")
        new, n = rewrite_perl(src)
        if n:
            p.write_text(new, encoding="utf-8", newline="")
            total += n
    (lib / "LSK.pm").write_text(LSK_MODULE, encoding="utf-8")
    stamp.write_text(digest)
    print(f"[bio_harness] overlay built: {total} hash iterations rewritten")
    return lib


def perl_environment(hash_seed: int = 0) -> dict[str, str]:
    env = dict(os.environ)
    bins = [STRAWBERRY / "perl" / "site" / "bin", STRAWBERRY / "perl" / "bin", STRAWBERRY / "c" / "bin"]
    env["PATH"] = os.pathsep.join(str(b) for b in bins) + os.pathsep + env.get("PATH", "")
    # shim/ holds placeholders for optional modules (after the real library)
    env["PERL5LIB"] = os.pathsep.join([str(PERL5 / "lib" / "perl5"), str(PERL5 / "shim")])
    gdal = os.environ.get("LANDSCHAFT_BIO_GDAL", str(Path.home() / "miniforge3" / "envs" / "geo" / "Library"))
    env["LANDSCHAFT_BIO_GDAL"] = gdal
    env["PATH"] = str(Path(gdal) / "bin") + os.pathsep + env["PATH"]
    env["PERL_HASH_SEED"] = str(hash_seed)
    env["PERL_PERTURB_KEYS"] = "0"
    return env


def available() -> bool:
    return (STRAWBERRY / "perl" / "bin" / "perl.exe").exists() and (BIODIVERSE / "lib" / "Biodiverse").is_dir() and (PERL5 / "lib" / "perl5").is_dir()


def run_job(job: dict, *, sorted_order: bool = True, hash_seed: int = 0, verbose: bool = False) -> dict:
    """Run ``tools/bio_reference.pl`` on ``job``; returns the decoded results."""
    lib = build_overlay() if sorted_order else BIODIVERSE / "lib"
    perl = STRAWBERRY / "perl" / "bin" / "perl.exe"
    with tempfile.TemporaryDirectory(prefix="bio_ref_") as tmp:
        job_file = Path(tmp) / "job.json"
        out_file = Path(tmp) / "out.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")
        cmd = [str(perl), "-I", str(lib)]
        if sorted_order:
            cmd += ["-MLSK"]
        cmd += [str(SCRIPT), str(job_file), str(out_file)]
        proc = subprocess.run(cmd, env=perl_environment(hash_seed), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if verbose or proc.returncode:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
        if proc.returncode:
            raise RuntimeError(f"Biodiverse run failed ({proc.returncode})")
        return decode(json.loads(out_file.read_text(encoding="utf-8")))


def decode(x):
    """Turn the "Inf"/"NaN" strings of the Perl encoder back into floats."""
    if isinstance(x, dict):
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v) for v in x]
    if x in ("Inf", "-Inf", "NaN"):
        return float(x.replace("Inf", "inf").replace("NaN", "nan"))
    return x
