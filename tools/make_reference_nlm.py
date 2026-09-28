# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden neutral-landscape-model outputs with the R package NLMR.

Usage::

    python tools/make_reference_nlm.py [rng] [models]

``rng`` stores draws of R's generator (``runif``, ``rnorm``, ``sample.int``
after ``set.seed``) and of ``std::mt19937`` with libstdc++'s distributions,
the latter from ``tools/nlm_stdcxx_probe.cpp`` compiled with the Rtools g++
(the compiler that builds NLMR's C++ code), in
``tests/data/reference/nlm/rng.npz``. ``models`` stores NLMR's landscapes for
the cases in ``tests/nlm_cases.py`` in ``tests/data/reference/nlm/models.npz``.

Needs Rscript (``LANDSCHAFT_RSCRIPT``) and an R library with NLMR
(``LANDSCHAFT_NLM_R_LIBS``, default ``%LOCALAPPDATA%/R/landschaft-nlm-lib``:
``R CMD INSTALL --library=<lib> NLMR`` from a clone of ropensci/NLMR), and for
``rng`` the Rtools g++ (``LANDSCHAFT_RTOOLS``, default ``C:/rtools44``). The
test-suite reads the stored files, so it runs anywhere without R.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference  # noqa: E402
import nlm_cases  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "nlm"
PROBE = ROOT / "tools" / "nlm_stdcxx_probe.cpp"
DEFAULT_NLM_LIBS = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "R" / "landschaft-nlm-lib"


def r_environment() -> dict[str, str]:
    env = dict(os.environ)
    lib = env.get("LANDSCHAFT_NLM_R_LIBS", str(DEFAULT_NLM_LIBS))
    env["R_LIBS"] = lib + os.pathsep + env.get("R_LIBS", "")
    return env


def run_r(code: str) -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    with tempfile.TemporaryDirectory(prefix="nlm_ref_") as tmp:
        script = Path(tmp) / "run.R"
        script.write_text(code, encoding="utf-8")
        subprocess.run([str(rscript), str(script)], check=True, env=r_environment())


def r_literal(x) -> str:
    if x is None:
        return "NULL"
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "c(" + ", ".join(r_literal(v) for v in x) + ")"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    v = float(x)
    # hexadecimal: R's parser may not round long decimals correctly
    return repr(v) if v.is_integer() or len(repr(v)) <= 12 else v.hex()


def r_op(op: tuple) -> str:
    kind = op[0]
    if kind in ("runif", "rnorm"):
        return f"{kind}({op[1]}, {r_literal(op[2])}, {r_literal(op[3])})"
    if kind == "rpois":
        return f"as.numeric(rpois({op[1]}, {r_literal(op[2])}))"
    if kind == "rexp":
        return f"rexp({op[1]})"
    if kind == "sample":
        return f"as.numeric(sample.int({r_literal(float(op[1]))}, {op[2]}, {r_literal(op[3])}, {r_literal(op[4])}))"
    raise ValueError(kind)


# -- rng -----------------------------------------------------------------------


def make_rng() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="nlm_rng_") as tmp:
        tmp = Path(tmp)
        lines = []
        keys = []
        for name, ops in nlm_cases.R_SEQUENCES.items():
            for seed in nlm_cases.R_SEEDS:
                lines.append(f"set.seed({seed})")
                for k, op in enumerate(ops):
                    key = f"r__{name}__{seed}__{k}"
                    path = (tmp / f"{key}.bin").as_posix()
                    lines.append(f"writeBin(as.double({r_op(op)}), '{path}')")
                    keys.append(key)
        run_r("\n".join(lines) + "\n")
        for key in keys:
            out[key] = np.fromfile(tmp / f"{key}.bin", dtype="<f8")

        rtools = Path(os.environ.get("LANDSCHAFT_RTOOLS", "C:/rtools44"))
        gxx = rtools / "x86_64-w64-mingw32.static.posix" / "bin" / "g++.exe"
        exe = tmp / "probe.exe"
        env = dict(os.environ)
        env["PATH"] = str(rtools / "usr" / "bin") + os.pathsep + str(gxx.parent) + os.pathsep + env["PATH"]
        # R's flags for C++ packages (Makeconf of R 4.4 with Rtools 4.4).
        subprocess.run(
            [str(gxx), "-std=gnu++17", "-O2", "-mfpmath=sse", "-msse2", "-mstackrealign", "-static",
             str(PROBE), "-o", str(exe)],
            check=True, env=env,
        )
        commands = []
        std_keys = []
        for name, ops in nlm_cases.STD_SEQUENCES.items():
            for seed in nlm_cases.STD_SEQUENCE_SEEDS.get(name, nlm_cases.STD_SEEDS):
                commands.append(f"seed {seed}")
                for k, op in enumerate(ops):
                    commands.append(" ".join(str(v) for v in op))
                    std_keys.append((f"std__{name}__{seed}__{k}", op[1]))
        (tmp / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
        subprocess.run([str(exe), str(tmp / "commands.txt"), str(tmp / "std.bin")], check=True, env=env)
        values = np.fromfile(tmp / "std.bin", dtype="<f8")
        pos = 0
        for key, n in std_keys:
            out[key] = values[pos : pos + n]
            pos += n
        assert pos == values.size
    return out


# -- models ----------------------------------------------------------------------

MODEL_PRELUDE = r"""
suppressMessages({library(NLMR); library(terra)})
save_raster <- function(r, path) {
  m <- as.matrix(r, wide = TRUE)
  writeBin(as.double(c(nrow(m), ncol(m))), paste0(path, ".dim"))
  writeBin(as.double(t(m)), paste0(path, ".bin"))
}
"""


def make_models() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="nlm_models_") as tmp:
        tmp = Path(tmp)
        lines = [MODEL_PRELUDE]
        for key, code in nlm_cases.MODEL_CASES.items():
            path = (tmp / key).as_posix()
            lines.append(f"save_raster({{{code}}}, '{path}')")
        run_r("\n".join(lines) + "\n")
        for key in nlm_cases.MODEL_CASES:
            dim = np.fromfile(tmp / f"{key}.dim", dtype="<f8").astype(int)
            out[key] = np.fromfile(tmp / f"{key}.bin", dtype="<f8").reshape(dim)
    return out


def make_rmath() -> dict[str, np.ndarray]:
    """R's exp(x), x^y and vector sqrt(x) of the inputs in nlm_cases.RMATH."""
    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="nlm_rmath_") as tmp:
        tmp = Path(tmp)
        lines = []
        for key, (fn, x, y) in nlm_cases.rmath_cases().items():
            x.astype("<f8").tofile(tmp / f"{key}.in")
            src, dst = (tmp / f"{key}.in").as_posix(), (tmp / f"{key}.out").as_posix()
            lines.append(f"x <- readBin('{src}', 'double', {x.size})")
            expr = {"exp": "exp(x)", "sqrt": "sqrt(x)", "pow": f"x^{r_literal(y)}"}[fn]
            lines.append(f"writeBin({expr}, '{dst}')")
            out[f"{key}__x"] = x
        run_r("\n".join(lines) + "\n")
        for key in nlm_cases.rmath_cases():
            out[key] = np.fromfile(tmp / f"{key}.out", dtype="<f8")
    return out


def make_fft() -> dict[str, np.ndarray]:
    """R's fft() of the complex matrices in nlm_cases.fft_cases()."""
    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="nlm_fft_") as tmp:
        tmp = Path(tmp)
        lines = []
        for key, (re, im, nrow, ncol, inverse) in nlm_cases.fft_cases().items():
            re.astype("<f8").tofile(tmp / f"{key}.re")
            im.astype("<f8").tofile(tmp / f"{key}.im")
            n = nrow * ncol
            lines.append(f"z <- matrix(complex(real = readBin('{(tmp / f'{key}.re').as_posix()}', 'double', {n}), "
                         f"imaginary = readBin('{(tmp / f'{key}.im').as_posix()}', 'double', {n})), {nrow}, {ncol})")
            lines.append(f"f <- fft(z, inverse = {r_literal(inverse)})")
            lines.append(f"writeBin(as.vector(rbind(as.vector(Re(f)), as.vector(Im(f)))), '{(tmp / f'{key}.out').as_posix()}')")
        run_r("\n".join(lines) + "\n")
        for key in nlm_cases.fft_cases():
            out[key] = np.fromfile(tmp / f"{key}.out", dtype="<f8")
    return out


def main(argv: list[str]) -> None:
    parts = argv or ["rng", "rmath", "models"]
    OUT.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [str(lsm_reference.find_rscript()), "-e", "cat(as.character(packageVersion('NLMR')), R.version.string)"],
        check=True, capture_output=True, text=True, env=r_environment(),
    ).stdout.strip()
    info_path = OUT / "versions.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if "rng" in parts:
        # integers (samples, shuffles) as int32 to keep the file small
        rng = {k: (v.astype(np.int32) if np.all(v == np.round(v)) and np.all(np.abs(v) < 2**31) else v)
               for k, v in make_rng().items()}
        np.savez_compressed(OUT / "rng.npz", **rng)
        info["rng"] = version
    if "rmath" in parts:
        np.savez_compressed(OUT / "rmath.npz", **make_rmath(), **make_fft())
        info["rmath"] = version
    if "models" in parts and nlm_cases.MODEL_CASES:
        np.savez_compressed(OUT / "models.npz", **make_models())
        info["models"] = version
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print("NLMR", version)


if __name__ == "__main__":
    main(sys.argv[1:])
