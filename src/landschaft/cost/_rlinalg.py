# SPDX-License-Identifier: GPL-3.0-or-later
"""R's ``solve()`` of a small matrix, bit for bit.

``solve(a)`` is LAPACK 3.12's ``dgesv`` (``dgetrf2``, the recursive LU with
partial pivoting, then ``dgetrs``) over R's reference BLAS (``dgemm``,
``dtrsm`` without the zero test, ``idamax``, ``dscal``). Python floats are
IEEE doubles, so these loops give R's results; the matrices are small
(one row and column per raster layer).
"""

from __future__ import annotations

import numpy as np


def _dlaswp(a: np.ndarray, col0: int, ncols: int, k1: int, k2: int, ipiv: list[int], ip0: int) -> None:
    """Row interchanges k1..k2 (1-based) of ipiv[ip0 + k - 1] on columns col0..col0+ncols."""
    for i in range(k1, k2 + 1):
        ip = ipiv[ip0 + i - 1]
        if ip != i:
            r1, r2 = i - 1, ip - 1
            for j in range(col0, col0 + ncols):
                a[r1, j], a[r2, j] = a[r2, j], a[r1, j]


def _dgetrf2(a: np.ndarray, r0: int, c0: int, m: int, n: int, ipiv: list[int], p0: int) -> int:
    """LU of the block a[r0:r0+m, c0:c0+n] in place; ipiv[p0:] 1-based, relative to the block."""
    info = 0
    if m == 0 or n == 0:
        return 0
    if m == 1:
        ipiv[p0] = 1
        if a[r0, c0] == 0.0:
            info = 1
    elif n == 1:
        sfmin = 2.2250738585072014e-308  # dlamch('S')
        i = 1
        dmax = abs(a[r0, c0])
        for k in range(2, m + 1):
            if abs(a[r0 + k - 1, c0]) > dmax:
                i, dmax = k, abs(a[r0 + k - 1, c0])
        ipiv[p0] = i
        if a[r0 + i - 1, c0] != 0.0:
            if i != 1:
                a[r0, c0], a[r0 + i - 1, c0] = a[r0 + i - 1, c0], a[r0, c0]
            if abs(a[r0, c0]) >= sfmin:
                s = 1.0 / a[r0, c0]
                for k in range(1, m):
                    a[r0 + k, c0] = s * a[r0 + k, c0]
            else:
                for k in range(1, m):
                    a[r0 + k, c0] = a[r0 + k, c0] / a[r0, c0]
        else:
            info = 1
    else:
        n1 = min(m, n) // 2
        n2 = n - n1
        iinfo = _dgetrf2(a, r0, c0, m, n1, ipiv, p0)
        if info == 0 and iinfo > 0:
            info = iinfo
        # dlaswp on the right block, rows r0.. (block-relative pivots)
        for i in range(1, n1 + 1):
            ip = ipiv[p0 + i - 1]
            if ip != i:
                for j in range(c0 + n1, c0 + n):
                    a[r0 + i - 1, j], a[r0 + ip - 1, j] = a[r0 + ip - 1, j], a[r0 + i - 1, j]
        # dtrsm L, L, N, U: A12 := L11^-1 A12
        for j in range(c0 + n1, c0 + n):
            for k in range(n1):
                bkj = a[r0 + k, j]
                for i in range(k + 1, n1):
                    a[r0 + i, j] = a[r0 + i, j] - bkj * a[r0 + i, c0 + k]
        # dgemm N, N: A22 := A22 - A21 A12
        for j in range(n2):
            for l in range(n1):
                temp = -1.0 * a[r0 + l, c0 + n1 + j]
                for i in range(m - n1):
                    a[r0 + n1 + i, c0 + n1 + j] = a[r0 + n1 + i, c0 + n1 + j] + temp * a[r0 + n1 + i, c0 + l]
        iinfo = _dgetrf2(a, r0 + n1, c0 + n1, m - n1, n2, ipiv, p0 + n1)
        if info == 0 and iinfo > 0:
            info = iinfo + n1
        for i in range(n1, min(m, n)):
            ipiv[p0 + i] += n1
        # dlaswp on the left block
        for i in range(n1 + 1, min(m, n) + 1):
            ip = ipiv[p0 + i - 1]
            if ip != i:
                for j in range(c0, c0 + n1):
                    a[r0 + i - 1, j], a[r0 + ip - 1, j] = a[r0 + ip - 1, j], a[r0 + i - 1, j]
    return info


def r_solve(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """R's ``solve(a, b)`` (``b`` the identity by default)."""
    a = np.array(a, dtype=np.float64)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("a must be square")
    b = np.eye(n) if b is None else np.array(b, dtype=np.float64).reshape(n, -1)
    lu = [[float(a[i, j]) for j in range(n)] for i in range(n)]
    lu = np.array(lu, dtype=object)
    ipiv = [0] * n
    info = _dgetrf2(lu, 0, 0, n, n, ipiv, 0)
    if info > 0:
        raise np.linalg.LinAlgError(f"Lapack routine dgesv: system is exactly singular: U[{info},{info}] = 0")
    x = np.array([[float(b[i, j]) for j in range(b.shape[1])] for i in range(n)], dtype=object)
    nrhs = x.shape[1]
    for i in range(1, n + 1):  # dlaswp
        ip = ipiv[i - 1]
        if ip != i:
            for j in range(nrhs):
                x[i - 1, j], x[ip - 1, j] = x[ip - 1, j], x[i - 1, j]
    for j in range(nrhs):  # dtrsm L, L, N, U
        for k in range(n):
            bkj = x[k, j]
            for i in range(k + 1, n):
                x[i, j] = x[i, j] - bkj * lu[i, k]
    for j in range(nrhs):  # dtrsm L, U, N, N
        for k in range(n - 1, -1, -1):
            x[k, j] = x[k, j] / lu[k, k]
            bkj = x[k, j]
            for i in range(k):
                x[i, j] = x[i, j] - bkj * lu[i, k]
    return x.astype(np.float64)


def quadratic_forms(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``(x %*% a) %*% x`` for every row x of d, as R computes it (two dgemv)."""
    n, k = d.shape
    z = np.zeros((n, k))
    for j in range(k):  # dgemv T: z_j = sum_i a[i, j] x_i
        t = np.zeros(n)
        for i in range(k):
            t = t + a[i, j] * d[:, i]
        z[:, j] = 0.0 + 1.0 * t
    out = np.zeros(n)
    for j in range(k):  # dgemv N (one row): sum_j x_j z_j
        out = out + (1.0 * d[:, j]) * z[:, j]
    return out
