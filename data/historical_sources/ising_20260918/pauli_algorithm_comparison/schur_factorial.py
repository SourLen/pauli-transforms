"""Chapter 5's shared-factor matrix-unit/Schur conversion.

For each row weight r, apply G, divide by (r-u)!(s-u)!, and apply F.
The inverse uses the transposed factors and the Hilbert--Schmidt weights.
No Hahn table or polynomial recurrence is used. Ordinary matrix multiplication
gives O(n**4) arithmetic, with O(n**3) coordinate storage and O(n**2) workspace.
The separable normalizer gamma[k,r,s] = q[k,r] q[k,s] avoids storing a cubic
normalization table. Factor preparation costs at most O(n**3) arithmetic.

These are arithmetic counts, not floating-point stability guarantees. The
alternating binomial factor can cause severe cancellation. Extended precision
is available explicitly through dtype; returned coordinates are complex128.
"""

from dataclasses import dataclass
import operator

import numpy as np

from .common import block_shapes


@dataclass(frozen=True)
class FactorTables:
    n: int
    F: np.ndarray
    G: np.ndarray
    factorials: np.ndarray
    inverse_factorials: np.ndarray
    q: np.ndarray
    multiplicities: np.ndarray


def _qubits(n):
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("n must be a nonnegative integer")
    try:
        n = operator.index(n)
    except TypeError as exc:
        raise ValueError("n must be a nonnegative integer") from exc
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    return n


def prepare(n, *, dtype=np.float64):
    """Build shared factors afresh, without constructing any (r,s) kernel bank."""
    n = _qubits(n)
    dtype = np.dtype(dtype)
    if dtype.kind != "f" or dtype.itemsize < np.dtype(np.float64).itemsize:
        raise TypeError("dtype must be float64 or a wider real floating dtype")
    side = n + 1
    factorials = np.ones(side, dtype=dtype)
    pascal = np.zeros((side, side), dtype=dtype)
    pascal[:, 0] = 1
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for j in range(1, side):
                factorials[j] = factorials[j-1] * j
                pascal[j, 1:j+1] = pascal[j-1, :j] + pascal[j-1, 1:j+1]
            inverse_factorials = 1 / factorials
            G = pascal.copy()
            for u in range(side):
                G[u, :u+1] *= np.where((u - np.arange(u+1)) % 2, -1, 1)
            F = np.zeros((side, side), dtype=dtype)
            q = np.zeros((n//2 + 1, side), dtype=dtype)
            for k in range(n//2 + 1):
                weights = np.arange(k, n-k+1)
                F[k, weights] = factorials[n-k-weights] / factorials[weights-k]
                q[k, weights] = np.sqrt(factorials[weights-k] / factorials[n-k-weights])
            multiplicities = pascal[n, :n//2+1].copy()
            multiplicities[1:] -= pascal[n, :n//2]
    except FloatingPointError as exc:
        raise ArithmeticError("factorial factors overflowed; use a wider dtype or higher precision") from exc
    arrays = (F, G, factorials, inverse_factorials, q, multiplicities)
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ArithmeticError("non-finite factorial factors")
    for array in arrays:
        array.setflags(write=False)
    return FactorTables(n, *arrays)


def _tables(n, factors):
    n = _qubits(n)
    if factors is None:
        factors = prepare(n)
    if not isinstance(factors, FactorTables) or factors.n != n:
        raise ValueError("factor tables must be prepared for the same n")
    return n, factors


def _complex_dtype(factors):
    return np.result_type(factors.F.dtype, np.complex128)


def _coordinates(n, coefficients, dtype):
    """Validate sparse input and group entries by r in O(n**3) storage."""
    grouped = [[] for _ in range(n+1)]
    for key, value in coefficients.items():
        try:
            if len(key) != 3 or any(isinstance(i, (bool, np.bool_)) for i in key):
                raise ValueError
            r, s, t = (operator.index(i) for i in key)
            if not (0 <= r <= n and 0 <= s <= n and max(0, r+s-n) <= t <= min(r, s)):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid matrix-unit key {key!r} for n={n}") from exc
        scalar = np.asarray(value, dtype=dtype)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError("matrix-unit coefficients must be finite scalars")
        grouped[r].append((s, t, scalar.item()))
    return grouped


def _rescale(matrix, r, factors):
    """Divide the (u,s) entries by factorials, zeroing forbidden u > r,s."""
    n = factors.n
    matrix[r+1:, :] = 0
    inv = factors.inverse_factorials
    for u in range(r+1):
        matrix[u, :u] = 0
        matrix[u, u:] *= inv[r-u] * inv[:n-u+1]


def _as_complex128(array):
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.asarray(array, dtype=np.complex128)
        except FloatingPointError as exc:
            raise ArithmeticError("factorial conversion output exceeds complex128 range") from exc
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("non-finite factorial conversion output")
    return result


def orbit_to_schur(n, coefficients, factors=None):
    """Convert literal common entries keyed by (r,s,t) to representative blocks."""
    n, factors = _tables(n, factors)
    dtype = _complex_dtype(factors)
    inputs = _coordinates(n, coefficients, dtype)
    blocks = [np.zeros((d, d), dtype=np.complex128) for d in block_shapes(n)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for r in range(n+1):
            X = np.zeros((n+1, n+1), dtype=dtype)
            for s, t, value in inputs[r]:
                X[t, s] = value
            intermediate = factors.G @ X
            _rescale(intermediate, r, factors)
            transformed = factors.F @ intermediate
            for k in range(min(r, n-r)+1):
                values = transformed[k, k:n-k+1] * factors.q[k, r] * factors.q[k, k:n-k+1]
                blocks[k][r-k, :] = _as_complex128(values)
    return blocks


def schur_to_orbit(n, blocks, factors=None):
    """Weighted inverse using F.T and G.T; accepts arbitrary complex blocks."""
    n, factors = _tables(n, factors)
    dtype = _complex_dtype(factors)
    shapes = block_shapes(n)
    if len(blocks) != len(shapes):
        raise ValueError("wrong number of Schur blocks")
    checked = []
    for block, side in zip(blocks, shapes):
        block = np.asarray(block, dtype=dtype)
        if block.shape != (side, side) or not np.all(np.isfinite(block)):
            raise ValueError("Schur blocks must have the expected shapes and finite entries")
        checked.append(block)
    output = {}
    fact = factors.factorials
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for r in range(n+1):
            X = np.zeros((n+1, n+1), dtype=dtype)
            for k in range(min(r, n-r)+1):
                X[k, k:n-k+1] = (checked[k][r-k, :] * factors.multiplicities[k]
                                 * factors.q[k, r] * factors.q[k, k:n-k+1])
            intermediate = factors.F.T @ X
            _rescale(intermediate, r, factors)
            transformed = factors.G.T @ intermediate
            for s in range(n+1):
                ts = np.arange(max(0, r+s-n), min(r, s)+1)
                inverse_sizes = fact[ts] * fact[r-ts] * fact[s-ts] * fact[n-r-s+ts] / fact[n]
                values = _as_complex128(transformed[ts, s] * inverse_sizes)
                for t, value in zip(ts, values):
                    output[r, s, int(t)] = complex(value)
    return output
