"""Separated Krawtchouk decomposition, with explicit optional table reuse."""

from math import comb

import numpy as np

from .common import block_array


def exact_column(m, h):
    """Exact integer values K_h(g;m), g=0,...,m."""
    if not 0 <= h <= m:
        raise ValueError("degree must lie in 0..m")
    values = [comb(m, h)]
    if m:
        values.append(comb(m - 1, h) - (comb(m - 1, h - 1) if h else 0))
    for g in range(1, m):
        values.append(((m - 2 * h) * values[g] - g * values[g - 1]) // (m - g))
    return tuple(values)


def build_matrix(m):
    """Fresh Q[g,h] = K_h(g;m), constructed by the degree recurrence."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    q = np.ones((m + 1, m + 1), dtype=np.float64)
    if m:
        for g in range(m + 1):
            q[g, 1] = m - 2 * g
            for h in range(1, m):
                q[g, h + 1] = ((m - 2 * g) * q[g, h] - (m - h + 1) * q[g, h - 1]) / (h + 1)
    return q


def prepare(n):
    """Build Q_0,...,Q_n afresh; retain the returned tuple for repeated calls."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    return tuple(build_matrix(m) for m in range(n + 1))


def decompose_slice(n, w, entry_block, q1=None, q0=None):
    if q1 is None:
        q1 = build_matrix(w)
    if q0 is None:
        q0 = build_matrix(n - w)
    after_h1 = (entry_block @ q1.T) * (2.0 ** (-w))
    block = (q0 @ after_h1) * (2.0 ** (-(n - w)))
    block *= np.array([1j ** (-g1) for g1 in range(w + 1)])[None, :]
    return block


def decompose(n, data, matrices=None):
    """Entry orbits -> Pauli orbits. Without matrices, setup is included."""
    if matrices is None:
        matrices = prepare(n)
    if len(matrices) != n + 1:
        raise ValueError("matrix bank must contain Q_0 through Q_n")
    output = {}
    for w in range(n + 1):
        block = decompose_slice(n, w, block_array(n, w, data), matrices[w], matrices[n - w])
        for g0 in range(n - w + 1):
            for g1 in range(w + 1):
                output[w, g0, g1] = complex(block[g0, g1])
    return output


def inverse(n, coefficients, matrices=None):
    """Pauli orbits -> literal entry orbits, with missing coefficients zero.

    Without a supplied bank, sparse slices use exact integer columns as in
    the original implementation. A supplied bank selects dense products.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if matrices is not None and len(matrices) != n + 1:
        raise ValueError("matrix bank must contain Q_0 through Q_n")
    support = {w: [] for w in range(n + 1)}
    for (w, g0, g1), value in coefficients.items():
        if not (0 <= w <= n and 0 <= g0 <= n - w and 0 <= g1 <= w):
            raise ValueError("invalid Pauli orbit key")
        if value:
            support[w].append((g0, g1, complex(value)))
    output = {}
    for w in range(n + 1):
        if matrices is None and len(support[w]) <= n + 2:
            block = np.zeros((n - w + 1, w + 1), dtype=np.complex128)
            for g0, g1, value in support[w]:
                outside = np.asarray(exact_column(n - w, g0), dtype=np.float64)
                inside = np.asarray(exact_column(w, g1), dtype=np.float64)
                block += (1j ** g1) * value * np.outer(outside, inside)
        else:
            q0 = matrices[n - w] if matrices is not None else build_matrix(n - w)
            q1 = matrices[w] if matrices is not None else build_matrix(w)
            phase = np.array([1j ** g1 for g1 in range(w + 1)])
            block = q0 @ (block_array(n, w, coefficients) * phase[None, :]) @ q1.T
        for h0 in range(n - w + 1):
            for h1 in range(w + 1):
                output[w, h0, h1] = complex(block[h0, h1])
    return output
