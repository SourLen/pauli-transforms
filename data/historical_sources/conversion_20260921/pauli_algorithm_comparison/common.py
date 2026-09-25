"""Shared indices and conversions. No timing or implicit precomputation."""

from math import comb

import numpy as np

OrbitKey = tuple[int, int, int]


def all_orbit_keys(n):
    """Pauli keys (w, g0, g1), also used for entry keys (w, h0, h1)."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    for w in range(n + 1):
        for g0 in range(n - w + 1):
            for g1 in range(w + 1):
                yield w, g0, g1


def pauli_type_from_key(n, key):
    """Return (nI, nX, nY, nZ)."""
    w, g0, g1 = key
    counts = n - w - g0, w - g1, g1, g0
    if min(counts) < 0:
        raise ValueError(f"invalid Pauli key {key} for n={n}")
    return counts


def block_array(n, w, data):
    """Pack a slice in [h0, h1] order; missing keys are zero."""
    return np.array(
        [[data.get((w, h0, h1), 0j) for h1 in range(w + 1)]
         for h0 in range(n - w + 1)],
        dtype=np.complex128,
    )


def matrix_qubits(matrix):
    shape = matrix.shape
    if len(shape) != 2 or shape[0] < 1 or shape[0] != shape[1]:
        raise ValueError("matrix must be nonempty and square")
    n = int(shape[0]).bit_length() - 1
    if shape[0] != 1 << n:
        raise ValueError("matrix dimension must be a power of two")
    return n


def entries_to_dense(n, data):
    """Expand literal entry values data[w,h0,h1] into a 2**n square matrix."""
    matrix = np.empty((1 << n, 1 << n), dtype=np.complex128)
    for row in range(1 << n):
        for column in range(1 << n):
            flip = row ^ column
            key = flip.bit_count(), (row & column).bit_count(), (flip & column).bit_count()
            matrix[row, column] = data.get(key, 0j)
    return matrix


def dense_to_entries(matrix):
    """Read orbit representatives; the caller supplies a permutation-invariant matrix."""
    n = matrix_qubits(matrix)
    entries = {}
    for w, h0, h1 in all_orbit_keys(n):
        flip = (1 << w) - 1
        column = ((1 << h1) - 1) | (((1 << h0) - 1) << w)
        entries[w, h0, h1] = complex(matrix[flip ^ column, column])
    return entries


def compress_pauli(n, coefficients):
    """Read Pauli-orbit representatives from a dense array or sparse (x,z) dict."""
    result = {}
    for key in all_orbit_keys(n):
        w, g0, g1 = key
        x = (1 << w) - 1
        z = ((1 << g1) - 1) | (((1 << g0) - 1) << w)
        value = coefficients.get((x, z), 0j) if isinstance(coefficients, dict) else coefficients[x, z]
        result[key] = complex(value)
    return result


def expand_pauli(n, coefficients):
    """Yield (x,z,value) for every distinct word in each nonzero orbit."""
    def words(counts, bit, x, z):
        if bit == n:
            yield x, z
            return
        for letter, count in enumerate(counts):
            if count:
                remaining = list(counts)
                remaining[letter] -= 1
                yield from words(
                    remaining, bit + 1,
                    x | ((1 << bit) if letter in (1, 2) else 0),
                    z | ((1 << bit) if letter in (2, 3) else 0),
                )

    for key, value in coefficients.items():
        if value:
            for x, z in words(pauli_type_from_key(n, key), 0, 0, 0):
                yield x, z, complex(value)


def block_shapes(n):
    return tuple(n - 2 * k + 1 for k in range(n // 2 + 1))


def zero_schur_blocks(n):
    return [np.zeros((side, side), dtype=np.complex128) for side in block_shapes(n)]


def t_values(n, r, s):
    return tuple(range(max(0, r + s - n), min(r, s) + 1))


def kappa(n, r, s):
    return min(r, s, n - r, n - s)


def specht_multiplicities(n):
    return tuple(comb(n, k) - (comb(n, k - 1) if k else 0) for k in range(n // 2 + 1))


def orbit_size(n, r, s, t):
    return comb(n, t) * comb(n - t, r - t) * comb(n - r, s - t)
