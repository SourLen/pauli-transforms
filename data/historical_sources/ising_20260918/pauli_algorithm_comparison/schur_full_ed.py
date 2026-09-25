"""Dense full-matrix baseline projected onto singlet/Dicke Schur bases."""

from math import factorial, sqrt

import numpy as np

from .common import expand_pauli


def singlet_dicke_basis(n, k):
    """Rows are |u[k,r]>, r=k,...,n-k, in the original block convention."""
    size = 1 << n
    state = np.zeros(size)
    for pattern in range(1 << k):
        index, sign = 0, 1
        for pair in range(k):
            if (pattern >> pair) & 1:
                index |= 1 << (2 * pair)
                sign = -sign
            else:
                index |= 1 << (2 * pair + 1)
        state[index] = sign * 2.0 ** (-0.5 * k)
    vectors = np.empty((n - 2 * k + 1, size))
    indices = np.arange(size)
    for r in range(k, n - k + 1):
        if r > k:
            lowered = np.zeros_like(state)
            for qubit in range(n):
                source = indices[(indices & (1 << qubit)) == 0]
                lowered[source | (1 << qubit)] += state[source]
            state = lowered
        scale = sqrt(factorial(n - k - r) / (factorial(r - k) * factorial(n - 2 * k)))
        vectors[r - k] = scale * state
    return vectors


def prepare(n):
    """Build all basis rows afresh; dense storage grows exponentially with n."""
    return tuple(singlet_dicke_basis(n, k) for k in range(n // 2 + 1))


def dense_operator(n, coefficients):
    """Expand Pauli orbits and sum their explicit dense matrices."""
    size = 1 << n
    columns = np.arange(size, dtype=np.uint64)
    matrix = np.zeros((size, size), dtype=np.complex128)
    for x, z, value in expand_pauli(n, coefficients):
        parity = np.bitwise_count(columns & np.uint64(z)) & 1
        signs = 1.0 - 2.0 * parity.astype(float)
        matrix[columns ^ np.uint64(x), columns] += value * (1j ** ((x & z).bit_count())) * signs
    return matrix


def pauli_to_schur(n, coefficients, bases=None):
    """Dense expansion and projection; optionally reuse prepare(n) output."""
    if bases is None:
        bases = prepare(n)
    matrix = dense_operator(n, coefficients)
    return [basis.conj() @ matrix @ basis.T for basis in bases]
