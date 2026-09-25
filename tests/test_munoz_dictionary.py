"""Check the Muñoz et al. orbit dictionary against literal tensor products.

Source: Tomography from collective measurements, QIP 17, 286 (2018),
https://doi.org/10.1007/s11128-018-2045-0, Eqs. (14), (17), (41)--(43).
These tests use newly written reference code, not code from the paper.
"""

from collections import defaultdict
from itertools import product
from math import comb, factorial, prod
import unittest

import numpy as np

from pauli_transforms import krawtchouk


PAULIS = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.diag([1, -1]).astype(complex),
)


def krawtchouk_binomial(h, g, m):
    """Integer coefficient of z**h in (1+z)**(m-g) (1-z)**g."""
    return sum((-1)**j * comb(g, j) * comb(m-g, h-j)
               for j in range(max(0, h-(m-g)), min(g, h)+1))


def source_counts(n, a, b, c):
    twice = (2*n-a-b-c, -a+b+c, a+b-c, a-b+c)
    if any(value < 0 or value % 2 for value in twice):
        return None
    return tuple(value // 2 for value in twice)


def source_f(n, a, c, x, y):
    """Literal binary sum in the published Eq. (42)."""
    return sum((-1)**((mu & x).bit_count()) for mu in range(1 << n)
               if mu.bit_count() == a and (mu ^ x ^ y).bit_count() == c)


def literal_orbits(n):
    """Enumerate every tensor word, independently of the coordinate library."""
    result = defaultdict(lambda: np.zeros((1 << n, 1 << n), complex))
    for word in product(range(4), repeat=n):
        counts = tuple(word.count(letter) for letter in range(4))
        matrix = np.ones((1, 1), complex)
        for letter in word:
            matrix = np.kron(matrix, PAULIS[letter])
        result[counts] += matrix
    return result


def binary_source_orbits(n):
    """Eq. (17), evaluated from its action on computational basis vectors."""
    result = defaultdict(lambda: np.zeros((1 << n, 1 << n), complex))
    for mu in range(1 << n):
        for lam in range(1 << n):
            labels = mu.bit_count(), lam.bit_count(), (mu ^ lam).bit_count()
            phase = (-1j)**((mu & lam).bit_count())
            for y in range(1 << n):
                x = y ^ lam
                result[labels][x, y] += phase * (-1)**((mu & x).bit_count())
    return result


def check_size(n):
    """Return one audit row per orbit. Every equality here is checked exactly.

    All reference calculations use integer binomial coefficients and Gaussian
    integers, stored in complex128 arrays. At the tested sizes 1..6 every
    integer is exactly representable. No tolerance or rounding is applied.
    """
    dense = literal_orbits(n)
    source = binary_source_orbits(n)
    admissible = {(a, b, c) for a, b, c in product(range(n+1), repeat=3)
                  if source_counts(n, a, b, c) is not None}
    assert set(source) == admissible
    assert len(admissible) == comb(n+3, 3)
    tables = krawtchouk.prepare(n)
    observations = []
    for labels, matrix in source.items():
        a, b, c = labels
        counts = source_counts(n, a, b, c)
        ni, nx, ny, nz = counts
        np.testing.assert_array_equal(matrix, dense[counts])
        orbit_size = factorial(n) // prod(factorial(count) for count in counts)
        assert np.vdot(matrix, matrix) == (1 << n) * orbit_size
        key = (nx+ny, nz, ny)
        production = [krawtchouk.inverse(n, {key: 1}, supplied)
                      for supplied in (None, tables)]
        formula = np.zeros_like(matrix)
        for x in range(1 << n):
            for y in range(1 << n):
                r, s, t = x.bit_count(), y.bit_count(), (x & y).bit_count()
                w = (x ^ y).bit_count()
                if w == b:
                    formula[x, y] = (1j)**ny * krawtchouk_binomial(nz, t, n-w) * krawtchouk_binomial(ny, s-t, w)
                for implementation in production:
                    assert implementation[w, t, s-t] == matrix[x, y]
        np.testing.assert_array_equal(formula, matrix)
        # One representative of every (w,t,s-t) orbit tests the binary f sum.
        checked_f = 0
        for t in range(n-b+1):
            for sy in range(b+1):
                y = ((1 << sy)-1) | (((1 << t)-1) << b)
                x = y ^ ((1 << b)-1)
                expected = (krawtchouk_binomial(nz, t, n-b)
                            * krawtchouk_binomial(ny, b-sy, b))
                assert source_f(n, a, c, x, y) == expected
                assert ((-1j)**ny * expected == matrix[x, y])
                checked_f += 1
        observations.append({
            "n": n, "source_labels": labels, "pauli_counts_IXYZ": counts,
            "matrix_entries": matrix.size, "f_orbit_representatives": checked_f,
            "source_vs_literal_max_abs": float(np.max(np.abs(matrix-dense[counts]))),
            "kernel_vs_literal_max_abs": float(np.max(np.abs(formula-dense[counts]))),
            "orbit_norm_squared": int(np.vdot(matrix, matrix).real),
            "cached_and_uncached_inverse": "exact agreement", "status": "passed",
        })
    return observations


class MunozDictionaryTests(unittest.TestCase):
    def test_all_orbits_and_matrix_entries(self):
        for n in range(1, 5):
            with self.subTest(n=n):
                check_size(n)

    def test_one_and_two_y_factors(self):
        # The integer overlap, not only its parity, determines (-i)**kY.
        one = binary_source_orbits(1)[1, 1, 0]
        two = binary_source_orbits(2)[2, 2, 0]
        np.testing.assert_array_equal(one, PAULIS[2])
        np.testing.assert_array_equal(two, np.kron(PAULIS[2], PAULIS[2]))
        self.assertEqual(one[0, 1], -1j)
        self.assertEqual(two[0, 3], -1)


if __name__ == "__main__":
    unittest.main()
