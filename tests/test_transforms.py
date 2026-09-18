"""Independent dense checks of phases, orbit weights, and both directions."""

from itertools import product
from math import comb
import unittest

import numpy as np

from pauli_transforms import entries_to_pauli, pauli_to_entries
from pauli_transforms import prepare, pauli_to_schur, schur_to_pauli
from pauli_transforms import common, krawtchouk, schur_factorial, schur_hahn, dense_reference


PAULIS = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.diag([1, -1]).astype(complex),
)


def tensor(factors):
    result = np.ones((1, 1), dtype=complex)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def explicit_pauli_matrix(n, coefficients):
    """Enumerate tensor words, without using the library's bit-mask expansion."""
    matrix = np.zeros((2**n, 2**n), dtype=complex)
    for word in product(range(4), repeat=n):
        key = (word.count(1) + word.count(2), word.count(3), word.count(2))
        coefficient = coefficients.get(key, 0)
        if coefficient:
            matrix += coefficient * tensor(PAULIS[letter] for letter in word)
    return matrix


def explicit_schur_blocks(n, matrix):
    """Project onto tensor products of k singlets and normalized Dicke states."""
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    blocks = []
    for k in range(n // 2 + 1):
        remaining = n - 2*k
        paired = np.ones(1, dtype=complex)
        for _ in range(k):
            paired = np.kron(paired, singlet)
        basis = []
        for weight in range(remaining + 1):
            dicke = np.array([int(index.bit_count() == weight)
                              for index in range(2**remaining)], dtype=complex)
            dicke /= np.sqrt(comb(remaining, weight))
            basis.append(np.kron(paired, dicke))
        basis = np.array(basis)
        blocks.append(basis.conj() @ matrix @ basis.T)
    return blocks


class TransformTests(unittest.TestCase):
    def test_krawtchouk_entries_and_independent_pauli_traces(self):
        rng = np.random.default_rng(1307)
        for n in range(6):
            coefficients = {key: rng.normal() + 1j*rng.normal()
                            for key in common.all_orbit_keys(n)}
            dense = explicit_pauli_matrix(n, coefficients)
            entries = common.dense_to_entries(dense)
            np.testing.assert_allclose(common.entries_to_dense(n, entries), dense)
            for tables in (None, krawtchouk.prepare(n)):
                recovered = entries_to_pauli(n, entries, tables)
                reconstructed = pauli_to_entries(n, coefficients, tables)
                for key, value in coefficients.items():
                    w, g0, g1 = key
                    word = [2]*g1 + [1]*(w-g1) + [3]*g0 + [0]*(n-w-g0)
                    trace = np.vdot(tensor(PAULIS[letter] for letter in word), dense) / 2**n
                    self.assertAlmostEqual(recovered[key], trace)
                    self.assertAlmostEqual(recovered[key], value)
                np.testing.assert_allclose(common.entries_to_dense(n, reconstructed), dense,
                                           atol=2e-12, rtol=2e-12)

    def test_both_schur_backends_against_independent_projection(self):
        rng = np.random.default_rng(1409)
        for n in range(6):
            coefficients = {key: rng.normal() + 1j*rng.normal()
                            for key in common.all_orbit_keys(n)}
            dense = explicit_pauli_matrix(n, coefficients)
            expected = explicit_schur_blocks(n, dense)
            for backend in ("hahn", "factorial"):
                tables = prepare(n, backend=backend)
                for supplied in (None, tables):
                    actual = pauli_to_schur(n, coefficients, supplied, backend=backend)
                    for block, reference in zip(actual, expected):
                        np.testing.assert_allclose(block, reference, atol=3e-11, rtol=3e-11)
                    recovered = schur_to_pauli(n, expected, supplied, backend=backend)
                    np.testing.assert_allclose(list(recovered.values()), list(coefficients.values()),
                                               atol=3e-11, rtol=3e-11)
            reference_blocks = dense_reference.pauli_to_schur(n, coefficients)
            for actual, expected_block in zip(reference_blocks, expected):
                np.testing.assert_allclose(actual, expected_block, atol=3e-11, rtol=3e-11)

    def test_schur_inverse_on_arbitrary_complex_blocks(self):
        rng = np.random.default_rng(1709)
        for n in range(7):
            blocks = [rng.normal(size=(side, side)) + 1j*rng.normal(size=(side, side))
                      for side in common.block_shapes(n)]
            for backend in (schur_hahn, schur_factorial):
                tables = backend.prepare(n)
                entries = backend.schur_to_orbit(n, blocks, tables)
                reconstructed = backend.orbit_to_schur(n, entries, tables)
                for actual, expected in zip(reconstructed, blocks):
                    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)

    def test_identity_y_phase_and_multiplicity_weighted_norm(self):
        for n in range(1, 7):
            tables = prepare(n)
            identity = pauli_to_schur(n, {(0, 0, 0): 1}, tables)
            for block in identity:
                np.testing.assert_allclose(block, np.eye(len(block)), atol=2e-12)
            self.assertEqual(sum(mu*side for mu, side in
                                 zip(common.specht_multiplicities(n), common.block_shapes(n))), 2**n)
            collective_y = pauli_to_schur(n, {(1, 0, 1): 1}, tables)
            for k, block in enumerate(collective_y):
                N = n-2*k
                raising = np.diag(np.sqrt((np.arange(N)+1)*(N-np.arange(N))), 1)
                np.testing.assert_allclose(block, -1j*(raising-raising.T), atol=2e-12)
            actual_norm = sum(mu*np.vdot(block, block).real for mu, block in
                              zip(common.specht_multiplicities(n), collective_y))
            self.assertAlmostEqual(actual_norm, n*2**n)

    def test_factorial_precision_and_backend_reuse(self):
        coefficients = {(1, 0, 1): .3j, (0, 0, 0): 1}
        tables = prepare(5, backend="factorial", dtype=np.longdouble)
        actual = pauli_to_schur(5, coefficients, tables)
        reference = explicit_schur_blocks(5, explicit_pauli_matrix(5, coefficients))
        for block, expected in zip(actual, reference):
            np.testing.assert_allclose(block, expected, atol=2e-12)
        with self.assertRaises(ValueError):
            pauli_to_schur(5, coefficients, tables, backend="hahn")
        with self.assertRaises(ValueError):
            pauli_to_schur(5, coefficients, tables, dtype=np.longdouble)


if __name__ == "__main__":
    unittest.main()
