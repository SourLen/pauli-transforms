"""Independent checks for the two newly introduced benchmark baselines."""

import unittest
from itertools import product

import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from importlib.util import find_spec

from pauli_transforms import run_application_baselines as bench


def literal_matrix(n, coefficients):
    """Explicit tensor products, independent of bit-mask and Walsh code."""
    paulis = (np.eye(2), np.array([[0, 1], [1, 0]]),
              np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))
    result = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for labels in product(range(4), repeat=n):
        nx, ny, nz = (labels.count(label) for label in (1, 2, 3))
        value = coefficients.get((nx + ny, nz, ny), 0)
        matrix = np.array([[value]], dtype=complex)
        for label in labels:
            matrix = np.kron(matrix, paulis[label])
        result += matrix
    return result


class BaselineChecks(unittest.TestCase):
    def test_dense_assembly_complex_coefficients(self):
        rng = np.random.default_rng(731)
        for n in (1, 2, 3, 4):
            coefficients = {key: value + 1j * rng.normal() for key, value in
                            bench.benchmark_cases.make_case(n, "general", 827 + n, locality=0).items()}
            np.testing.assert_allclose(bench.dense_hamiltonian(n, coefficients),
                                       literal_matrix(n, coefficients), atol=2e-13, rtol=2e-13)

    @unittest.skipUnless(find_spec("qutip"), "optional qutip dependency not installed")
    def test_piqs_blocks_all_sectors(self):
        for n in (2, 3, 5, 8):
            for p in (0., .6, -.4):
                blocks = bench.piqs_ising_blocks(n, g=.9, h=.3, p=p)
                references = (bench.pm.ising_collective_blocks(n, .9, .3),
                              bench.pm.product_x_collective_blocks(n, p),
                              bench.pm.collective_observable_blocks(n, "x"))
                for actual, expected in zip(blocks, references, strict=True):
                    for a, b in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-13)
                self.assertAlmostEqual(bench.pm.block_trace(n, blocks[1]).real, 1.)

    @unittest.skipUnless(find_spec("qutip"), "optional qutip dependency not installed")
    def test_piqs_curve_against_full_matrix_exponential(self):
        times = np.array([0., .13, .8, 2.7, 12.])
        for n in (2, 3, 5):
            blocks = bench.piqs_ising_blocks(n)
            values = bench.pm.expectation_time_series(n, bench.pm.diagonalize_blocks(blocks[0]),
                                                       blocks[1], blocks[2], times)
            hamiltonian = literal_matrix(n, bench.pm.ising_pauli(n))
            state = literal_matrix(n, bench.pm.product_x_pauli(n, .6))
            observable = literal_matrix(n, {(1, 0, 0): 1 / n})
            expected = []
            for t in times:
                unitary = expm(-1j * t * hamiltonian)
                expected.append(np.trace(observable @ unitary @ state @ unitary.conj().T))
            np.testing.assert_allclose(values, expected, atol=3e-13, rtol=3e-13)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
