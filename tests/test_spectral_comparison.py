"""Small independent checks of the complete-eigensystem benchmark contract."""

import numpy as np
import unittest

from pauli_transforms import benchmark_cases, dense_reference, transforms
from pauli_transforms.run_spectral_comparison import eigensystem_checks, reference_checks, solve


class SpectralComparisonTests(unittest.TestCase):
    def test_all_eigenpairs_against_full_pauli_matrix(self):
        for n in [2, 3, 4]:
            with self.subTest(n=n):
                coefficients = benchmark_cases.make_case(n, "general", 1234 + n, 0)
                dense = dense_reference.dense_operator(n, coefficients)
                bases = dense_reference.prepare(n)
                reference = [basis @ dense @ basis.T for basis in bases]
                blocks = transforms.pauli_to_schur(n, coefficients)
                systems = solve(blocks)
                metrics = eigensystem_checks(blocks, systems)
                metrics.update(reference_checks(n, coefficients, blocks, systems, reference, dense, bases))
                self.assertTrue(all(value < 1e-11 for value in metrics.values() if value is not None))


    def test_degenerate_eigenspaces_are_valid_without_vector_matching(self):
        blocks = [np.eye(3), np.zeros((1, 1))]
        systems = solve(blocks)
        self.assertEqual(max(eigensystem_checks(blocks, systems).values()), 0)


    def test_incomplete_or_incorrect_eigenbasis_fails_checks(self):
        block = np.diag([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "complete"):
            eigensystem_checks([block], [(np.array([1.0]), np.array([[1.0], [0.0]]))])
        metrics = eigensystem_checks([block], [(np.array([1.0, 2.0]), np.ones((2, 2)))])
        self.assertGreater(metrics["residual"], 0.1)
        self.assertGreater(metrics["orthogonality"], 0.1)


if __name__ == "__main__":
    unittest.main()
