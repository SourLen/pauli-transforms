"""Small independent application regressions; the larger sweep is a separate CLI."""
import unittest
import numpy as np

from pauli_transforms.application_accuracy import dense_case, evaluate_case, reference_curve


class ApplicationAccuracyTests(unittest.TestCase):
    def test_spectra_with_multiplicity_and_independent_dense_dynamics(self):
        times = np.array([0., .3, 2., 12.])
        for n in (2, 3):
            for family in ("ising_product_x", "random_hermitian_product_state"):
                inputs, references, dense, _ = dense_case(n, family, 250926+n)
                expected = reference_curve(n, references, dense, times)
                for backend in ("hahn", "factorial"):
                    with self.subTest(n=n, family=family, backend=backend):
                        row = evaluate_case(n, backend, inputs, references, dense, times, expected)
                        self.assertTrue(row["passed"], row)
                        self.assertLess(row["full_spectrum_error"], 1e-10)
                        self.assertLess(row["max_expectation_error"], 1e-10)
                        self.assertEqual(len(row["curves"]), len(times))


if __name__ == "__main__":
    unittest.main()
