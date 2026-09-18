"""Independent checks of the physical examples and shared conversion setup."""

import unittest
from unittest.mock import patch

import numpy as np

from pauli_transforms import application_conversions as conversions
from pauli_transforms import anschuetz_optimized, ising_example, physical_models as pm
from pauli_transforms import dense_reference, transforms
from pauli_transforms.common import specht_multiplicities
from pauli_transforms.run_random_dynamics_comparison import timed_sample


class PhysicalExamplesTests(unittest.TestCase):
    def test_ising_full_spectrum_includes_all_sector_multiplicities(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                blocks = transforms.pauli_to_schur(n, pm.ising_pauli(n, .83, .47))
                for actual, expected in zip(blocks, pm.ising_collective_blocks(n, .83, .47), strict=True):
                    np.testing.assert_allclose(actual, expected, atol=2e-13)
                spectrum = pm.diagonalize_blocks(blocks)
                expanded = np.sort(np.concatenate([np.repeat(values, mu) for values, mu in
                    zip(spectrum.energies, specht_multiplicities(n), strict=True)]))
                dense = dense_reference.dense_operator(n, pm.ising_pauli(n, .83, .47))
                np.testing.assert_allclose(expanded, np.linalg.eigvalsh(dense), atol=2e-13)

    def test_collective_product_state_blocks_agree_with_full_tensor_product(self):
        for n in (2, 3, 5):
            for p in (-1., -.4, 0., .6, 1.):
                with self.subTest(n=n, p=p):
                    one_site = np.array([[1., p], [p, 1.]]) / 2
                    dense = np.ones((1, 1))
                    for _ in range(n):
                        dense = np.kron(dense, one_site)
                    blocks = pm.product_x_collective_blocks(n, p)
                    for actual, basis in zip(blocks, dense_reference.prepare(n), strict=True):
                        np.testing.assert_allclose(actual, basis @ dense @ basis.T, atol=2e-14)
                    self.assertAlmostEqual(pm.block_trace(n, blocks).real, 1.)

    def test_ising_curve_matches_independent_collective_spin_calculation(self):
        for n in (3, 8):
            _, actual, expected, checks = ising_example.compute(n, np.array([0., .5, 1.7]))
            np.testing.assert_allclose(actual, expected, atol=2e-13)
            self.assertAlmostEqual(actual[0].real, .6)
            self.assertLess(max(checks.values()), 1e-12)

    def test_nonhermitian_hamiltonian_is_rejected(self):
        with self.assertRaisesRegex(ArithmeticError, "not Hermitian"):
            pm.diagonalize_blocks([np.array([[1., 2.], [0., 3.]])])


class SharedConversionTests(unittest.TestCase):
    def test_each_dynamics_trial_builds_one_fresh_bank_for_all_three_inputs(self):
        n = 4
        inputs = (pm.ising_pauli(n), pm.product_x_pauli(n, .6), {(1, 0, 0): 1 / n})
        keys = set().union(*(set(mapping) for mapping in inputs))
        tables = []
        original = anschuetz_optimized.prepare

        def tracked(n, keys, **kwargs):
            self.assertEqual(set(keys), expected_keys)
            table = original(n, keys=keys, **kwargs)
            tables.append(table)
            return table

        expected_keys = keys
        with patch.object(anschuetz_optimized, "prepare", side_effect=tracked) as prepare:
            for _ in range(2):
                timed_sample(n, "anschuetz_optimized", inputs, [0., .2], {})
        self.assertEqual(prepare.call_count, 2)
        self.assertIsNot(tables[0], tables[1])

    def test_public_method_requires_explicit_loaded_source(self):
        with self.assertRaisesRegex(ValueError, "load the pinned"):
            conversions.prepare(2, conversions.PUBLIC_METHOD, ({},), {})


if __name__ == "__main__":
    unittest.main()
