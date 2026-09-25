"""Deterministic independent regressions; expensive sweep is an explicit command."""
import unittest

from pauli_transforms.accuracy import dense_campaign, kernel_campaign


class AccuracyTests(unittest.TestCase):
    def test_all_six_directions_and_weighted_norms(self):
        observations = dense_campaign(3)
        failures = [r for r in observations if not r["passed"]]
        self.assertEqual(failures, [])

    def test_kernel_boundaries_signs_zeros_and_admissible_endpoint(self):
        observations = kernel_campaign([0, 1, 2, 3, 6])
        self.assertTrue(any(r["maximum_zero_entry_error"] is not None for r in observations))
        self.assertEqual([r for r in observations if not r["passed"]], [])


if __name__ == "__main__":
    unittest.main()
