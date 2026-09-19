"""Matrix-unit conventions and replay of the native permqit comparison."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pauli_transforms import dense_reference, permqit_adapter, schur_factorial, schur_hahn
from pauli_transforms.common import specht_multiplicities
from pauli_transforms.permqit_comparison_reference import (
    block_errors, dense_matrix, integer_kernel_blocks, keys, random_entries,
)
from pauli_transforms.plot import read_matrix_unit_campaign


DATA = Path(__file__).resolve().parents[1] / "data/thesis/matrix_units"


def dense_blocks(n, coefficients):
    matrix = dense_matrix(n, coefficients)
    return [basis.conj() @ matrix @ basis.T for basis in dense_reference.prepare(n)]


class MatrixUnitComparisonTests(unittest.TestCase):
    def test_reference_and_local_backends_against_dense_projection(self):
        for n in (2, 3, 6):
            for hermitian in (False, True):
                with self.subTest(n=n, hermitian=hermitian):
                    coefficients = random_entries(n, 92000+n, hermitian=hermitian)
                    expected = dense_blocks(n, coefficients)
                    self.assertTrue(block_errors(n, integer_kernel_blocks(n, coefficients), expected)["passed"])
                    for method in (schur_factorial, schur_hahn):
                        actual = method.orbit_to_schur(n, coefficients, method.prepare(n))
                        self.assertTrue(block_errors(n, actual, expected)["passed"])

    def test_replay_worker_protocol_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix_units"
            command = [sys.executable, "-m", "pauli_transforms.run_permqit_comparison",
                       "--output", str(path), "--sizes", "2", "--trials", "1",
                       "--app-repeats", "2", "--methods", "factorial_float64", "hahn_float64",
                       "--input-directory", str(DATA / "inputs")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cfg, rows = read_matrix_unit_campaign(path)
            self.assertEqual(cfg["app_repeats"], 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual((path / "inputs/n2_trial0.json").read_bytes(),
                             (DATA / "inputs/n2_trial0.json").read_bytes())
            before = (path / "measurements.jsonl").read_bytes()
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output directory must be empty", result.stderr)
            self.assertEqual((path / "measurements.jsonl").read_bytes(), before)

    def test_worker_failures_are_retained_and_fail_the_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix_units"
            # Too short even to import the worker: exercise timeout bookkeeping.
            result = subprocess.run([
                sys.executable, "-m", "pauli_transforms.run_permqit_comparison",
                "--output", str(path), "--sizes", "2", "--trials", "1",
                "--methods", "hahn_float64", "--timeout", "0.000001",
            ], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            rows = [json.loads(line) for line in (path / "measurements.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["status"], "timeout")
            with self.assertRaises(ValueError):
                read_matrix_unit_campaign(path)


@unittest.skipUnless(importlib.util.find_spec("permqit"), "optional pinned permqit installation")
class NativePermqitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PERMQIT_USE_GPU"] = "false"

    def test_every_matrix_unit_orbit(self):
        for n in range(1, 6):
            prepared = permqit_adapter.prepare(n)
            for key in keys(n):
                with self.subTest(n=n, key=key):
                    coefficients = {key: 1j}
                    expected = dense_blocks(n, coefficients)
                    actual = permqit_adapter.orbit_to_schur(n, coefficients, prepared)
                    reference = integer_kernel_blocks(n, coefficients)
                    for a, b, c in zip(actual, expected, reference):
                        np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
                        np.testing.assert_allclose(c, b, atol=2e-13, rtol=2e-13)

    def test_dense_projection_norm_and_spectrum(self):
        for n in (6, 7, 8):
            for hermitian in (False, True):
                with self.subTest(n=n, hermitian=hermitian):
                    coefficients = random_entries(n, 92000+n, hermitian=hermitian)
                    matrix = dense_matrix(n, coefficients)
                    expected = dense_blocks(n, coefficients)
                    blocks = permqit_adapter.orbit_to_schur(n, coefficients)
                    self.assertTrue(block_errors(n, blocks, expected)["passed"])
                    self.assertTrue(block_errors(n, integer_kernel_blocks(n, coefficients), expected)["passed"])
                    mu = specht_multiplicities(n)
                    norm = sum(m*np.linalg.norm(a)**2 for m, a in zip(mu, blocks))
                    np.testing.assert_allclose(norm, np.linalg.norm(matrix)**2, atol=1e-12)
                    if hermitian:
                        for block in blocks:
                            np.testing.assert_allclose(block, block.conj().T, atol=1e-13)
                        spectrum = np.concatenate([np.repeat(np.linalg.eigvalsh(a), m)
                                                   for m, a in zip(mu, blocks)])
                        np.testing.assert_allclose(np.sort(spectrum), np.linalg.eigvalsh(matrix), atol=1e-12)

    def test_identity_and_multiplication(self):
        n = 4
        prepared = permqit_adapter.prepare(n)
        identity = {(r, r, r): 1 for r in range(n+1)}
        for block in permqit_adapter.orbit_to_schur(n, identity, prepared):
            np.testing.assert_allclose(block, np.eye(len(block)), atol=1e-13)
        a, b = (random_entries(n, seed) for seed in (1, 2))
        product = dense_matrix(n, a) @ dense_matrix(n, b)
        actual = [x @ y for x, y in zip(permqit_adapter.orbit_to_schur(n, a, prepared),
                                      permqit_adapter.orbit_to_schur(n, b, prepared))]
        expected = [u.conj() @ product @ u.T for u in dense_reference.prepare(n)]
        self.assertTrue(block_errors(n, actual, expected)["passed"])

    def test_invalid_coordinates_and_mismatched_preparation(self):
        prepared = permqit_adapter.prepare(2)
        for key, value in (((0, 0, 1), 1), ((True, 0, 0), 1), ((1., 0, 0), 1),
                           ((0, 0, 0), np.nan)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                permqit_adapter.orbit_to_schur(2, {key: value}, prepared)
        with self.assertRaises(ValueError):
            permqit_adapter.orbit_to_schur(3, {}, prepared)


if __name__ == "__main__":
    unittest.main()
