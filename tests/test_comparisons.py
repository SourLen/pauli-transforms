"""Compare the published-formula implementations with explicit small matrices."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pauli_transforms import anschuetz_optimized, anschuetz_public, chang, common
from test_transforms import explicit_pauli_matrix, explicit_schur_blocks


class ComparisonTests(unittest.TestCase):
    def test_every_small_orbit_against_independent_projection(self):
        for n in range(5):
            for key in common.all_orbit_keys(n):
                with self.subTest(n=n, key=key):
                    coefficients = {key: .5-.25j}
                    expected = explicit_schur_blocks(n, explicit_pauli_matrix(n, coefficients))
                    for backend in (anschuetz_optimized, chang):
                        actual = backend.pauli_to_schur(n, coefficients)
                        for block, reference in zip(actual, expected):
                            if hasattr(block, "toarray"):
                                block = block.toarray()
                            np.testing.assert_allclose(block, reference, atol=1e-11, rtol=1e-11)

    def test_reused_comparison_tables_and_sparse_chang_output(self):
        n = 9
        coefficients = {(2, 2, 1): 1e-6, (1, 1, 1): -.002, (0, 0, 0): .25}
        for backend in (anschuetz_optimized, chang):
            tables = (backend.prepare(n, coefficients) if backend is anschuetz_optimized
                      else backend.prepare(n, 4))
            cached = backend.pauli_to_schur(n, coefficients, tables)
            fresh = backend.pauli_to_schur(n, coefficients)
            for a, b in zip(cached, fresh):
                if backend is chang:
                    self.assertEqual((a-b).nnz, 0)
                    self.assertLessEqual(a.nnz, 9*a.shape[0])
                else:
                    np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            chang.pauli_to_schur(n, coefficients, chang.prepare(n, 2))

    def test_public_source_hash_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"utils.py"
            path.write_text("raise RuntimeError('unverified source must not execute')\n")
            with self.assertRaisesRegex(ValueError, "expected utils.py"):
                anschuetz_public.load(path)

    @unittest.skipUnless(os.environ.get("ANSCHUETZ_SOURCE"), "optional pinned public checkout")
    def test_original_public_source_against_independent_projection(self):
        reference = anschuetz_public.load(os.environ["ANSCHUETZ_SOURCE"])
        for n in range(1, 4):
            coefficients = {key: (-1)**i / (i+1)
                            for i, key in enumerate(common.all_orbit_keys(n))}
            expected = explicit_schur_blocks(n, explicit_pauli_matrix(n, coefficients))
            actual = anschuetz_public.pauli_to_schur(n, coefficients, reference)
            for block, target in zip(actual, expected):
                np.testing.assert_allclose(block, target, atol=2e-11, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
