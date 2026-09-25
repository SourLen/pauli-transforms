"""Convention checks against computational-basis matrices, never timed."""
import os
os.environ['PERMQIT_USE_GPU'] = 'false'
import numpy as np
import pytest
pytest.importorskip('permqit')
from . import permqit_adapter as adapter, schur_full_ed
from .common import specht_multiplicities
from .permqit_comparison_reference import (
    keys, dense_matrix, random_entries, integer_kernel_blocks, block_errors,
)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_every_matrix_unit_orbit(n):
    prepared = adapter.prepare(n)
    bases = schur_full_ed.prepare(n)
    for key in keys(n):
        coefficients = {key: 1j}  # Also detects accidental conjugation.
        matrix = dense_matrix(n, coefficients)
        dense_blocks = [u.conj() @ matrix @ u.T for u in bases]
        native = adapter.orbit_to_schur(n, coefficients, prepared)
        exact = integer_kernel_blocks(n, coefficients)
        for a, b, c in zip(native, dense_blocks, exact):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
            np.testing.assert_allclose(c, b, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize('n', [6, 7, 8])
@pytest.mark.parametrize('hermitian', [False, True])
def test_dense_projection_and_weighted_norm(n, hermitian):
    coefficients = random_entries(n, 92000+n, hermitian=hermitian)
    matrix = dense_matrix(n, coefficients)
    bases = schur_full_ed.prepare(n)
    reference = [u.conj() @ matrix @ u.T for u in bases]
    blocks = adapter.orbit_to_schur(n, coefficients)
    assert block_errors(n, blocks, reference)['passed']
    assert block_errors(n, integer_kernel_blocks(n, coefficients), reference)['passed']
    norm = sum(m*np.linalg.norm(a)**2 for m, a in zip(specht_multiplicities(n), blocks))
    np.testing.assert_allclose(norm, np.linalg.norm(matrix)**2, atol=1e-12)
    if hermitian:
        for a in blocks:
            np.testing.assert_allclose(a, a.conj().T, atol=1e-13)
        spectrum = np.concatenate([np.repeat(np.linalg.eigvalsh(a), m)
                                   for m, a in zip(specht_multiplicities(n), blocks)])
        np.testing.assert_allclose(np.sort(spectrum), np.linalg.eigvalsh(matrix), atol=1e-12)


def test_identity_and_multiplication():
    n = 4
    prepared = adapter.prepare(n)
    identity = {(r, r, r): 1 for r in range(n+1)}
    for block in adapter.orbit_to_schur(n, identity, prepared):
        np.testing.assert_allclose(block, np.eye(len(block)), atol=1e-13)
    a, b = (random_entries(n, seed) for seed in [1, 2])
    product = dense_matrix(n, a) @ dense_matrix(n, b)
    actual = [x @ y for x, y in zip(adapter.orbit_to_schur(n, a, prepared),
                                   adapter.orbit_to_schur(n, b, prepared))]
    reference = [u.conj() @ product @ u.T for u in schur_full_ed.prepare(n)]
    assert block_errors(n, actual, reference)['passed']


@pytest.mark.parametrize('key,value', [((0,0,1), 1), ((True,0,0), 1),
                                     ((1.,0,0), 1), ((0,0,0), np.nan)])
def test_rejects_invalid_coordinates(key, value):
    with pytest.raises(ValueError):
        adapter.orbit_to_schur(2, {key: value})
