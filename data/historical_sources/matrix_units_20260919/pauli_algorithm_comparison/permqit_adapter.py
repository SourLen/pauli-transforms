"""Thin CPU adapter for permqit's native orbit-to-orthonormal-block map.

Optional dependency; no permqit import occurs until prepare(). Tested against
upstream commit 22af3cd245bd0e950df49f6ce16ccd422b6eb2c5 (MIT).
No local recurrence or coefficient construction replaces the upstream map.
"""

from dataclasses import dataclass
import operator

import numpy as np

UPSTREAM_COMMIT = '22af3cd245bd0e950df49f6ce16ccd422b6eb2c5'


@dataclass
class PreparedPermqit:
    n: int
    isomorphism: object
    input_indices: dict
    output_orders: tuple
    sparse_format: object
    numeric_map_bytes: int
    nnz: int


def _sparse_bytes(matrix):
    # The native n=1 identity is DIA; the nontrivial maps are CSR.
    return sum(getattr(matrix, name).nbytes
               for name in ('data', 'indices', 'indptr', 'offsets')
               if hasattr(matrix, name))


def prepare(n):
    """Force native Gijswijt construction AND Gram/Cholesky normalization.

    permqit memoizes objects. Use a fresh process for a cold preparation timing.
    Set PERMQIT_USE_GPU=false before importing it for the CPU comparison.
    """
    if isinstance(n, (bool, np.bool_)):
        raise ValueError('n must be a positive integer')
    n = operator.index(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    from permqit.representation.isomorphism import (
        EndSnAlgebraIsomorphism, EndSnBlockDiagonalizationGijswijt,
    )
    from permqit.representation.young_tableau import SSYT
    from permqit.algebra.linear_map import StorageFormat
    from permqit.utilities import backend
    if backend.USE_GPU:
        raise RuntimeError('this adapter requires PERMQIT_USE_GPU=false before import')
    raw = EndSnBlockDiagonalizationGijswijt(n, 2)
    iso = EndSnAlgebraIsomorphism(raw)
    fmt = StorageFormat.SCIPY_SPARSE
    matrix = iso.coefficient_transition_matrix(fmt)
    raw_matrix = raw.coefficient_transition_matrix(fmt)
    indices = {}
    for r in range(n + 1):
        for s in range(n + 1):
            for t in range(max(0, r + s - n), min(r, s) + 1):
                counts = np.array([[n-r-s+t, s-t], [r-t, t]])
                indices[r, s, t] = iso.basis_from.count_matrix_to_index(counts)
    orders = []
    for native_index, partition in enumerate(iso.basis_to.partitions):
        k = n - partition.as_tuple()[0]
        weights = [int(sum(tab.flat)) for tab in SSYT.generate_all(partition, 2)]
        if sorted(weights) != list(range(k, n-k+1)):
            raise ValueError('unexpected qubit tableau weights in permqit')
        orders.append((k, native_index, np.argsort(weights)))
    map_bytes = _sparse_bytes(matrix)
    if raw_matrix is not matrix:
        map_bytes += _sparse_bytes(raw_matrix)
    return PreparedPermqit(n, iso, indices, tuple(sorted(orders)), fmt,
                          map_bytes, matrix.nnz)


def orbit_to_schur(n, coefficients, prepared=None):
    """Literal entries A[x,y]=a[|x|,|y|,|x & y|] -> blocks k=0,...,floor(n/2).

    All input packing, native sparse application, and output reordering are
    included here. Complex coefficients and non-Hermitian operators are allowed.
    """
    if prepared is None:
        prepared = prepare(n)
    if prepared.n != n:
        raise ValueError('prepared map has a different n')
    vector = np.zeros(len(prepared.input_indices), dtype=np.complex128)
    for key, value in coefficients.items():
        try:
            if len(key) != 3 or any(isinstance(x, (bool, np.bool_)) for x in key):
                raise ValueError
            index = prepared.input_indices[tuple(operator.index(x) for x in key)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'invalid matrix-unit key {key!r} for n={n}') from exc
        scalar = np.asarray(value, dtype=np.complex128)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError('matrix-unit coefficients must be finite scalars')
        vector[index] = scalar.item()
    output = prepared.isomorphism.apply_to_coefficient_vector(
        vector, format=prepared.sparse_format,
    )
    native = prepared.isomorphism.basis_to.linear_combination(output).blocks
    return [np.array(native[j][np.ix_(order, order)], dtype=np.complex128)
            for _, j, order in prepared.output_orders]
