"""Common sparse orbit-image reuse for an exactly fixed Pauli weight.

The supplied builder determines how the orbit SUM images are computed. This
module only stores those images and applies their linear combination. Every
builder uses the same full bandwidth pattern, without numerical thresholding.
For fixed weight, retained storage and an application are O(n**2); the cost of
preparing the images depends on the chosen builder and must be counted separately.
"""

from dataclasses import dataclass
import operator

import numpy as np
from scipy import sparse

from .common import block_shapes


@dataclass(frozen=True)
class OrbitImageCache:
    n: int
    weight: int
    keys: tuple
    indices: tuple
    indptr: tuple
    offsets: tuple
    values: np.ndarray
    max_forbidden_abs: float
    max_forbidden_relative: float


def fixed_weight_keys(n, weight):
    """Sorted (w,g0,g1) keys for all orbit sums of exactly this weight."""
    n, weight = operator.index(n), operator.index(weight)
    if n < 0 or not 0 <= weight <= n:
        raise ValueError("require n >= 0 and 0 <= weight <= n")
    return tuple((w, weight - w, ny)
                 for w in range(weight + 1) for ny in range(w + 1))


def _pattern(side, weight):
    rows = []
    columns = []
    pointers = [0]
    for row in range(side):
        selected = range(max(0, row - weight), min(side, row + weight + 1))
        columns.extend(selected)
        rows.extend([row] * len(selected))
        pointers.append(len(columns))
    return (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32),
            np.asarray(pointers, dtype=np.int32))


def _read_orbit(n, key, blocks, rows, indices, offsets, destination, atol, rtol):
    blocks = tuple(blocks)
    shapes = block_shapes(n)
    if len(blocks) != len(shapes):
        raise ValueError("orbit builder must return every representative Schur block")
    flip_weight = key[0]
    max_forbidden_abs = max_forbidden_relative = 0.0
    for sector, (block, side) in enumerate(zip(blocks, shapes)):
        if block.shape != (side, side):
            raise ValueError(f"wrong orbit block shape for sector {sector}")
        if sparse.issparse(block):
            matrix = sparse.csr_matrix(block, dtype=np.complex128)
            entries = matrix.tocoo(copy=False)
            displacement = entries.row - entries.col
            forbidden = ((np.abs(displacement) > flip_weight)
                         | ((displacement - flip_weight) % 2 != 0))
            absolute = np.abs(entries.data)
            selected = np.asarray(matrix[rows[sector], indices[sector]]).ravel()
        else:
            matrix = np.asarray(block, dtype=np.complex128)
            coordinates = np.arange(side)
            displacement = coordinates[:, None] - coordinates[None, :]
            forbidden = ((np.abs(displacement) > flip_weight)
                         | ((displacement - flip_weight) % 2 != 0))
            absolute = np.abs(matrix)
            selected = matrix[rows[sector], indices[sector]]
        if not np.isfinite(absolute).all():
            raise ValueError(f"non-finite orbit image for key {key}, sector {sector}")
        scale = float(np.max(absolute, initial=0.0))
        leakage = float(np.max(absolute[forbidden], initial=0.0))
        relative = leakage / scale if scale else 0.0
        max_forbidden_abs = max(max_forbidden_abs, leakage)
        max_forbidden_relative = max(max_forbidden_relative, relative)
        if leakage > atol + rtol * scale:
            raise ValueError(f"forbidden-band or parity leakage for key {key}, sector {sector}: "
                             f"{leakage:.6g} exceeds {atol + rtol * scale:.6g}")
        destination[offsets[sector]:offsets[sector + 1]] = selected
    return max_forbidden_abs, max_forbidden_relative


def prepare(n, weight, build_orbit, *, structure_atol=1e-12, structure_rtol=1e-12):
    """Build and retain every orbit-SUM image of the promised exact weight.

    ``build_orbit(key)`` returns all native dense or sparse Schur blocks for a
    unit coefficient on that orbit. The callback may reuse its own prepared
    factors; their construction is the caller's responsibility and belongs in
    preparation timing. Native blocks are released after each image is packed.

    Before compression, check the individual orbit's exact flip bandwidth and
    displacement parity against ``atol + rtol * max(abs(block))``. No retained
    band entry is thresholded. The complete family includes both displacement
    parities, so its shared pattern contains every diagonal within ``weight``.
    """
    keys = fixed_weight_keys(n, weight)
    n, weight = operator.index(n), operator.index(weight)
    if (not np.isfinite(structure_atol) or not np.isfinite(structure_rtol)
            or min(structure_atol, structure_rtol) < 0):
        raise ValueError("structure tolerances must be finite and nonnegative")
    patterns = tuple(_pattern(side, weight) for side in block_shapes(n))
    rows, indices, indptr = tuple(zip(*patterns))
    offsets = [0]
    for columns in indices:
        offsets.append(offsets[-1] + columns.size)
    offsets = tuple(offsets)
    values = np.empty((offsets[-1], len(keys)), dtype=np.complex128)
    max_forbidden_abs = max_forbidden_relative = 0.0
    for column, key in enumerate(keys):
        leakage, relative = _read_orbit(n, key, build_orbit(key), rows, indices, offsets,
                                        values[:, column], structure_atol, structure_rtol)
        max_forbidden_abs = max(max_forbidden_abs, leakage)
        max_forbidden_relative = max(max_forbidden_relative, relative)
    for array in (*indices, *indptr, values):
        array.setflags(write=False)
    return OrbitImageCache(n, weight, keys, indices, indptr, offsets, values,
                           max_forbidden_abs, max_forbidden_relative)


def pack_coefficients(cache, coefficients):
    """Pack new orbit-SUM coefficients; reject nonzero terms outside the family."""
    key_set = set(cache.keys)
    for key, value in coefficients.items():
        if value and key not in key_set:
            raise ValueError(f"nonzero Pauli key {key} is outside the prepared weight family")
    packed = np.asarray([coefficients.get(key, 0j) for key in cache.keys], dtype=np.complex128)
    if not np.isfinite(packed).all():
        raise ValueError("coefficients must be finite")
    return packed


def combine(cache, packed):
    """Apply the common stored map to an already packed coefficient vector."""
    packed = np.asarray(packed, dtype=np.complex128)
    if packed.shape != (len(cache.keys),):
        raise ValueError("packed coefficients must match the prepared orbit keys")
    return cache.values @ packed


def format_blocks(cache, values):
    """Construct all CSR blocks, retaining the same full band including zeros.

    Output arrays are independent of the cache and the supplied values, so
    later modification or sparse zero elimination cannot corrupt future calls.
    """
    values = np.asarray(values, dtype=np.complex128)
    if values.shape != (cache.offsets[-1],):
        raise ValueError("combined values must match the prepared band slots")
    blocks = []
    for sector, side in enumerate(block_shapes(cache.n)):
        start, stop = cache.offsets[sector:sector + 2]
        blocks.append(sparse.csr_matrix(
            (values[start:stop].copy(), cache.indices[sector].copy(), cache.indptr[sector].copy()),
            shape=(side, side), copy=False))
    return blocks


def apply(cache, coefficients):
    """Pack, combine and return all sparse Schur blocks for a new input."""
    return format_blocks(cache, combine(cache, pack_coefficients(cache, coefficients)))
