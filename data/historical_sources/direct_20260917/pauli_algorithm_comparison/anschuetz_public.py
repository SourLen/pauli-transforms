"""Adapter for the pinned public Anschuetz code (external utils.py required)."""

import hashlib
from pathlib import Path
from types import ModuleType

import numpy as np

from .common import all_orbit_keys, pauli_type_from_key, zero_schur_blocks

REPOSITORY = "https://github.com/bkiani/symmetric_hamiltonians.git"
COMMIT = "24ce1a5fe4f3234f5f9a2f1ad65c2909423d7dc8"
UTILS_SHA256 = "ea99d1dda523facaea60ba2cf46b6c636ecc2465dd660b581d90e59dfc7995cd"


def load(path):
    """Load the pinned utils.py, given its path or the checkout directory.

    The same two import compatibility patches as the original adapter are
    applied in memory. The block-construction algorithm remains unchanged.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "utils.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UTILS_SHA256:
        raise ValueError(f"expected utils.py from commit {COMMIT}")
    source = raw.decode().replace("fac_fun = np.math.factorial", "fac_fun = math.factorial")
    source = source.replace(
        "from sympy.utilities.iterables import multiset_permutations",
        "def multiset_permutations(*args, **kwargs):\n"
        "\traise RuntimeError('The block adapter does not expose the SymPy full-matrix helper')",
    )
    module = ModuleType("_anschuetz_public")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def prepare(n, reference, keys=None):
    """Precompute selected public get_matrices columns, or all keys by default."""
    selected = all_orbit_keys(n) if keys is None else dict.fromkeys(keys)
    tables = {}
    for key in selected:
        _, nx, ny, nz = pauli_type_from_key(n, key)
        tables[key] = tuple(np.asarray(block, dtype=np.complex128)
                            for block in reference.get_matrices(n, nx, ny, nz))
    return tables


def pauli_to_schur(n, coefficients, reference=None, tables=None):
    """Call public construct_matrix_blocks, or combine explicitly prepared tables."""
    active = {key: value for key, value in coefficients.items() if value}
    if tables is not None:
        blocks = zero_schur_blocks(n)
        for key, value in active.items():
            for k, column in enumerate(tables[key]):
                blocks[k] += value * column
        return blocks
    if reference is None:
        raise ValueError("provide a loaded reference or prepared tables")
    if not active:
        return zero_schur_blocks(n)
    counts = [pauli_type_from_key(n, key)[1:] for key in active]
    return [np.asarray(block, dtype=np.complex128)
            for block in reference.construct_matrix_blocks(n, counts, list(active.values()))]
