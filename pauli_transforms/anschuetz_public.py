"""Fresh calls to the pinned original Anschuetz block-construction routine.

The external source is not bundled. ``load`` verifies its SHA-256 before
applying two import compatibility fixes in memory. No cached variant is
provided: every conversion calls the original ``construct_matrix_blocks``.
"""

import hashlib
from pathlib import Path
from types import ModuleType

import numpy as np

from .common import pauli_type_from_key, zero_schur_blocks

REPOSITORY = "https://github.com/bkiani/symmetric_hamiltonians.git"
COMMIT = "24ce1a5fe4f3234f5f9a2f1ad65c2909423d7dc8"
UTILS_SHA256 = "ea99d1dda523facaea60ba2cf46b6c636ecc2465dd660b581d90e59dfc7995cd"


def load(path):
    """Load the pinned utils.py, given its path or checkout directory."""
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
        "\traise RuntimeError('The adapter does not expose the SymPy full-matrix helper')",
    )
    module = ModuleType("_anschuetz_public")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def pauli_to_schur(n, coefficients, reference):
    """Convert by a fresh original call using the module returned by ``load``."""
    active = {key: value for key, value in coefficients.items() if value}
    if not active:
        return zero_schur_blocks(n)
    counts = [pauli_type_from_key(n, key)[1:] for key in active]
    return [np.asarray(block, dtype=np.complex128)
            for block in reference.construct_matrix_blocks(n, counts, list(active.values()))]
