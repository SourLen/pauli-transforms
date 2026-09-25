"""Pauli <-> Schur conversion with selectable Hahn or shared-factor backend.

Hahn recurrence remains the default for compatibility with saved benchmarks.
Select backend="factorial" for the Chapter 5 matrix-product implementation.
dtype controls the Schur stage only; the Krawtchouk stage uses float64 tables.
"""

import numpy as np

from . import separated, schur_hahn, schur_factorial
from .common import all_orbit_keys


def prepare(n, *, backend="hahn", dtype=np.float64):
    """Build fresh Krawtchouk tables and the explicitly selected Schur tables."""
    if backend not in ("hahn", "factorial"):
        raise ValueError("Schur backend must be 'hahn' or 'factorial'")
    module = schur_hahn if backend == "hahn" else schur_factorial
    schur_tables = module.prepare(n, dtype=np.dtype(dtype).type)
    return separated.prepare(n), schur_tables


def _resolve(n, tables, backend, dtype):
    if tables is None:
        tables = prepare(n, backend="hahn" if backend is None else backend,
                         dtype=np.float64 if dtype is None else dtype)
    elif dtype is not None:
        raise ValueError("specify dtype when preparing tables, not when reusing them")
    matrices, kernels = tables
    actual = "factorial" if isinstance(kernels, schur_factorial.FactorTables) else "hahn"
    if backend is not None and backend != actual:
        raise ValueError("requested Schur backend does not match the supplied tables")
    module = schur_factorial if actual == "factorial" else schur_hahn
    return matrices, kernels, module


def pauli_to_orbit(n, coefficients, matrices=None):
    """Pauli (w,g0,g1) -> matrix-unit (r,s,t), storing common values."""
    entries = separated.inverse(n, coefficients, matrices)
    return {(w + h0 - h1, h0 + h1, h0): value
            for (w, h0, h1), value in entries.items()}


def orbit_to_pauli(n, coefficients, matrices=None):
    """Matrix-unit (r,s,t) -> Pauli (w,g0,g1)."""
    entries = {(w, h0, h1): coefficients.get((w + h0 - h1, h0 + h1, h0), 0j)
               for w, h0, h1 in all_orbit_keys(n)}
    return separated.decompose(n, entries, matrices)


def pauli_to_schur(n, coefficients, tables=None, *, backend=None, dtype=None):
    """Return canonical Schur blocks; setup is included unless tables are supplied."""
    matrices, kernels, module = _resolve(n, tables, backend, dtype)
    return module.orbit_to_schur(n, pauli_to_orbit(n, coefficients, matrices), kernels)


def schur_to_pauli(n, blocks, tables=None, *, backend=None, dtype=None):
    """Return the common coefficient of each individual word in each Pauli orbit."""
    matrices, kernels, module = _resolve(n, tables, backend, dtype)
    return orbit_to_pauli(n, module.schur_to_orbit(n, blocks, kernels), matrices)
