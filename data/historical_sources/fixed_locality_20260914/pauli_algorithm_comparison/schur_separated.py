"""Pauli <-> Schur transform using separated Krawtchouk and Hahn kernels."""

from . import separated, schur_hahn
from .common import all_orbit_keys


def prepare(n):
    """Build a fresh complete Krawtchouk bank and a fresh complete Hahn bank."""
    return separated.prepare(n), schur_hahn.prepare(n)


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


def pauli_to_schur(n, coefficients, tables=None):
    """Return canonical Schur blocks; setup is included unless tables are supplied."""
    matrices, kernels = prepare(n) if tables is None else tables
    return schur_hahn.orbit_to_schur(n, pauli_to_orbit(n, coefficients, matrices), kernels)


def schur_to_pauli(n, blocks, tables=None):
    """Return the common coefficient of each individual word in each Pauli orbit."""
    matrices, kernels = prepare(n) if tables is None else tables
    return orbit_to_pauli(n, schur_hahn.schur_to_orbit(n, blocks, kernels), matrices)
