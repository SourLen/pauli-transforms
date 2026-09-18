"""Permutation-invariant Pauli, matrix-entry, and Schur coordinates.

Use ``prepare(n)`` once and pass the returned tables to repeated Pauli/Schur
conversions. The default is the Hahn recurrence used in the thesis timings;
``prepare(n, backend="factorial")`` selects the Chapter 5 factorization.

Pauli keys ``(w,g0,g1)`` count X/Y sites, Z sites, and Y sites respectively.
Coefficients multiply unnormalized orbit sums. Schur blocks appear in order
``k=0,...,n//2`` with side ``n-2*k+1`` and basis weights ``k,...,n-k``.
"""

from .krawtchouk import decompose as entries_to_pauli
from .krawtchouk import inverse as pauli_to_entries
from .transforms import prepare, pauli_to_schur, schur_to_pauli

__all__ = [
    "prepare", "pauli_to_schur", "schur_to_pauli",
    "entries_to_pauli", "pauli_to_entries",
]
