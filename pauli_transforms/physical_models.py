"""Ising inputs and multiplicity-weighted dynamics in representative Schur blocks.

Pauli mappings store common individual-string coefficients. The model is
H=-(g sum_{i<j} Z_i Z_j + h sum_i X_i)/n, with hbar=1.
"""

from dataclasses import dataclass
import numpy as np
from .common import all_orbit_keys, block_shapes, pauli_type_from_key, specht_multiplicities


def _qubits(n):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")


def _real(value, name):
    if not np.isreal(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _validate_blocks(n, blocks):
    _qubits(n)
    if len(blocks) != n // 2 + 1:
        raise ValueError("one block is required for every spin sector")
    for block, side in zip(blocks, block_shapes(n)):
        if np.shape(block) != (side, side) or not np.all(np.isfinite(block)):
            raise ValueError("invalid Schur block shape or non-finite entry")


def spin_matrices(n, k=0):
    """Return ``Sx,Sy,Sz`` in canonical order ``m=n/2-k,...,-n/2+k``.

    This independent angular-momentum construction does not call a transform.
    Row ``q`` corresponds to the repo's weight index ``r=k+q``.
    """
    _qubits(n)
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= n // 2:
        raise ValueError("invalid sector k")
    twice_j = n - 2 * k
    q = np.arange(twice_j, dtype=float)
    raising = np.diag(np.sqrt((q + 1) * (twice_j - q)), 1).astype(complex)
    lowering = raising.T
    return ((raising + lowering) / 2, (raising - lowering) / (2j),
            np.diag(twice_j / 2 - np.arange(twice_j + 1)).astype(complex))


def ising_pauli(n, g=1.0, h=0.5):
    """Sparse input for the fully connected Berta/Louloudis Ising model."""
    _qubits(n)
    g, h = _real(g, "g"), _real(h, "h")
    result = {(1, 0, 0): -h / n} if h else {}
    if n >= 2 and g:
        result[0, 2, 0] = -g / n
    return result


def ising_collective_blocks(n, g=1.0, h=0.5):
    """Independent reference blocks, including the ``g/2`` energy shift."""
    _qubits(n)
    g, h = _real(g, "g"), _real(h, "h")
    blocks = []
    for k in range(n // 2 + 1):
        sx, _, sz = spin_matrices(n, k)
        blocks.append(-2 * g / n * (sz @ sz) - 2 * h / n * sx
                      + g / 2 * np.eye(len(sz)))
    return blocks


def product_state_pauli(n, bloch):
    """Coefficients of ``[(I+xX+yY+zZ)/2]**tensor(n)``.

    Mixed inputs are PI but generally occupy all spin sectors.  Nonzero y
    also provides a sensitive check of the transform's Pauli-Y convention.
    """
    _qubits(n)
    if len(bloch) != 3:
        raise ValueError("bloch must contain x,y,z")
    x, y, z = (_real(value, "Bloch coordinate") for value in bloch)
    if x * x + y * y + z * z > 1 + 8 * np.finfo(float).eps:
        raise ValueError("Bloch vector must have norm at most one")
    result = {}
    for key in all_orbit_keys(n):
        _, nx, ny, nz = pauli_type_from_key(n, key)
        coefficient = 2.0 ** (-n) * x ** nx * y ** ny * z ** nz
        if coefficient:
            result[key] = coefficient
    return result


def product_x_pauli(n, p=1.0):
    """Coefficients of ``[(I+pX)/2]**tensor(n)``, for ``-1<=p<=1``."""
    return product_state_pauli(n, (p, 0.0, 0.0))


def product_x_collective_blocks(n, p=0.6):
    """Independent blocks of [(I+pX)/2]^tensor(n), including all sectors.

    In the Sx basis a magnetic level m has eigenvalue
    ((1+p)/2)^(n/2+m) ((1-p)/2)^(n/2-m).
    Integer exponents also handle pure states p=+/-1 without a limit.
    """
    _qubits(n)
    p = _real(p, "p")
    if abs(p) > 1:
        raise ValueError("p must lie in [-1,1]")
    blocks = []
    for k in range(n // 2 + 1):
        sx, _, _ = spin_matrices(n, k)
        _, vectors = np.linalg.eigh(sx)
        q = np.arange(n - 2 * k + 1)
        values = ((1 + p) / 2) ** (k + q) * ((1 - p) / 2) ** (n - k - q)
        blocks.append((vectors * values) @ vectors.conj().T)
    return blocks


def collective_observable_blocks(n, axis):
    """Return blocks of ``sum_i sigma_axis(i)/n``."""
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y, or z")
    index = ("x", "y", "z").index(axis)
    return [2 / n * spin_matrices(n, k)[index] for k in range(n // 2 + 1)]


def block_trace(n, blocks):
    """Full-space trace from representative Schur blocks."""
    _validate_blocks(n, blocks)
    return sum(mu * np.trace(block) for mu, block in zip(specht_multiplicities(n), blocks))


def block_expectation(n, rho, observable):
    """Multiplicity-weighted ``Tr(rho O)``; return complex without discarding error."""
    _validate_blocks(n, rho)
    _validate_blocks(n, observable)
    return sum(mu * np.einsum("ij,ji->", state, operator)
               for mu, state, operator in zip(specht_multiplicities(n), rho, observable))


@dataclass(frozen=True)
class BlockSpectrum:
    energies: tuple[np.ndarray, ...]
    vectors: tuple[np.ndarray, ...]
    hermiticity_max_abs: float


def diagonalize_blocks(blocks, *, hermiticity_atol=2e-11, hermiticity_rtol=2e-11):
    """Diagonalize each Hermitian block and record its original skew residual.

    Only skew below the explicit tolerance is symmetrized.  Large errors raise
    instead of silently returning the spectrum of one matrix triangle.
    """
    energies, vectors, skew = [], [], 0.0
    for block in blocks:
        block = np.asarray(block, dtype=complex)
        if block.ndim != 2 or block.shape[0] != block.shape[1] or not np.all(np.isfinite(block)):
            raise ValueError("expected finite square blocks")
        residual = float(np.max(np.abs(block - block.conj().T)))
        skew = max(skew, residual)
        if residual > hermiticity_atol + hermiticity_rtol * np.max(np.abs(block)):
            raise ArithmeticError(f"Schur block is not Hermitian: residual={residual:.3g}")
        values, basis = np.linalg.eigh((block + block.conj().T) / 2)
        energies.append(values)
        vectors.append(basis)
    return BlockSpectrum(tuple(energies), tuple(vectors), skew)


def expectation_time_series(n, spectrum, rho, observable, times):
    """Preprocess energy-basis contractions once, then use O(sum d_k**2) per time."""
    _validate_blocks(n, rho)
    _validate_blocks(n, observable)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array")
    output = np.zeros(len(times), dtype=complex)
    for mu, energies, basis, state, operator in zip(
            specht_multiplicities(n), spectrum.energies, spectrum.vectors, rho, observable):
        state_e = basis.conj().T @ state @ basis
        operator_e = basis.conj().T @ operator @ basis
        contraction = mu * state_e * operator_e.T
        differences = energies[:, None] - energies[None, :]
        for i, time in enumerate(times):
            output[i] += np.sum(contraction * np.exp(-1j * time * differences))
    return output
