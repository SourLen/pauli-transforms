"""Compressed collective-spin applications for the thesis experiments.

The Pauli dictionary convention is that of :mod:`common`: ``(w,g0,g1)``
means ``(nI,nX,nY,nZ)=(n-w-g0,w-g1,g1,g0)``.  Values are the common
coefficients of *individual* Pauli strings, not normalized orbit sums.
Schur states below are representative blocks, so traces require the Specht
multiplicities.  No function here constructs a ``2**n`` square matrix.

The Ising normalization is Eq. (25) of Louloudis et al., arXiv:2608.02038v1:
``H = -(g sum_{i<j} Zi Zj + h sum_i Xi)/n``.  The writing guide's rotated
model has a differently normalized field; the two are not interchanged.
One-axis twisting uses ``H = chi Sz**2`` and ``hbar = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from . import schur_separated
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


def twisting_pauli(n, chi=1.0):
    """Input coefficients for ``H=chi Sz**2`` (constant shift retained)."""
    _qubits(n)
    chi = _real(chi, "chi")
    result = {(0, 0, 0): chi * n / 4}
    if n >= 2:
        result[0, 2, 0] = chi / 2
    return result


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


def collective_observable_blocks(n, axis):
    """Return blocks of ``sum_i sigma_axis(i)/n``."""
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y, or z")
    index = ("x", "y", "z").index(axis)
    return [2 / n * spin_matrices(n, k)[index] for k in range(n // 2 + 1)]


def zz_observable_blocks(n):
    """Return blocks of the uniform average of distinct-site ``Zi Zj``."""
    _qubits(n)
    if n < 2:
        raise ValueError("a two-site correlation requires n>=2")
    result = []
    for k in range(n // 2 + 1):
        _, _, sz = spin_matrices(n, k)
        result.append((4 * sz @ sz - n * np.eye(len(sz))) / (n * (n - 1)))
    return result


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


@dataclass(frozen=True)
class GibbsState:
    blocks: tuple[np.ndarray, ...]
    log_partition: float
    energy: float
    sector_probabilities: np.ndarray


def gibbs_state(n, spectrum, beta):
    """Return all-sector Gibbs blocks with a common stable spectral shift.

    ``log_partition`` remains usable when the ordinary partition function
    overflows.  No projection onto the largest-spin block is performed.
    """
    _validate_blocks(n, spectrum.vectors)
    beta = _real(beta, "beta")
    reference = (min(float(e[0]) for e in spectrum.energies) if beta >= 0
                 else max(float(e[-1]) for e in spectrum.energies))
    weights = [np.exp(-beta * (e - reference)) for e in spectrum.energies]
    multiplicities = specht_multiplicities(n)
    partition_shifted = sum(mu * float(np.sum(w)) for mu, w in zip(multiplicities, weights))
    blocks = tuple((v * (w / partition_shifted)[None, :]) @ v.conj().T
                   for v, w in zip(spectrum.vectors, weights))
    probabilities = np.array([mu * float(np.sum(w)) / partition_shifted
                              for mu, w in zip(multiplicities, weights)])
    energy = sum(mu * float(np.dot(e, w)) / partition_shifted
                 for mu, e, w in zip(multiplicities, spectrum.energies, weights))
    return GibbsState(blocks, float(np.log(partition_shifted) - beta * reference),
                      energy, probabilities)


def evolve_blocks(spectrum, rho, time):
    """Closed evolution ``exp(-itH) rho exp(itH)`` in each spin sector."""
    time = _real(time, "time")
    if len(rho) != len(spectrum.energies):
        raise ValueError("state and Hamiltonian must have the same sectors")
    result = []
    for state, energies, basis in zip(rho, spectrum.energies, spectrum.vectors):
        state_e = basis.conj().T @ state @ basis
        phase = np.exp(-1j * time * (energies - energies[0]))
        result.append(basis @ (phase[:, None] * state_e * phase.conj()[None, :]) @ basis.conj().T)
    return result


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


def twisting_x_exact(n, chi, times, p=1.0):
    """Independent ``<Xi>(t)=p cos(chi*t)**(n-1)`` for the x-product input."""
    _qubits(n)
    chi, p = _real(chi, "chi"), _real(p, "p")
    if abs(p) > 1:
        raise ValueError("p must lie in [-1,1]")
    return p * np.cos(chi * np.asarray(times, dtype=float)) ** (n - 1)


def depolarize_pauli(n, coefficients, p):
    """Apply D_p**tensor(n), D_p(A)=(1-p)A+p(XAX+YAY+ZAZ)/3."""
    _qubits(n)
    p = _real(p, "p")
    if not 0 <= p <= 1:
        raise ValueError("depolarizing p must lie in [0,1]")
    eta = 1 - 4 * p / 3
    result = {}
    for key, value in coefficients.items():
        ni, _, _, _ = pauli_type_from_key(n, key)
        result[key] = value * eta ** (n - ni)
    return result


def depolarize_blocks(n, blocks, p, tables=None):
    """Schur -> Pauli -> identical local depolarization -> Schur.

    Return ``(noisy_blocks, noisy_pauli)``; every conversion is the existing
    separated Krawtchouk/Hahn implementation.  Supplying tables reuses setup.
    """
    _validate_blocks(n, blocks)
    if tables is None:
        tables = schur_separated.prepare(n)
    coefficients = schur_separated.schur_to_pauli(n, blocks, tables)
    noisy = depolarize_pauli(n, coefficients, p)
    return schur_separated.pauli_to_schur(n, noisy, tables), noisy


def dephase_pauli(n, coefficients, gamma, time):
    """Apply exp(t L), L(A)=gamma sum_i(Zi A Zi-A), in Pauli coordinates."""
    _qubits(n)
    gamma, time = _real(gamma, "gamma"), _real(time, "time")
    if gamma < 0 or time < 0:
        raise ValueError("gamma and time must be nonnegative")
    result = {}
    for key, value in coefficients.items():
        _, nx, ny, _ = pauli_type_from_key(n, key)
        result[key] = value * np.exp(-2 * gamma * time * (nx + ny))
    return result


def pauli_orbit_size(n, key):
    """Number of individual strings in a Pauli orbit, as an exact integer."""
    ni, nx, ny, _ = pauli_type_from_key(n, key)
    return comb(n, ni) * comb(n - ni, nx) * comb(n - ni - nx, ny)


def pauli_weight_statistics(n, coefficients, *, threshold=1e-12):
    """Orbit-weighted Pauli norms and numerical support grouped by weight.

    Norms include every supplied value.  Only support counts use the absolute
    *individual coefficient* threshold.  ``2**n * coefficient_l2_squared``
    is ``Tr(A†A)``.  Squared norms can be divided by their total to plot the
    fraction of Hilbert--Schmidt weight, independent of overall normalization.
    """
    _qubits(n)
    threshold = _real(threshold, "threshold")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    l1 = np.zeros(n + 1)
    squared = np.zeros(n + 1)
    orbits, strings = [0] * (n + 1), [0] * (n + 1)
    for key, value in coefficients.items():
        ni, _, _, _ = pauli_type_from_key(n, key)
        weight, size, magnitude = n - ni, pauli_orbit_size(n, key), abs(value)
        l1[weight] += size * magnitude
        squared[weight] += size * magnitude ** 2
        if magnitude > threshold:
            orbits[weight] += 1
            strings[weight] += size
    return {"weight": np.arange(n + 1), "l1_by_weight": l1,
            "squared_norm_by_weight": squared, "l2_squared_by_weight": squared,
            "orbit_count_by_weight": orbits,
            "string_count_by_weight": strings, "coefficient_l1": float(np.sum(l1)),
            "coefficient_l2_squared": float(np.sum(squared)),
            "hilbert_schmidt_squared": float(2.0 ** n * np.sum(squared)),
            "threshold": threshold}
