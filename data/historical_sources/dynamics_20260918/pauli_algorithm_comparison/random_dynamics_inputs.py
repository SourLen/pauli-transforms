"""Shared random Pauli inputs and untimed references for invariant dynamics.

H and M use the general Gaussian orbit ensemble of the spectral benchmark.
For d_k=n-2k+1, independent d_k by 2d_k complex Gaussian matrices give
rho_k=(d_k/2**n) G_k G_k^dagger / Tr(G_k G_k^dagger).  Thus the complete
state has trace one, including Specht multiplicities, and mean I/2**n.
The state inverse conversion is input generation, never timed conversion work.
Prepared with Codex assistance in September 2026.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from . import anschuetz_optimized, benchmark_cases, physical_models as pm
from . import schur_full_ed, schur_separated
from .common import block_shapes, specht_multiplicities


def make_inputs(n, seed):
    """Return three Pauli mappings and the independently generated state blocks."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    streams = np.random.SeedSequence(seed).spawn(3)
    seeds = [int(stream.generate_state(1, dtype=np.uint64)[0]) for stream in streams]
    h = benchmark_cases.make_case(n, "general", seeds[0], locality=0)
    observable = benchmark_cases.make_case(n, "general", seeds[1], locality=0)
    rng = np.random.default_rng(seeds[2])
    state = []
    for side in block_shapes(n):
        gaussian = rng.normal(size=(side, 2 * side)) + 1j * rng.normal(size=(side, 2 * side))
        positive = gaussian @ gaussian.conj().T
        state.append((side / float(2**n)) * positive / float(np.trace(positive).real))
    coefficients = schur_separated.schur_to_pauli(n, state, backend="hahn")
    # Preserve the returned complex coefficients, including roundoff. No clipping
    # or Hermitian/positive projection is used to make an input pass validation.
    return (h, coefficients, observable), state


def relative_blocks(actual, expected, weights=None):
    if len(actual) != len(expected):
        raise ValueError("block lists must have equal length")
    if weights is None:
        weights = [1.] * len(expected)
    numerator = sum(float(w) * float(np.linalg.norm(a - b, "fro") ** 2)
                    for w, a, b in zip(weights, actual, expected, strict=True))
    denominator = sum(float(w) * float(np.linalg.norm(b, "fro") ** 2)
                      for w, b in zip(weights, expected, strict=True))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def state_diagnostics(n, state):
    """Diagnostics of the supplied state, with physical multiplicity factors."""
    if len(state) != len(block_shapes(n)) or any(
            a.shape != (side, side) for a, side in zip(state, block_shapes(n), strict=True)):
        raise ValueError("invalid state block shapes")
    if any(not np.all(np.isfinite(a)) for a in state):
        raise ArithmeticError("state contains nonfinite values")
    weights = specht_multiplicities(n)
    norm = sum(float(mu) * float(np.linalg.norm(a, "fro") ** 2)
               for mu, a in zip(weights, state, strict=True))
    skew = sum(float(mu) * float(np.linalg.norm(a - a.conj().T, "fro") ** 2)
               for mu, a in zip(weights, state, strict=True))
    # The Hermitian part defines this diagnostic only; the input is not changed.
    negative = sum(float(mu) * float(np.maximum(
        -np.linalg.eigvalsh((a + a.conj().T) / 2), 0).sum())
        for mu, a in zip(weights, state, strict=True))
    relative_negative = max(float(np.maximum(
        -np.linalg.eigvalsh((a + a.conj().T) / 2), 0).sum())
        / max(float(np.linalg.norm(a, "fro")), np.finfo(float).tiny) for a in state)
    return dict(state_trace_error=float(abs(pm.block_trace(n, state) - 1)),
                state_hermiticity_error=float(np.sqrt(skew / max(norm, np.finfo(float).tiny))),
                state_max_block_hermiticity_error=operator_skew(state),
                state_negative_mass=negative, state_relative_negative_mass=relative_negative)


def operator_skew(blocks):
    return max(float(np.linalg.norm(a - a.conj().T, "fro"))
               / max(float(np.linalg.norm(a, "fro")), np.finfo(float).tiny) for a in blocks)


def require_valid(metrics, tolerance, context):
    invalid = {key: value for key, value in metrics.items()
               if not np.isfinite(value) or value < 0 or value > tolerance}
    if invalid:
        raise ArithmeticError(f"{context}: {invalid}")


def independent_reference(n, inputs, generated_state, *, dense_max_n=5, tolerance=1e-8):
    """Construct all three with Appendix E and validate against original rho.

    At small n, literal full Pauli matrices additionally check every block,
    state positivity and normalization. Larger systems use cross-route checks;
    this is not a claim of full-Hilbert-space validation at those sizes.
    """
    keys = dict.fromkeys(key for mapping in inputs for key, value in mapping.items() if value)
    tables = anschuetz_optimized.prepare(n, keys)
    converted = tuple(anschuetz_optimized.pauli_to_schur(n, mapping, tables) for mapping in inputs)
    metrics = state_diagnostics(n, generated_state)
    metrics.update({"converted_" + key: value for key, value in state_diagnostics(n, converted[1]).items()})
    metrics.update(state_inverse_error=relative_blocks(converted[1], generated_state, specht_multiplicities(n)),
                   h_hermiticity_error=operator_skew(converted[0]),
                   observable_hermiticity_error=operator_skew(converted[2]))
    dense = None
    if n <= dense_max_n:
        dense = tuple(schur_full_ed.dense_operator(n, mapping) for mapping in inputs)
        bases = schur_full_ed.prepare(n)
        projected = tuple([basis.conj() @ matrix @ basis.T for basis in bases] for matrix in dense)
        for name, actual, expected in zip(("h", "state", "observable"), converted, projected, strict=True):
            metrics["dense_" + name + "_block_error"] = relative_blocks(actual, expected)
        metrics["dense_state_trace_error"] = float(abs(np.trace(dense[1]) - 1))
        metrics["dense_state_negative_mass"] = float(np.maximum(
            -np.linalg.eigvalsh((dense[1] + dense[1].conj().T) / 2), 0).sum())
        metrics["dense_state_hermiticity_error"] = operator_skew([dense[1]])
    require_valid(metrics, tolerance, "Random input reference failed")
    # Retain the originally generated rho as the reference; its forward transform
    # is checked above rather than used to define the answer to be reproduced.
    return (converted[0], generated_state, converted[2]), dense, metrics


def full_space_curve(dense, times):
    """Independent full-space unitaries and matrix products, without block kernels."""
    h, state, observable = dense
    result = []
    for time in times:
        unitary = expm(-1j * time * h)
        evolved = unitary @ state @ unitary.conj().T
        result.append(np.trace(evolved @ observable))
    return np.asarray(result, dtype=complex)


def spectrum_diagnostics(blocks, spectrum):
    metrics = dict(eigenvector_residual=0., eigenvector_orthogonality_error=0.)
    for block, values, vectors in zip(blocks, spectrum.energies, spectrum.vectors, strict=True):
        scale = max(float(np.linalg.norm(block, "fro")), np.finfo(float).tiny)
        metrics["eigenvector_residual"] = max(metrics["eigenvector_residual"],
            float(np.linalg.norm(block @ vectors - vectors * values, "fro")) / scale)
        metrics["eigenvector_orthogonality_error"] = max(metrics["eigenvector_orthogonality_error"],
            float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(len(values)), "fro")))
    return metrics
