"""Hahn recurrence: matrix-unit orbit coefficients <-> Schur blocks."""

from __future__ import annotations
from fractions import Fraction
from math import comb, log, sqrt
from typing import Sequence
import numpy as np
from .common import kappa, orbit_size, specht_multiplicities, t_values, zero_schur_blocks


def log_orbit_sizes(n, r, s, t_vals):
    return np.array([log(orbit_size(n, r, s, t)) for t in t_vals])


def hahn_parameters(n: int, r: int, s: int) -> tuple[int, int, int]:
    """Return ``(alpha, beta, N)`` for the weight pair ``(r, s)``."""
    r_lo, r_hi = (min(r, s), max(r, s))
    return (-(n - r_hi) - 1, -r_hi - 1, r_lo)


def _ratio(numerator: int, denominator: int, dtype):
    """Divide two exact integers in the requested arithmetic."""
    if dtype is Fraction:
        return Fraction(numerator, denominator)
    return dtype(numerator) / dtype(denominator)


def hahn_columns(n: int, r: int, s: int, *, dtype=np.float64) -> tuple[tuple[int, ...], np.ndarray]:
    """Return ``(t_vals, Q)`` with ``Q[k, j] = Q_k(r_lo - t_vals[j])``."""
    t_vals = t_values(n, r, s)
    limit = kappa(n, r, s)
    alpha, beta, degree_n = hahn_parameters(n, r, s)
    r_lo = min(r, s)
    exact = dtype is Fraction
    array_dtype = object if exact else dtype
    convert = Fraction if exact else dtype
    nodes = np.array([convert(r_lo - t) for t in t_vals], dtype=array_dtype)
    previous = np.array([convert(0)] * len(t_vals), dtype=array_dtype)
    current = np.array([convert(1)] * len(t_vals), dtype=array_dtype)
    columns = np.empty((limit + 1, len(t_vals)), dtype=array_dtype)
    for k in range(limit + 1):
        columns[k, :] = current
        if k == limit:
            break
        a = _ratio(
            (k + alpha + beta + 1) * (k + alpha + 1) * (degree_n - k),
            (2 * k + alpha + beta + 1) * (2 * k + alpha + beta + 2), dtype,
        )
        c = _ratio(
            k * (k + alpha + beta + degree_n + 1) * (k + beta),
            (2 * k + alpha + beta) * (2 * k + alpha + beta + 1), dtype,
        )
        previous, current = (current, ((a + c - nodes) * current - c * previous) / a)
    return (t_vals, columns)


def build_U(
    n: int, r: int, s: int, mu: Sequence[int] | None = None, *, dtype=np.float64
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return ``(t_vals, U)`` for one weight pair."""
    if dtype is Fraction or not np.issubdtype(np.dtype(dtype), np.floating):
        raise TypeError('build_U needs a floating dtype')
    limit = kappa(n, r, s)
    if mu is None:
        mu = specht_multiplicities(n)
    if len(mu) <= limit:
        raise ValueError(f'mu must cover k=0..{limit} for n={n}, r={r}, s={s}')
    t_vals, columns = hahn_columns(n, r, s, dtype=dtype)
    log_v = log_orbit_sizes(n, r, s, t_vals)
    r_lo, r_hi = (min(r, s), max(r, s))
    log_denominator = log(comb(n, r_lo)) + log(comb(n, r_hi))
    transform = np.empty((limit + 1, limit + 1), dtype=dtype)
    log_rho = 0.0
    with np.errstate(divide='ignore'):
        log_abs_q = np.log(np.abs(columns))
    for k in range(limit + 1):
        log_scale = 0.5 * (log(mu[k]) + log_v + log_rho - log_denominator)
        transform[k, :] = np.sign(columns[k]) * np.exp(log_scale + log_abs_q[k])
        if k < limit:
            log_rho += log(r_lo - k) + log(n - r_hi - k) - log(r_hi - k) - log(n - r_lo - k)
    if not np.all(np.isfinite(transform)):
        raise ArithmeticError(f'non-finite Hahn kernel at n={n}, r={r}, s={s}; increase precision')
    return (t_vals, transform)


def prepare(n, *, dtype=np.float64):
    """Build every distinct Hahn kernel afresh."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    return {(r, s): build_U(n, r, s, dtype=dtype)
            for r in range(n + 1) for s in range(r, n + 1)}


def orbit_to_schur(n, coefficients, kernels=None):
    """Input keys (r,s,t) hold literal common matrix entries."""
    blocks = zero_schur_blocks(n)
    mu = specht_multiplicities(n)
    for r in range(n + 1):
        for s in range(n + 1):
            ts, u = build_U(n, r, s) if kernels is None else kernels[min(r, s), max(r, s)]
            roots = np.sqrt([float(orbit_size(n, r, s, t)) for t in ts])
            x = roots * np.array([coefficients.get((r, s, t), 0j) for t in ts])
            y = u @ x
            for k, value in enumerate(y):
                blocks[k][r - k, s - k] = value / sqrt(mu[k])
    return blocks


def schur_to_orbit(n, blocks, kernels=None):
    """Inverse Hahn transform using U.T and the same orbit weights."""
    mu = specht_multiplicities(n)
    output = {}
    for r in range(n + 1):
        for s in range(n + 1):
            ts, u = build_U(n, r, s) if kernels is None else kernels[min(r, s), max(r, s)]
            y = np.array([sqrt(mu[k]) * blocks[k][r - k, s - k] for k in range(len(ts))])
            roots = np.sqrt([float(orbit_size(n, r, s, t)) for t in ts])
            x = (u.T @ y) / roots
            for t, value in zip(ts, x):
                output[r, s, t] = complex(value)
    return output
