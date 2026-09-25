"""Inputs, common representations, and checks for the four comparisons.

Benchmark plumbing stays outside the algorithm modules. Conventions and
timing boundaries follow the existing pauli_symmetry_bench campaigns.
"""

from itertools import combinations
from math import comb, sqrt

import numpy as np
from scipy import sparse

from . import common, schur_full_ed, schur_separated, separated


def pauli_orbit_size(n, key):
    _, nx, ny, nz = common.pauli_type_from_key(n, key)
    return comb(n, nx) * comb(n - nx, ny) * comb(n - nx - ny, nz)


def make_case(n, family, seed, locality):
    """Real coefficients give Hermitian operators, with every sector retained."""
    rng = np.random.default_rng(seed)
    if family == "general":
        keys = list(common.all_orbit_keys(n))
        # E[Tr(A^2)/2^n]=1, accounting for each orbit's multiplicity.
        coefficients = {key: float(rng.normal()) / sqrt(len(keys) * pauli_orbit_size(n, key))
                        for key in keys}
    elif family == "local":
        if locality > n:
            raise ValueError("locality exceeds n")
        keys = [key for key in common.all_orbit_keys(locality)]
        coefficients = {key: float(rng.uniform(0.5, 1.5) * rng.choice([-1, 1]))
                        / (len(keys) * pauli_orbit_size(n, key)) for key in keys}
    elif family == "fixed_weight":
        if locality > n:
            raise ValueError("fixed weight exceeds n")
        # A fixed set of all orbit averages of exactly this weight. Coefficients
        # vary across input instances; no prepared object stores an answer.
        keys = [(nx + ny, locality - nx - ny, ny)
                for nx in range(locality + 1) for ny in range(locality - nx + 1)]
        coefficients = {key: float(rng.uniform(0.5, 1.5) * rng.choice([-1, 1]))
                        / (len(keys) * pauli_orbit_size(n, key)) for key in keys}
    else:
        raise ValueError(f"unknown family {family!r}")
    return coefficients


def encode_mapping(values):
    return [[*key, float(complex(value).real), float(complex(value).imag)]
            for key, value in sorted(values.items())]


def decode_mapping(records):
    return {tuple(row[:-2]): complex(row[-2], row[-1]) for row in records}


class OrbitRowOracle:
    """Sparse queries constructed ONLY from the supplied entry-orbit table.

    Candidate displacements have weights of nonzero entry slices. Enumerating
    them is timed preparation. Each row inspects those displacements and
    returns all nonzero entries. There is no planted Pauli spectrum inside.
    """

    def __init__(self, n, entries, *, max_queries=None):
        self.n = n
        self.entries = dict(entries)
        weights = sorted({w for (w, _, _), value in entries.items() if value != 0})
        self.flips = [(sum(1 << j for j in bits), w) for w in weights
                      for bits in combinations(range(n), w)]
        self.max_queries = max_queries
        self.reset_query_count()

    def reset_query_count(self):
        self.query_count = self.row_queries = self.entry_queries = 0
        self.returned_entries = 0

    def _count(self, row):
        if self.max_queries is not None and self.query_count >= self.max_queries:
            raise RuntimeError("query_limit: configured oracle query budget exhausted")
        self.query_count += 1
        if row:
            self.row_queries += 1
        else:
            self.entry_queries += 1

    def row(self, v):
        self._count(True)
        result = []
        for x, w in self.flips:
            u = v ^ x
            value = self.entries.get((w, (v & u).bit_count(), (x & u).bit_count()), 0j)
            if value != 0:
                result.append((u, value))
        self.returned_entries += len(result)
        return result

    def bx(self, x, u):
        self._count(False)
        return self.entries.get((x.bit_count(), ((x ^ u) & u).bit_count(), (x & u).bit_count()), 0j)


def expanded_dictionary(n, coefficients, tol=0):
    active = {key: value for key, value in coefficients.items() if abs(value) > tol}
    return {(x, z): value for x, z, value in common.expand_pauli(n, active)}


def expanded_array(n, coefficients):
    result = np.zeros((1 << n, 1 << n), dtype=complex)
    for x, z, value in common.expand_pauli(n, coefficients):
        result[x, z] = value
    return result


def banded_blocks(blocks, bandwidth):
    """Format the known locality band in CSR, without scanning dense zeros.

    Accuracy checks also inspect the original dense blocks before formatting,
    so this conversion cannot hide a wrong off-band result.
    """
    output = []
    for block in blocks:
        side = block.shape[0]
        offsets = range(-min(bandwidth, side - 1), min(bandwidth, side - 1) + 1)
        converted = sparse.diags([np.diag(block, k) for k in offsets], list(offsets),
                                 shape=block.shape, format="csr", dtype=complex)
        converted.eliminate_zeros()
        output.append(converted)
    return output


def collective_reference(n, coefficients):
    """Independent fixed-locality reference from collective Pauli algebra.

    If S_a is the distinct-word orbit sum, multiplying (sum_i P_mu[i])*S_a
    either adds a Pauli on a new site or contracts with an occupied site.
    Solving for S_(a+e_mu) gives this recurrence. It does not use Chang's
    combinatorial formula or either coordinate transform.
    """
    max_weight = max((w + g0 for (w, g0, _), value in coefficients.items() if value), default=0)
    output = []
    for sector in range(n // 2 + 1):
        N = n - 2 * sector
        side = N + 1
        q = np.arange(N)
        off = np.sqrt((q + 1.0) * (N - q))
        collective = (
            sparse.diags([off, off], [-1, 1], shape=(side, side), format="csr", dtype=complex),
            sparse.diags([1j * off, -1j * off], [-1, 1], shape=(side, side), format="csr"),
            sparse.diags(N - 2 * np.arange(side), format="csr", dtype=complex),
        )
        bank = {(0, 0, 0): sparse.eye(side, format="csr", dtype=complex)}
        for weight in range(1, max_weight + 1):
            for nx in range(weight + 1):
                for ny in range(weight - nx + 1):
                    target = (nx, ny, weight - nx - ny)
                    mu = next(j for j, value in enumerate(target) if value)
                    previous = list(target)
                    previous[mu] -= 1
                    result = collective[mu] @ bank[tuple(previous)]
                    if previous[mu]:
                        contracted = previous.copy()
                        contracted[mu] -= 1
                        result -= (n - weight + 2) * bank[tuple(contracted)]
                    for nu in range(3):
                        if nu == mu or not previous[nu]:
                            continue
                        lam = 3 - mu - nu
                        contracted = previous.copy()
                        contracted[nu] -= 1
                        contracted[lam] += 1
                        sign = 1 if (mu, nu) in ((0, 1), (1, 2), (2, 0)) else -1
                        result -= 1j * sign * (previous[lam] + 1) * bank[tuple(contracted)]
                    bank[target] = result / target[mu]
        result = sparse.csr_matrix((side, side), dtype=complex)
        for key, value in coefficients.items():
            counts = common.pauli_type_from_key(n, key)[1:]
            result += value * bank[counts]
        result.eliminate_zeros()
        output.append(result)
    return output


def array_errors(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"max_abs_error": None, "max_scaled_error": None, "relative_error": None, "valid": False}
    difference = np.abs(actual - expected)
    allowance = atol + rtol * np.abs(expected)
    scaled = np.divide(difference, allowance, out=np.full_like(difference, np.inf), where=allowance > 0)
    scaled[(difference == 0) & (allowance == 0)] = 0
    reference_norm = np.linalg.norm(expected.ravel())
    relative = float(np.linalg.norm(difference.ravel()) / reference_norm) if reference_norm else float(np.linalg.norm(difference.ravel()))
    maximum = float(np.max(difference, initial=0))
    scale = float(np.max(scaled, initial=0))
    return {"max_abs_error": maximum, "max_scaled_error": scale, "relative_error": relative,
            "valid": bool(scale <= 1 and relative <= rtol + atol)}


def mapping_errors(actual, expected, atol, rtol, weights=None):
    keys = sorted(set(actual) | set(expected))
    a = np.array([actual.get(key, 0j) for key in keys])
    b = np.array([expected.get(key, 0j) for key in keys])
    metrics = array_errors(a, b, atol, rtol)
    if weights is not None:
        roots = np.array([sqrt(weights(key)) for key in keys])
        norm = np.linalg.norm(roots * b)
        relative = float(np.linalg.norm(roots * (a - b)) / norm) if norm else float(np.linalg.norm(roots * (a - b)))
        metrics["relative_error"] = relative if np.isfinite(relative) else None
        metrics["valid"] = bool(metrics["max_scaled_error"] is not None and metrics["max_scaled_error"] <= 1
                                and np.isfinite(relative) and relative <= rtol + atol)
    return metrics


def block_errors(n, actual, expected, atol, rtol):
    """Full-operator Hilbert--Schmidt error, using every sector's multiplicity."""
    error_sq = norm_sq = 0.0
    max_abs = max_scaled = 0.0
    valid = len(actual) == len(expected)
    for sector, (a, b) in enumerate(zip(actual, expected)):
        if sparse.issparse(a) or sparse.issparse(b):
            a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
            delta = (a - b).tocoo()
            diff = np.abs(delta.data)
            if delta.nnz:
                sampled = b[delta.row, delta.col]
                if sparse.issparse(sampled):
                    sampled = sampled.toarray()
                reference = np.abs(np.asarray(sampled, dtype=complex).ravel())
            else:
                reference = np.empty(0, dtype=float)
            norm_b = float(np.sum(np.abs(b.data) ** 2))
        else:
            diff = np.abs(a - b).ravel()
            reference = np.abs(b).ravel()
            norm_b = float(np.sum(np.abs(b) ** 2))
        if not np.isfinite(diff).all() or not np.isfinite(reference).all():
            return {"valid": False, "relative_error": None, "max_abs_error": None, "max_scaled_error": None}
        allowance = atol + rtol * reference
        scaled = np.divide(diff, allowance, out=np.full_like(diff, np.inf), where=allowance > 0)
        scaled[(diff == 0) & (allowance == 0)] = 0
        max_abs = max(max_abs, float(np.max(diff, initial=0)))
        max_scaled = max(max_scaled, float(np.max(scaled, initial=0)))
        multiplicity = comb(n, sector) - (comb(n, sector - 1) if sector else 0)
        weight = multiplicity / (1 << n)
        error_sq += weight * float(np.sum(diff ** 2))
        norm_sq += weight * norm_b
    relative = float(np.sqrt(error_sq / norm_sq)) if norm_sq else float(np.sqrt(error_sq))
    return {"relative_error": relative, "max_abs_error": max_abs, "max_scaled_error": max_scaled,
            "valid": bool(valid and max_scaled <= 1 and relative <= atol + rtol)}


def validate_schur(n, actual, coefficients, reference, atol, rtol):
    if reference is not None:
        return block_errors(n, actual, reference, atol, rtol)
    # Large general cases use a recorded round-trip check, not a claimed
    # independent dense reference. Small-n tests establish the basis convention.
    recovered = schur_separated.schur_to_pauli(n, actual)
    return mapping_errors(recovered, coefficients, atol, rtol,
                          weights=lambda key: pauli_orbit_size(n, key))
