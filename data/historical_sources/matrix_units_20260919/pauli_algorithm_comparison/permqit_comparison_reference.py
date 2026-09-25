"""Untimed inputs and independent finite-size checks for the permqit pilot."""

from math import comb, sqrt
import numpy as np
from .common import orbit_size, specht_multiplicities


def keys(n):
    return [(r, s, t) for r in range(n+1) for s in range(n+1)
            for t in range(max(0, r+s-n), min(r, s)+1)]


def random_entries(n, seed, *, hermitian=False):
    """Dense Gaussian orbit coordinates, weighted Hilbert--Schmidt norm one.

    Before normalization each complex orbit coordinate has variance 1/v_rst.
    Hermitian mode correlates transposed entries; diagonal slices are real.
    """
    rng = np.random.default_rng(seed)
    coefficients = {}
    for r, s, t in keys(n):
        if hermitian and r > s:
            coefficients[r, s, t] = coefficients[s, r, t].conjugate()
        else:
            z = complex(rng.normal(), rng.normal()) / sqrt(2)
            if hermitian and r == s:
                z = complex(z.real * sqrt(2))
            coefficients[r, s, t] = z / sqrt(orbit_size(n, r, s, t))
    norm = sqrt(sum(orbit_size(n, *key)*abs(z)**2 for key, z in coefficients.items()))
    return {key: z/norm for key, z in coefficients.items()}


def dense_matrix(n, coefficients):
    """Direct computational-basis expansion; validation only."""
    if n > 8:
        raise ValueError('dense validation is restricted to n <= 8')
    return np.array([[coefficients.get((x.bit_count(), y.bit_count(), (x & y).bit_count()), 0j)
                      for y in range(1 << n)] for x in range(1 << n)])


def _binom(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def integer_kernel_blocks(n, coefficients):
    """Exact integer binomial cancellation, final long-double square roots.

    Independent of both the shared-factor multiplication and Hahn recurrence.
    K[k;r,s,t] = sqrt(C(n-2k,r-k)/C(n-2k,s-k)) times
    sum_l (-1)^(s-t-l) C(n-k-r,l) C(r-k,s-k-l) C(k,s-t-l).
    Not an interval enclosure: only the combinatorial sum is exact.
    """
    blocks = [np.zeros((n-2*k+1, n-2*k+1), dtype=np.clongdouble)
              for k in range(n//2+1)]
    for (r, s, t), value in coefficients.items():
        for k in range(min(r, s, n-r, n-s)+1):
            total = sum((-1 if (s-t-l) % 2 else 1)
                        * _binom(n-k-r, l)*_binom(r-k, s-k-l)*_binom(k, s-t-l)
                        for l in range(s-k+1))
            scale = np.sqrt(np.longdouble(comb(n-2*k, r-k))
                            / np.longdouble(comb(n-2*k, s-k)))
            blocks[k][r-k, s-k] += np.clongdouble(value)*np.longdouble(total)*scale
    return blocks


def block_errors(n, actual, reference, *, atol=1e-9, rtol=1e-8):
    shapes = [(n-2*k+1, n-2*k+1) for k in range(n//2+1)]
    if ([a.shape for a in actual] != shapes or
            [b.shape for b in reference] != shapes):
        raise ValueError('missing Schur blocks or incorrect block shapes')
    mu = specht_multiplicities(n)
    delta2 = sum(m*np.sum(np.abs(a-b)**2) for m, a, b in zip(mu, actual, reference))
    ref2 = sum(m*np.sum(np.abs(b)**2) for m, b in zip(mu, reference))
    relative = float(np.sqrt(delta2/ref2))
    max_abs = max(float(np.max(np.abs(a-b))) for a, b in zip(actual, reference))
    finite = all(np.all(np.isfinite(a)) for a in actual)
    return {'weighted_hs_relative': relative, 'max_abs': max_abs,
            'passed': bool(finite and relative <= rtol and
                           all(np.allclose(a, b, atol=atol, rtol=rtol)
                               for a, b in zip(actual, reference)))}
