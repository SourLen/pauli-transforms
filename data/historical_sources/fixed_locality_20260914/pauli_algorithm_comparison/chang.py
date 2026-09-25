"""Chang--Larocca--Cerezo fixed-weight Schur construction (arXiv:2603.13072v1).

Algorithm 1 / Appendix B, with the Dicke normalization from the theorem:
sqrt(C(N,q)/C(N,q')), correcting the reciprocal in algorithm line 20.
Output uses CSR blocks, so fixed-locality inputs never allocate dense zeros.
Input coefficients multiply orbit SUMS, as in the other local modules.
This is our implementation of the published formula, not authors' software.
"""

from math import comb, sqrt
import operator

from scipy.sparse import csr_matrix

from .common import pauli_type_from_key


def prepare(n, max_weight):
    """Cache only binomial coefficients, not Schur blocks or answers.

    For fixed maximum weight this takes O(n) arithmetic and storage.
    tables[r][m] = C(m,r); the zero convention handles impossible choices.
    """
    n, max_weight = operator.index(n), operator.index(max_weight)
    if n < 0 or not 0 <= max_weight <= n:
        raise ValueError("require n >= 0 and 0 <= max_weight <= n")
    return tuple(tuple(comb(m, r) if r <= m else 0 for m in range(n + 1))
                 for r in range(max_weight + 1))


def _multinomial(total, parts, tables):
    value = 1
    for count in parts:
        if count > total:
            return 0
        value *= tables[count][total]
        total -= count
    return value


def _dicke_ratio(N, q, target):
    """sqrt(C(N,q)/C(N,target)), using at most the Pauli weight many factors."""
    ratio = 1.0
    for j in range(q, target):
        ratio *= (j + 1) / (N - j)
    for j in range(target, q):
        ratio *= (N - j) / (j + 1)
    return sqrt(ratio)


def _column(n, sector, q, counts, tables):
    """Nonzero (output-weight, value) entries for one orbit-sum column."""
    nx, ny, nz = counts
    N = n - 2 * sector
    raw = {}
    for ax in range(min(sector, nx // 2) + 1):
        for ay in range(min(sector - ax, ny // 2) + 1):
            for az in range(min(sector - ax - ay, nz // 2) + 1):
                pairs = _multinomial(sector, (ax, ay, az), tables)
                rx, ry, rz = nx - 2 * ax, ny - 2 * ay, nz - 2 * az
                if rx + ry + rz > N:
                    continue
                for sx in range(min(q, rx) + 1):
                    for sy in range(min(q - sx, ry) + 1):
                        for sz in range(min(q - sx - sy, rz) + 1):
                            zeros = _multinomial(N - q, (rx - sx, ry - sy, rz - sz), tables)
                            if not zeros:
                                continue
                            ones = _multinomial(q, (sx, sy, sz), tables)
                            target = q + rx + ry - 2 * (sx + sy)
                            sign = -1 if (ax + az + sy + sz) & 1 else 1
                            # Integer accumulation preserves cancellations before
                            # applying the floating Dicke normalization.
                            raw[target] = raw.get(target, 0) + sign * pairs * ones * zeros
    phase = (1, 1j, -1, -1j)[ny & 3]
    return [(target, phase * value * _dicke_ratio(N, q, target))
            for target, value in raw.items() if value]


def pauli_to_schur(n, coefficients, tables=None):
    """Return every representative Schur block as a SciPy CSR matrix.

    keys are (w,g0,g1), i.e. (nX,nY,nZ)=(w-g1,g1,g0).
    A coefficient is per individual Pauli word. For an orbit AVERAGE, pass
    the desired amplitude divided by that orbit's number of distinct words.
    With a fixed set of fixed-weight keys, work and output storage are O(n^2)
    in the constant-time arithmetic model. Growing weights are not covered.
    """
    n = operator.index(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    terms = [(pauli_type_from_key(n, key)[1:], value)
             for key, value in coefficients.items() if value]
    weight = max((sum(counts) for counts, _ in terms), default=0)
    if tables is None:
        tables = prepare(n, weight)
    if len(tables) <= weight or any(len(row) != n + 1 for row in tables):
        raise ValueError("binomial bank does not cover this n and Pauli weight")
    blocks = []
    for sector in range(n // 2 + 1):
        side = n - 2 * sector + 1
        rows, columns, values = [], [], []
        for counts, coefficient in terms:
            for q in range(side):
                for target, value in _column(n, sector, q, counts, tables):
                    rows.append(target)
                    columns.append(q)
                    values.append(coefficient * value)
        block = csr_matrix((values, (rows, columns)), shape=(side, side), dtype=complex)
        block.sum_duplicates()
        block.eliminate_zeros()
        blocks.append(block)
    return blocks
