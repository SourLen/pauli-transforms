"""Independent collective-spin and QuTiP baselines for Ising dynamics.

The common task is ``H = -(g sum_{i<j} Zi Zj + h sum_i Xi)/n``,
``rho0 = [(I+pX)/2]**tensor(n)``, and ``M = sum_i Xi/n``.  Both Hamiltonian
terms carry the factor ``1/n``.  The mixed state occupies every spin sector.

``prepare_direct`` returns representative Schur blocks, whose physical traces
require Specht multiplicities.  QuTiP's Dicke representation instead stores
``mu_j * rho_j`` in each state block.  Its ordinary trace and expectation value
therefore already include multiplicity; no additional weights belong in the
solver output.  No coordinate transform is used by either baseline.

QuTiP is an optional dependency, imported only by its adapters.  The adapters
were checked with QuTiP 5.3.1.  They specify a reproducible solver configuration,
not a claim that this configuration is optimal for every input size.
"""

from __future__ import annotations

import numpy as np

from .common import specht_multiplicities
from .physical_models import spin_matrices


def _parameters(n, g, h, p):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    values = []
    for value, name in ((g, "g"), (h, "h"), (p, "p")):
        if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and real")
        values.append(float(value))
    if not -1 < values[2] < 1:
        raise ValueError("this mixed-state baseline requires -1 < p < 1")
    return int(n), *values


def _product_x_block(n, k, sx, p):
    """Raw product-state block, using stable log probabilities.

    With ``beta=atanh(p)``, the block is
    ``exp(2 beta Jx)/(2 cosh(beta))**n``.  In a Jx eigenbasis its eigenvalues
    are ``qplus**(n/2+m) * qminus**(n/2-m)``.  Evaluating their logarithms
    avoids forming an exponentially large matrix exponential or denominator.
    """
    side = len(sx)
    if p == 0:
        return np.eye(side, dtype=complex) * np.exp(-n * np.log(2.0))
    _, vectors = np.linalg.eigh(sx)
    # eigh orders eigenvectors by increasing m; use the exact half-integer
    # eigenvalues rather than their numerically computed approximations.
    plus_count = k + np.arange(side, dtype=float)
    minus_count = n - plus_count
    log_plus = np.log1p(p) - np.log(2.0)
    log_minus = np.log1p(-p) - np.log(2.0)
    weights = np.exp(plus_count * log_plus + minus_count * log_minus)
    return (vectors * weights[None, :]) @ vectors.conj().T


def prepare_direct(n, g=1.0, h=0.5, p=0.6):
    """Return ``(H_blocks, rho_blocks, M_blocks)`` in descending spin order.

    This model-specific shortcut constructs collective-spin matrices directly.
    It does not call the general Pauli-to-Schur conversion algorithm.  States
    are raw representative blocks, not separately normalized sector states.
    """
    n, g, h, p = _parameters(n, g, h, p)
    h_blocks, rho_blocks, m_blocks = [], [], []
    for k in range(n // 2 + 1):
        sx, _, sz = spin_matrices(n, k)
        h_blocks.append(-2 * g / n * (sz @ sz) - 2 * h / n * sx
                        + g / 2 * np.eye(len(sx)))
        rho_blocks.append(_product_x_block(n, k, sx, p))
        m_blocks.append(2 / n * sx)
    return h_blocks, rho_blocks, m_blocks


def prepare_qutip(n, g=1.0, h=0.5, p=0.6, *, representation="piqs"):
    """Return QuTiP ``(H, rho0, M)`` in ``'piqs'`` or ``'full'`` space.

    PIQS supplies collective operators in the all-sector Dicke representation.
    The state is assembled independently from the product-state block formula,
    with multiplicities absorbed into its blocks.  The full-space alternative
    constructs collective operators and the tensor-product state on n qubits.
    """
    n, g, h, p = _parameters(n, g, h, p)
    if representation not in ("piqs", "full"):
        raise ValueError("representation must be 'piqs' or 'full'")
    import qutip
    from qutip import piqs

    if representation == "piqs":
        from scipy.sparse import block_diag

        sx, _, sz = piqs.jspin(n)
        state_blocks = [
            float(mu) * _product_x_block(n, k, spin_matrices(n, k)[0], p)
            for k, mu in enumerate(specht_multiplicities(n))
        ]
        state = qutip.Qobj(block_diag(state_blocks, format="csr"), isherm=True)
        identity = qutip.qeye(sx.shape[0])
    else:
        sx, _, sz = piqs.jspin(n, basis="uncoupled")
        one_site = (qutip.qeye(2) + p * qutip.sigmax()) / 2
        state = qutip.tensor([one_site] * n)
        identity = qutip.qeye([2] * n)
    hamiltonian = -2 * g / n * (sz @ sz) - 2 * h / n * sx + g / 2 * identity
    return hamiltonian, state, 2 / n * sx


def solve_qutip(prepared, times, *, method="vern9", atol=1e-10, rtol=1e-9,
                matrix_form=False):
    """Return the magnetization curve from closed-system ``qutip.mesolve``.

    ``times`` are absolute nonnegative times from the prepared initial state.
    No density-matrix trajectory is stored and no output renormalization is
    applied.  By default QuTiP evolves its sparse Liouvillian; ``matrix_form``
    requests its direct matrix-commutator option (available in QuTiP 5.3).
    Solver construction is deliberately inside this call's timing boundary.
    """
    import qutip

    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite, nonnegative, and strictly increasing")
    if len(times) == 0:
        return np.empty(0, dtype=complex)
    hamiltonian, state, observable = prepared
    prepend_zero = times[0] != 0
    solver_times = np.r_[0.0, times] if prepend_zero else times
    if len(solver_times) == 1:
        return np.asarray([qutip.expect(observable, state)], dtype=complex)
    options = {
        "method": method,
        "atol": atol,
        "rtol": rtol,
        "nsteps": 100000,
        "store_states": False,
        "store_final_state": False,
        "normalize_output": False,
        "progress_bar": "",
    }
    if matrix_form:
        if "matrix_form" not in qutip.MESolver.solver_options:
            raise ValueError("matrix_form requires a QuTiP version supporting that option")
        options["matrix_form"] = True
    result = qutip.mesolve(hamiltonian, state, solver_times, c_ops=[],
                          e_ops=[observable], options=options)
    curve = np.asarray(result.expect[0], dtype=complex)
    return curve[1:] if prepend_zero else curve
