"""Separate conversion, eigensolver, and phase-sum accuracy checks.

Run with ``python -m pauli_transforms.application_accuracy --output DIR``.
Small systems use literal tensor products, singlet/Dicke projections, and
full-space scipy.linalg.expm. Larger Ising cases use independent collective-spin
blocks and block expm, not the conversion or phase-sum implementation under test.
All arithmetic here is float64/complex128. This is a finite-grid check.
"""

import argparse
from itertools import product
import json
from math import comb, factorial
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from . import common, physical_models as pm, transforms
from .benchmark_helpers import environment_record, sha256, write_json


# Fixed before the campaign. The relative term retains the existing application's
# 1e-8 threshold. The absolute term handles zero or very small reference blocks.
ATOL = 1e-10
RTOL = 1e-8
PAULIS = (np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]], complex), np.diag([1, -1]).astype(complex))
MEASURED_MODULES = ("application_accuracy.py", "physical_models.py", "transforms.py", "common.py",
                    "krawtchouk.py", "schur_hahn.py", "schur_factorial.py", "benchmark_helpers.py", "__init__.py")


def tensor(factors):
    value = np.ones((1, 1), complex)
    for factor in factors:
        value = np.kron(value, factor)
    return value


def literal_operator(n, coefficients):
    result = np.zeros((2**n, 2**n), complex)
    for word in product(range(4), repeat=n):
        key = (word.count(1) + word.count(2), word.count(3), word.count(2))
        value = coefficients.get(key, 0)
        if value:
            result += value * tensor(PAULIS[j] for j in word)
    return result


def representative_bases(n):
    """Columns are k singlets tensor normalized Dicke states, constructed literally."""
    singlet = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    result = []
    for k in range(n // 2 + 1):
        paired = np.ones(1, complex)
        for _ in range(k):
            paired = np.kron(paired, singlet)
        m = n - 2*k
        columns = []
        for weight in range(m + 1):
            dicke = np.array([int(index.bit_count() == weight) for index in range(2**m)], complex)
            columns.append(np.kron(paired, dicke / np.sqrt(comb(m, weight))))
        result.append(np.column_stack(columns))
    return result


def dense_case(n, family, seed):
    rng = np.random.default_rng(seed)
    if family == "ising_product_x":
        h = {(0, 2, 0): -1/n, (1, 0, 0): -.5/n}
        observable = {(1, 0, 0): 1/n}
        bloch = (.6, 0., 0.)
    elif family == "random_hermitian_product_state":
        def random_operator():
            coefficients = {key: rng.normal() for key in common.all_orbit_keys(n)}
            norm_squared = 0.
            for (w, z, y), value in coefficients.items():
                orbit_size = factorial(n) // (factorial(n-w-z)*factorial(w-y)*factorial(y)*factorial(z))
                norm_squared += 2**n * orbit_size * value**2
            return {key: value / np.sqrt(norm_squared) for key, value in coefficients.items()}
        h, observable = random_operator(), random_operator()
        bloch = (.25, .35, -.2)
    else:
        raise ValueError(f"Unknown family: {family}")
    x, y, z = bloch
    state_coefficients = {(w, nz, ny): 2.**(-n)*x**(w-ny)*y**ny*z**nz
                          for w, nz, ny in common.all_orbit_keys(n)}
    single = (PAULIS[0] + x*PAULIS[1] + y*PAULIS[2] + z*PAULIS[3]) / 2
    dense = (literal_operator(n, h), tensor([single]*n), literal_operator(n, observable))
    bases = representative_bases(n)
    blocks = tuple([basis.conj().T @ matrix @ basis for basis in bases] for matrix in dense)
    return (h, state_coefficients, observable), blocks, dense, dict(bloch=bloch, normalization="unit Hilbert-Schmidt norm for random H,M")


def reference_curve(n, blocks, dense, times):
    if dense is not None:
        h, rho, observable = dense
        return np.array([np.trace(observable @ (u := expm(-1j*t*h)) @ rho @ u.conj().T)
                         for t in times])
    values = np.zeros(len(times), complex)
    for mu, h, rho, observable in zip(common.specht_multiplicities(n), *blocks):
        for index, t in enumerate(times):
            u = expm(-1j*t*h)
            values[index] += mu * np.trace(observable @ u @ rho @ u.conj().T)
    return values


def norm(matrix):
    return float(np.linalg.norm(matrix, 2))


def block_diagnostics(n, actual, reference):
    sectors, weighted_error, weighted_reference = [], 0., 0.
    for k, (mu, block, expected) in enumerate(zip(common.specht_multiplicities(n), actual, reference)):
        error, scale = norm(block-expected), norm(expected)
        nonzero = scale != 0
        sectors.append(dict(k=k, absolute_spectral_error=error, reference_spectral_norm=scale,
                            relative_spectral_error=error/scale if nonzero else None,
                            absolute_tolerance_dominates=RTOL*scale <= ATOL,
                            tolerance=ATOL+RTOL*scale, passed=error <= ATOL+RTOL*scale))
        weighted_error += mu * float(np.vdot(block-expected, block-expected).real)
        weighted_reference += mu * float(np.vdot(expected, expected).real)
    return dict(sectors=sectors, max_block_spectral_error=max(row["absolute_spectral_error"] for row in sectors),
                weighted_hs_error=np.sqrt(weighted_error),
                weighted_relative_hs_error=np.sqrt(weighted_error/weighted_reference) if weighted_reference else None,
                passed=all(row["passed"] for row in sectors))


def evaluate_case(n, backend, inputs, references, dense, times, expected_curve, *, cached=True):
    tables = transforms.prepare(n, backend=backend) if cached else None
    blocks = tuple(transforms.pauli_to_schur(n, mapping, tables, backend=backend) for mapping in inputs)
    conversion = {name: block_diagnostics(n, actual, expected)
                  for name, actual, expected in zip(("hamiltonian", "state", "observable"), blocks, references)}
    # The production solver checks skew before symmetrizing. Preserve both the
    # original skew and the exact size of this correction, in spectral norm.
    skew = [norm(h-h.conj().T) for h in blocks[0]]
    spectrum = pm.diagonalize_blocks(blocks[0])
    spectral = []
    for k, (h, expected, energies, vectors) in enumerate(zip(blocks[0], references[0], spectrum.energies, spectrum.vectors)):
        scale = norm(expected)
        eig_error = float(np.max(np.abs(energies - np.linalg.eigvalsh(expected))))
        residual = norm(h @ vectors - vectors * energies)
        orthogonality = norm(vectors.conj().T @ vectors - np.eye(len(h)))
        spectral.append(dict(k=k, hermiticity_error=skew[k], symmetrization_correction=skew[k]/2,
                             eigensystem_residual=residual, orthogonality_residual=orthogonality,
                             eigenvalue_error=eig_error, tolerance=ATOL+RTOL*scale,
                             passed=max(skew[k], residual, eig_error) <= ATOL+RTOL*scale
                             and orthogonality <= ATOL+RTOL))
    full_spectrum_error = None
    if dense is not None:
        repeated = np.sort(np.concatenate([np.repeat(e, mu) for e, mu in
                                           zip(spectrum.energies, common.specht_multiplicities(n))]))
        full_spectrum_error = float(np.max(np.abs(repeated-np.linalg.eigvalsh(dense[0]))))
    values = pm.expectation_time_series(n, spectrum, blocks[1], blocks[2], times)
    curve_error = float(np.max(np.abs(values-expected_curve)))
    imaginary = float(np.max(np.abs(values.imag)))
    state_trace = sum(mu*np.trace(rho) for mu, rho in zip(common.specht_multiplicities(n), blocks[1]))
    reference_trace = sum(mu*np.trace(rho) for mu, rho in zip(common.specht_multiplicities(n), references[1]))
    min_state_eigenvalue = min(float(np.linalg.eigvalsh((rho+rho.conj().T)/2)[0]) for rho in blocks[1])
    reference_minimum = min(float(np.linalg.eigvalsh((rho+rho.conj().T)/2)[0]) for rho in references[1])
    state_skew = max(norm(rho-rho.conj().T) for rho in blocks[1])
    absolute_summands = 0.
    for mu, basis, rho, observable in zip(common.specht_multiplicities(n), spectrum.vectors, blocks[1], blocks[2]):
        rho_e, m_e = basis.conj().T @ rho @ basis, basis.conj().T @ observable @ basis
        absolute_summands += mu * float(np.sum(np.abs(rho_e * m_e.T)))
    nonzero = np.abs(values) > 0
    cancellation = float(np.max(absolute_summands/np.abs(values[nonzero]))) if np.any(nonzero) else None
    observable_scale = max(norm(m) for m in references[2])
    curve_tolerance = ATOL+RTOL*observable_scale
    state_error = float(abs(state_trace-1))
    state_valid = (state_error <= ATOL+RTOL and state_skew <= ATOL+RTOL
                   and min_state_eigenvalue >= -ATOL and abs(reference_trace-1) <= ATOL+RTOL
                   and reference_minimum >= -ATOL)
    passed = (all(value["passed"] for value in conversion.values()) and all(row["passed"] for row in spectral)
              and curve_error <= curve_tolerance and imaginary <= curve_tolerance and state_valid
              and (full_spectrum_error is None or full_spectrum_error <= ATOL+RTOL*norm(dense[0])))
    return dict(conversion=conversion, spectral=spectral, full_spectrum_error=full_spectrum_error,
                max_expectation_error=curve_error, max_imaginary_residual=imaginary,
                curve_tolerance=curve_tolerance, state_trace_real=float(state_trace.real),
                state_trace_imag=float(state_trace.imag), state_trace_error=state_error,
                state_hermiticity_error=state_skew, state_minimum_eigenvalue=min_state_eigenvalue,
                state_symmetric_part_correction=state_skew/2,
                reference_trace_error=float(abs(reference_trace-1)), reference_minimum_eigenvalue=reference_minimum,
                phase_sum_absolute_summands=absolute_summands, max_cancellation_ratio=cancellation,
                exact_zero_expectation_count=int(np.sum(~nonzero)), passed=bool(passed),
                curves=[dict(time=float(t), real=float(v.real), imag=float(v.imag),
                             reference_real=float(r.real), reference_imag=float(r.imag), absolute_error=float(abs(v-r)))
                        for t, v, r in zip(times, values, expected_curve)])


def summary(rows):
    result = []
    for family, backend in sorted({(row["family"], row["backend"]) for row in rows}):
        selected = [row for row in rows if (row["family"], row["backend"]) == (family, backend)]
        completed = [row for row in selected if "conversion" in row]
        result.append(dict(family=family, backend=backend, sizes=sorted({row["n"] for row in selected}),
                           cases=len(selected), failures=sum(not row["passed"] for row in selected),
                           max_h_block_error=max((row["conversion"]["hamiltonian"]["max_block_spectral_error"] for row in completed), default=None),
                           max_eigensystem_residual=max((s["eigensystem_residual"] for row in completed for s in row["spectral"]), default=None),
                           max_eigenvalue_error=max((s["eigenvalue_error"] for row in completed for s in row["spectral"]), default=None),
                           max_expectation_error=max((row["max_expectation_error"] for row in completed), default=None),
                           max_imaginary_residual=max((row["max_imaginary_residual"] for row in completed), default=None)))
    return result


def write_tex_summary(path, rows):
    def scientific(value):
        if value is None:
            return "---"
        mantissa, exponent = f"{value:.1e}".split("e")
        return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
    families = {"ising_collective_reference": "Collective Ising", "ising_product_x": "Dense Ising",
                "random_hermitian_product_state": "Dense random"}
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{llcrrr}", r"\toprule",
             r"Reference/input & Method & $n$ & Block error & Residual & Dynamics error \\", r"\midrule"]
    for row in summary(rows):
        sizes = row["sizes"]
        size_text = f"{sizes[0]}--{sizes[-1]}" if sizes == list(range(sizes[0], sizes[-1]+1)) else ",".join(map(str, sizes))
        lines.append(" & ".join([families[row["family"]], "Hahn" if row["backend"] == "hahn" else "Shared factor",
                                 size_text, scientific(row["max_h_block_error"]),
                                 scientific(row["max_eigensystem_residual"]), scientific(row["max_expectation_error"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Maximum absolute Hamiltonian block error and eigensystem residual in spectral norm, and expectation error over 241 equally spaced times in $[0,12]$. References and computations use double precision. Dense tests include cached and uncached conversion.}",
              r"\label{tab:application_accuracy}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-max", type=int, default=6)
    parser.add_argument("--ising-sizes", type=int, nargs="+", default=[8, 12, 20, 40])
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--tmax", type=float, default=12.)
    parser.add_argument("--seed", type=int, default=250926)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.dense_max, args.ising_sizes, args.points = 3, [8], 5
    if args.dense_max < 2 or args.dense_max > 6 or min(args.ising_sizes) < 2 or args.points < 1 or not 0 < args.tmax < np.inf:
        parser.error("Require 2<=dense-max<=6, Ising n>=2, positive finite tmax and points")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = [f"pauli_transforms/{name}" for name in MEASURED_MODULES]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=repo, text=True).splitlines()
    hashes = {path: sha256(repo/path) for path in paths}
    write_json(out / "source_hashes.json", hashes)
    write_json(out / "config.json", dict(vars(args), output=args.output.name, atol=ATOL, rtol=RTOL,
               relative_error_rule="absolute only for zero reference; flag when absolute tolerance dominates",
               dtype="complex128", reference_precision="float64",
               source_revision=revision, source_worktree_changes=dirty, source_sha256=hashes))
    rows = []
    times = np.linspace(0, args.tmax, args.points)
    cases = [(n, family) for n in range(2, args.dense_max+1)
             for family in ("ising_product_x", "random_hermitian_product_state")]
    cases += [(n, "ising_collective_reference") for n in args.ising_sizes]
    with threadpool_limits(limits=1):
        environment = environment_record()
        environment["command"] = ["python", "-m", "pauli_transforms.application_accuracy", "--output", args.output.name,
                                  "--dense-max", str(args.dense_max), "--ising-sizes", *map(str, args.ising_sizes),
                                  "--points", str(args.points), "--tmax", str(args.tmax), "--seed", str(args.seed)]
        for pool in environment["threadpools"]:
            pool["filepath"] = Path(pool["filepath"]).name
        write_json(out / "environment.json", environment)
        for n, family in cases:
            seed = args.seed+n
            if family == "ising_collective_reference":
                inputs = (pm.ising_pauli(n), pm.product_x_pauli(n, .6), {(1, 0, 0): 1/n})
                references = (pm.ising_collective_blocks(n), pm.product_x_collective_blocks(n, .6),
                              pm.collective_observable_blocks(n, "x"))
                dense, generation = None, dict(g=1., h=.5, p=.6)
            else:
                inputs, references, dense, generation = dense_case(n, family, seed)
            expected = reference_curve(n, references, dense, times)
            for backend in ("hahn", "factorial"):
                for cached in (True, False) if n <= args.dense_max else (True,):
                    print(f"{family} n={n} backend={backend} cached={cached}", flush=True)
                    row = dict(n=n, family=family, backend=backend, cached=cached, seed=seed,
                               source_revision=revision, direction="Pauli to Schur, eigensystem, dynamics",
                               dtype="complex128", reference="literal dense tensor + expm" if dense is not None else "collective spin + block expm",
                               reference_precision="float64", generation=generation)
                    try:
                        row.update(evaluate_case(n, backend, inputs, references, dense, times, expected, cached=cached))
                    except Exception as exc:
                        row.update(passed=False, error_type=type(exc).__name__, error=str(exc))
                    rows.append(row)
                    write_json(out / "results.json", rows)
    write_json(out / "summary.json", summary(rows))
    if args.points == 241 and args.tmax == 12.:
        write_tex_summary(out / "summary.tex", rows)
    print(json.dumps(summary(rows), indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
