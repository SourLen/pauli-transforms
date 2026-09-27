"""Matched spectral and Ising comparisons requested on 26 September 2026.

Run with the scientific Python environment documented in README.md. Numerical
kernels are an unchanged snapshot of the public v0.2.0 package. New code and
benchmark design were prepared with Codex assistance. Timings are never pooled
with older campaigns. Each output directory must be new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pauli-baselines-matplotlib")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_snapshot"))

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import qutip
from qutip import piqs

from pauli_transforms import application_conversions as conversions
from pauli_transforms import anschuetz_optimized, benchmark_cases, dense_reference
from pauli_transforms import physical_models as pm, transforms
from pauli_transforms.benchmark_helpers import environment_record, sha256, write_json, write_csv
from pauli_transforms.common import specht_multiplicities
from pauli_transforms.run_spectral_comparison import solve, eigensystem_checks


SPECTRAL_SIZES = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
ISING_SIZES = [2, 3, 4, 5, 8, 12, 16, 20, 30, 40]
TOLERANCE = 1e-8


def dense_hamiltonian(n, coefficients):
    """Assemble H directly in computational coordinates with Walsh transforms.

    P(x,z)|b> = i^|x&z| (-1)^(z.b) |b xor x>. For each x, the
    Walsh transform of c(x,z)*i^|x&z| gives every H[b xor x,b].
    This uses O(n*4**n) arithmetic and O(4**n) storage, avoids summing
    4**n separate dense Pauli matrices, and performs no Schur reduction.
    The transform is unnormalized, matching orbit sums of distinct words.
    """
    size = 1 << n
    masks = np.arange(size, dtype=np.uint64)
    x, z = masks[:, None], masks[None, :]
    ny = np.bitwise_count(x & z)
    nz = np.bitwise_count((~x) & z)
    w = np.bitwise_count(x)
    lookup = np.zeros((n + 1, n + 1, n + 1), dtype=complex)
    for key, value in coefficients.items():
        lookup[key] = value
    rows = lookup[w, nz, ny] * np.array([1, 1j, -1, -1j])[ny % 4]
    for width in (1 << bit for bit in range(n)):
        pieces = rows.reshape(size, -1, 2, width)
        left = pieces[:, :, 0, :].copy()
        right = pieces[:, :, 1, :].copy()
        pieces[:, :, 0, :] = left + right
        pieces[:, :, 1, :] = left - right
    result = np.empty_like(rows)
    result[x ^ z, z] = rows
    return result


def piqs_ising_blocks(n, g=1., h=.5, p=.6):
    """Native PIQS collective operators plus explicit model/state formulas.

    Each returned rho block is unweighted, as required by the common spectral
    routine, which inserts Specht multiplicities. No general orbit recurrence
    or thesis transform is used. All construction is inside the sample timer.
    """
    # Request only the two needed operators, then extract all spin sectors.
    x_blocks = piqs.dicke_blocks(piqs.jspin(n, "x"))
    z_blocks = piqs.dicke_blocks(piqs.jspin(n, "z"))
    hamiltonian, state, observable = [], [], []
    for k, (jx, jz) in enumerate(zip(x_blocks, z_blocks, strict=True)):
        jx, jz = np.asarray(jx), np.asarray(jz)
        hamiltonian.append(-2 * g / n * (jz @ jz) - 2 * h / n * jx
                           + g / 2 * np.eye(len(jx)))
        _, vectors = np.linalg.eigh(jx)
        exponent = k + np.arange(len(jx))
        weights = ((1 + p) / 2) ** exponent * ((1 - p) / 2) ** (n - exponent)
        state.append((vectors * weights) @ vectors.conj().T)
        observable.append(2 / n * jx)
    return hamiltonian, state, observable


def spectral_sample(n, method, coefficients):
    start = time.perf_counter()
    if method == "full_space":
        matrix = dense_hamiltonian(n, coefficients)
        prepared = converted = time.perf_counter()
        systems = solve([matrix])
        blocks = [matrix]
    else:
        convert, tables = conversions.prepare(n, method, (coefficients,), {})
        prepared = time.perf_counter()
        blocks = convert(n, coefficients, tables)
        converted = time.perf_counter()
        systems = solve(blocks)
    finished = time.perf_counter()
    phases = dict(prepare_seconds=converted - start,
                  eigensolve_seconds=finished - converted,
                  total_seconds=finished - start)
    if method != "full_space":
        phases.update(tables_seconds=prepared - start,
                      conversion_seconds=converted - prepared)
    return blocks, systems, phases


def ising_sample(n, method, times):
    start = time.perf_counter()
    if method == "thesis":
        inputs = (pm.ising_pauli(n), pm.product_x_pauli(n, .6), {(1, 0, 0): 1 / n})
        tables = transforms.prepare(n, backend="hahn")
        blocks = tuple(transforms.pauli_to_schur(n, mapping, tables) for mapping in inputs)
    elif method == "piqs_blocks":
        blocks = piqs_ising_blocks(n)
    else:
        raise ValueError(method)
    prepared = time.perf_counter()
    spectrum = pm.diagonalize_blocks(blocks[0])
    diagonalized = time.perf_counter()
    curve = pm.expectation_time_series(n, spectrum, blocks[1], blocks[2], times)
    finished = time.perf_counter()
    return blocks, spectrum, curve, dict(prepare_seconds=prepared - start,
        eigensolve_seconds=diagonalized - prepared,
        curve_seconds=finished - diagonalized, total_seconds=finished - start)


def relative_blocks(actual, expected, weights=None):
    if weights is None:
        weights = [1] * len(expected)
    numerator = sum(float(w) * np.linalg.norm(a - b) ** 2
                    for w, a, b in zip(weights, actual, expected, strict=True))
    denominator = sum(float(w) * np.linalg.norm(b) ** 2
                      for w, b in zip(weights, expected, strict=True))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def spectral_checks(n, method, coefficients, blocks, systems, references, dense):
    checks = eigensystem_checks(blocks, systems)
    expected_energies = [eigh(block, eigvals_only=True, driver="evr") for block in references]
    if method == "full_space":
        expanded = np.sort(np.concatenate([np.repeat(e, mu) for e, mu in
                          zip(expected_energies, specht_multiplicities(n), strict=True)]))
        checks["spectrum_relative_error"] = float(np.linalg.norm(systems[0][0] - expanded)
                                                  / np.linalg.norm(expanded))
        if dense is not None:
            checks["literal_matrix_relative_error"] = relative_blocks(blocks, [dense])
    else:
        checks["block_relative_error"] = relative_blocks(blocks, references)
        checks["spectrum_relative_error"] = max(float(np.linalg.norm(e - ref)) /
            max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
            for (e, _), ref in zip(systems, expected_energies, strict=True))
        if dense is not None:
            bases = dense_reference.prepare(n)
            checks["literal_block_relative_error"] = relative_blocks(
                blocks, [basis.conj() @ dense @ basis.T for basis in bases])
    return checks


def ising_checks(n, blocks, spectrum, curve, references, expected):
    weights = specht_multiplicities(n)
    checks = {name + "_relative_error": relative_blocks(a, b, weights)
              for name, a, b in zip(("h", "state", "observable"), blocks, references, strict=True)}
    checks["curve_max_abs_error"] = float(np.max(np.abs(curve - expected)))
    checks["trace_error"] = float(abs(pm.block_trace(n, blocks[1]) - 1))
    checks["initial_expectation_error"] = float(abs(curve[0] - .6))
    checks["negative_state_mass"] = sum(float(mu) * float(np.maximum(
        -np.linalg.eigvalsh((rho + rho.conj().T) / 2), 0).sum())
        for mu, rho in zip(weights, blocks[1], strict=True))
    checks.update(eigensystem_checks(blocks[0], list(zip(spectrum.energies, spectrum.vectors))))
    return checks


def code_hashes():
    paths = [Path(__file__), *sorted((ROOT / "source_snapshot").rglob("*.py"))]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("spectral", "ising"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--dense-max-n", type=int, default=10)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    sizes = args.sizes or (SPECTRAL_SIZES if args.kind == "spectral" else ISING_SIZES)
    repeats = args.repeats or (5 if args.kind == "spectral" else 7)
    instances = args.instances if args.kind == "spectral" else 1
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir()
    cfg = dict(kind=args.kind, sizes=sizes, repeats=repeats, instances=instances,
               warmups=args.warmups, dense_max_n=args.dense_max_n, seed=20260926,
               acceptance_tolerance=TOLERANCE, threads=1,
               spectral_solver="scipy.linalg.eigh(driver='evr'), all eigenpairs",
               ising_solver="common blockwise numpy.linalg.eigh and spectral phase sum",
               ising_parameters=dict(g=1., h=.5, p=.6, times=np.linspace(0, 12, 241).tolist()),
               timing="Fresh preparation plus complete requested result; validation, imports and I/O excluded",
               full_space_output="Full computational-basis eigenvectors; Schur routes return compact representative eigenvectors",
               piqs_role="Native collective operators, model-specific state formula, common block spectral solver; no mesolve")
    hashes = code_hashes()
    write_json(output / "config.json", cfg)
    write_json(output / "source_hashes.json", hashes)
    shutil.copy2(__file__, output / "run_comparisons.py")
    records, curve_rows, job_order = [], [], []
    rng = random.Random(cfg["seed"])
    with threadpool_limits(limits=1), (output / "runs.jsonl").open("w") as stream:
        write_json(output / "environment.json", dict(environment_record(), qutip=qutip.__version__))
        for n in sizes:
            for instance in range(instances):
                if args.kind == "spectral":
                    path = output / "inputs" / f"n{n}_i{instance}.json"
                    archived = ROOT / "archived_inputs" / path.name
                    if archived.exists():
                        shutil.copy2(archived, path)
                        payload = json.loads(path.read_text())
                        coefficients = benchmark_cases.decode_mapping(payload["coefficients"])
                    else:
                        seed = 20260917 + 1009 * n + instance
                        coefficients = benchmark_cases.make_case(n, "general", seed, locality=0)
                        write_json(path, dict(n=n, instance=instance, seed=seed,
                            coefficients=benchmark_cases.encode_mapping(coefficients)))
                    # Independent conversion, with literal full-space checks through n=5.
                    references = anschuetz_optimized.pauli_to_schur(n, coefficients)
                    dense = dense_reference.dense_operator(n, coefficients) if n <= 5 else None
                    methods = ["thesis", "anschuetz_optimized"]
                    if n <= args.dense_max_n:
                        methods.append("full_space")
                else:
                    times = np.asarray(cfg["ising_parameters"]["times"])
                    references = (pm.ising_collective_blocks(n), pm.product_x_collective_blocks(n, .6),
                                  pm.collective_observable_blocks(n, "x"))
                    expected = pm.expectation_time_series(n, pm.diagonalize_blocks(references[0]),
                                                        references[1], references[2], times)
                    methods = ["thesis", "piqs_blocks"]
                for repeat in range(-args.warmups, repeats):
                    order = methods.copy()
                    rng.shuffle(order)
                    for method in order:
                        if args.kind == "spectral":
                            blocks, systems, phases = spectral_sample(n, method, coefficients)
                            checks = spectral_checks(n, method, coefficients, blocks, systems, references, dense)
                        else:
                            blocks, spectrum, curve, phases = ising_sample(n, method, times)
                            checks = ising_checks(n, blocks, spectrum, curve, references, expected)
                            if repeat == repeats - 1:
                                curve_rows.extend(dict(n=n, method=method, time=float(t),
                                    value_real=float(v.real), value_imag=float(v.imag),
                                    reference_real=float(r.real), reference_imag=float(r.imag))
                                    for t, v, r in zip(times, curve, expected, strict=True))
                        valid = all(np.isfinite(value) and value <= TOLERANCE for value in checks.values())
                        identity = dict(n=n, instance=instance, repeat=repeat, method=method)
                        job_order.append(identity)
                        if repeat >= 0 or not valid:
                            row = dict(**identity, valid=bool(valid), status="ok" if valid else "failed",
                                       **phases, checks=checks)
                            if args.kind == "spectral":
                                row["input_sha256"] = sha256(path)
                            records.append(row)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                        if not valid:
                            raise ArithmeticError(f"Validation failed: {identity}: {checks}")
                print(f"{args.kind}: n={n}, input={instance}, all {len(methods) * repeats} trials valid", flush=True)
                # Do not retain large eigenvector arrays between sizes.
                del blocks
    write_json(output / "job_order.json", job_order)
    write_csv(output / "runs.csv", [{k: v for k, v in row.items() if k != "checks"} for row in records])
    if curve_rows:
        write_csv(output / "curves.csv", curve_rows)
    maxima = {key: max(row["checks"].get(key, 0.) for row in records)
              for key in sorted({key for row in records for key in row["checks"]})}
    unchanged = code_hashes() == hashes
    write_json(output / "validation.json", dict(samples=len(records), all_valid=all(r["valid"] for r in records),
               source_unchanged=unchanged, maxima=maxima))
    if not unchanged:
        raise RuntimeError("Source changed during measurements")


if __name__ == "__main__":
    main()
