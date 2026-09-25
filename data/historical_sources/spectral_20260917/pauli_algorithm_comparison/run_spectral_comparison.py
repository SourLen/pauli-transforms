"""Complete representative eigensystems from general Hermitian Pauli orbits.

Every timed trial starts with fresh conversion tables and returns every block's
eigenvalues and orthonormal eigenbasis. A second, separately timed solve requests
only the lowest pair of each already constructed block; it is a diagnostic of
the solver stage, not a different Hamiltonian or a ground-state shortcut.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.linalg import eigh
from threadpoolctl import threadpool_info, threadpool_limits

from . import application_conversions as conversions
from . import anschuetz_optimized, benchmark_cases, schur_full_ed, schur_separated


METHODS = ("thesis", "anschuetz_optimized", "anschuetz_public_original")


def solve(blocks, *, lowest=False):
    """Same dense Hermitian driver for complete and selected eigenpairs."""
    options = {"subset_by_index": [0, 0]} if lowest else {}
    return [eigh((block + block.conj().T) / 2, driver="evr", **options)
            for block in blocks]


def eigensystem_checks(blocks, eigensystems):
    residual = orthogonality = skew = reconstruction = 0.0
    for block, (values, vectors) in zip(blocks, eigensystems, strict=True):
        if values.shape != (len(block),) or vectors.shape != block.shape:
            raise ValueError("complete block eigenpairs are required")
        scale = max(float(np.linalg.norm(block)), np.finfo(float).tiny)
        residual = max(residual, float(np.linalg.norm(block @ vectors - vectors * values)) / scale)
        orthogonality = max(orthogonality, float(np.linalg.norm(vectors.conj().T @ vectors - np.eye(len(block)))))
        skew = max(skew, float(np.linalg.norm(block - block.conj().T)) / scale)
        reconstruction = max(reconstruction, float(np.linalg.norm((vectors * values) @ vectors.conj().T - block)) / scale)
    return dict(residual=residual, orthogonality=orthogonality,
                hermiticity_error=skew, reconstruction_error=reconstruction)


def reference_checks(n, coefficients, blocks, eigensystems, reference, dense=None, bases=None):
    """Small full-Hilbert-space checks and larger cross-route/round-trip checks."""
    delta = sum(float(np.linalg.norm(a - b) ** 2) for a, b in zip(blocks, reference, strict=True))
    norm = sum(float(np.linalg.norm(b) ** 2) for b in reference)
    eigenvalue_error = max(float(np.max(np.abs(values - np.linalg.eigvalsh((ref + ref.conj().T) / 2))))
                           / max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
                           for (values, _), ref in zip(eigensystems, reference, strict=True))
    recovered = schur_separated.schur_to_pauli(n, blocks, backend="hahn")
    roundtrip = benchmark_cases.mapping_errors(recovered, coefficients, 1e-9, 1e-8,
                    weights=lambda key: benchmark_cases.pauli_orbit_size(n, key))
    metrics = dict(reference_block_error=float(np.sqrt(delta / norm)),
                   reference_eigenvalue_error=eigenvalue_error,
                   pauli_roundtrip_error=roundtrip["relative_error"],
                   dense_spectrum_error=None, dense_eigenvector_residual=None)
    if dense is not None:
        scale = max(float(np.linalg.norm(dense)), np.finfo(float).tiny)
        expanded = np.concatenate([np.repeat(values, comb(n, k) - (comb(n, k - 1) if k else 0))
                                   for k, (values, _) in enumerate(eigensystems)])
        metrics["dense_spectrum_error"] = float(np.linalg.norm(np.sort(expanded) - np.linalg.eigvalsh(dense))) / scale
        metrics["dense_eigenvector_residual"] = max(
            float(np.linalg.norm(dense @ (basis.T @ vectors) - (basis.T @ vectors) * values)) / scale
            for basis, (values, vectors) in zip(bases, eigensystems, strict=True))
    return metrics


def timed_sample(n, method, coefficients, cfg, public_reference=None):
    started = time.perf_counter()
    convert, tables = conversions.prepare(n, method, (coefficients,), cfg, reference=public_reference)
    prepared = time.perf_counter()
    blocks = convert(n, coefficients, tables)
    converted = time.perf_counter()
    eigensystems = solve(blocks)
    finished = time.perf_counter()
    lowest = solve(blocks, lowest=True)
    lowest_finished = time.perf_counter()
    stages = dict(setup_seconds=prepared - started, conversion_seconds=converted - prepared,
                  eigensolve_seconds=finished - converted, total_seconds=finished - started,
                  lowest_eigensolve_seconds=lowest_finished - finished,
                  lowest_total_seconds=(converted - started) + (lowest_finished - finished))
    # This total uses paired stages of this trial. It is not a separately timed
    # end-to-end run, nor subtraction of medians from unrelated measurements.
    return stages, blocks, eigensystems, lowest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 5, 8, 12, 16, 20])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--public-max-n", type=int, default=5)
    parser.add_argument("--anschuetz-source")
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--dense-max-n", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.n) < 1 or min(args.instances, args.repeats) < 1 or args.warmup < 0:
        parser.error("positive sizes, instances and repeats are required")
    out = args.output.resolve()
    if (out / "runs.jsonl").exists():
        parser.error("output already contains measurements; select a new directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    cfg = vars(args).copy()
    cfg.update(output=str(out), schur_backend="hahn", schur_dtype="float64", threads=1,
               input_family="general real Gaussian common coefficients; variance 1/(number of orbits * orbit size)",
               output_contract="Every eigenvalue and orthonormal eigenbasis in every representative Schur block; multiplicities encoded, no full-state expansion",
               timing_contract="Fresh tables + conversion + full scipy.linalg.eigh(evr); imports, input generation, reference, diagnostics and I/O excluded",
               lowest_contract="Separate selected evr solve on the same prepared blocks; paired setup+conversion+selected solve sum, no claimed avoided construction",
               acceptance_tolerance=1e-8)
    conversions.configure_public_source(cfg)
    write_json(out / "config.json", cfg)
    folder = Path(__file__).resolve().parent
    names = ["run_spectral_comparison.py", "application_conversions.py", "benchmark_cases.py", "common.py", "anschuetz_optimized.py", "anschuetz_public.py", "schur_full_ed.py", "schur_separated.py", "separated.py", "schur_hahn.py"]
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in names:
        (snapshot / name).write_bytes((folder / name).read_bytes())
    environment = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                       processor=platform.processor(), cpu_model=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None),
                       source_sha256={name: sha256(folder / name) for name in names},
                       utc_started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    public_reference = conversions.load_public_reference("anschuetz_public_original", cfg) if "anschuetz_public_original" in args.methods else None
    records = []
    with threadpool_limits(limits=1), (out / "runs.jsonl").open("w") as stream:
        environment["threadpools"] = threadpool_info()
        write_json(out / "environment.json", environment)
        for n in args.n:
            for instance in range(args.instances):
                seed = args.seed + 1009 * n + instance
                coefficients = benchmark_cases.make_case(n, "general", seed, locality=0)
                input_file = out / "inputs" / f"n{n}_i{instance}.json"
                write_json(input_file, dict(n=n, instance=instance, seed=seed, coefficients=benchmark_cases.encode_mapping(coefficients)))
                dense = schur_full_ed.dense_operator(n, coefficients) if n <= args.dense_max_n else None
                bases = schur_full_ed.prepare(n) if dense is not None else None
                reference = ([basis.conj() @ dense @ basis.T for basis in bases] if dense is not None
                             else anschuetz_optimized.pauli_to_schur(n, coefficients))
                methods = [method for method in args.methods if method != "anschuetz_public_original" or n <= args.public_max_n]
                # Rotate which method comes first across inputs, while retaining
                # strictly sequential measurements and the same per-method repeats.
                shift = instance % len(methods)
                methods = methods[shift:] + methods[:shift]
                for method in methods:
                    print(f"n={n} instance={instance} method={method}", flush=True)
                    for repeat in range(-args.warmup, args.repeats):
                        stages, blocks, eigensystems, lowest = timed_sample(n, method, coefficients, cfg, public_reference)
                        metrics = eigensystem_checks(blocks, eigensystems)
                        metrics.update(reference_checks(n, coefficients, blocks, eigensystems, reference, dense, bases))
                        metrics["lowest_energy_error"] = max(abs(float(values[0] - selected[0][0])) / max(float(np.linalg.norm(block)), np.finfo(float).tiny)
                            for block, (values, _), selected in zip(blocks, eigensystems, lowest, strict=True))
                        metrics["lowest_residual"] = max(float(np.linalg.norm(block @ vec - vec * val)) / max(float(np.linalg.norm(block)), np.finfo(float).tiny)
                            for block, (val, vec) in zip(blocks, lowest, strict=True))
                        valid = all(np.isfinite(value) and value <= cfg["acceptance_tolerance"] for value in metrics.values() if value is not None)
                        if repeat >= 0:
                            row = dict(n=n, method=method, instance=instance, seed=seed, repeat=repeat,
                                       input_sha256=sha256(input_file), reference_kind="explicit_dense_Pauli" if dense is not None else "optimized_cross_route_and_Hahn_roundtrip",
                                       valid=bool(valid), **stages, **metrics)
                            records.append(row)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                        if not valid:
                            raise ArithmeticError(f"numerical validation failed: n={n}, {method}, {metrics}")
                        if n == 12 and instance == 0 and method == "thesis" and repeat == 0:
                            np.savez(out / "spectrum_example.npz", n=n,
                                **{f"values_{k}": values for k, (values, _) in enumerate(eigensystems)},
                                **{f"vectors_{k}": vectors for k, (_, vectors) in enumerate(eigensystems)})
    with (out / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(out / "validation.json", dict(timed_samples=len(records), all_valid=all(row["valid"] for row in records),
        maxima={key: max(row[key] for row in records if row[key] is not None) for key in metrics},
        note="Small systems use explicit full matrices and lifted representative eigenvectors. Larger checks combine two conversion routes, a Hahn round trip, and original-block residuals; they are not full-space validation."))
    print(f"Saved {len(records)} valid timed samples to {out}", flush=True)


if __name__ == "__main__":
    main()
