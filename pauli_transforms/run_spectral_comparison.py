"""Complete representative eigensystems from general Hermitian Pauli orbits.

Every trial builds fresh conversion tables and returns each block's complete
eigenvalues and orthonormal eigenbasis. Validation is outside the timers.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from .benchmark_helpers import environment_record, source_snapshot, sha256, write_json, write_csv
from . import application_conversions as conversions
from . import anschuetz_optimized, benchmark_cases, dense_reference, transforms


METHODS = ("thesis", "anschuetz_optimized", "anschuetz_public_original")


def solve(blocks):
    """Complete block eigensystems with the common SciPy evr driver."""
    return [eigh((block + block.conj().T) / 2, driver="evr") for block in blocks]


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
    recovered = transforms.schur_to_pauli(n, blocks, backend="hahn")
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
    stages = dict(setup_seconds=prepared - started, conversion_seconds=converted - prepared,
                  eigensolve_seconds=finished - converted, total_seconds=finished - started)
    return stages, blocks, eigensystems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 5, 8, 12, 16, 20])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(conversions.DEFAULT_CONVERSION_METHODS))
    parser.add_argument("--public-max-n", type=int, default=5)
    parser.add_argument("--anschuetz-source")
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--dense-max-n", type=int, default=5)
    parser.add_argument("--input-directory", type=Path,
                        help="Read exact saved JSON inputs from a prior campaign or its inputs directory")
    parser.add_argument("--smoke", action="store_true", help="Two small sizes, one input and one repetition")
    args = parser.parse_args(argv)
    if args.smoke:
        args.n, args.instances, args.repeats, args.warmup = [2, 3], 1, 1, 0
    if (min(args.n) < 1 or min(args.instances, args.repeats) < 1 or args.warmup < 0
            or min(args.seed, args.dense_max_n, args.public_max_n) < 0):
        parser.error("Positive sizes/repetitions and nonnegative warmup, seed and cutoffs required")
    sizes = [n for n in dict.fromkeys(args.n)
             if any(method != conversions.PUBLIC_METHOD or n <= args.public_max_n for method in args.methods)]
    if not sizes:
        parser.error("No methods are available at the requested sizes")
    example_n = 12 if 12 in sizes else max(sizes)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output already contains measurements; select a new directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    cfg = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    cfg.update(output=str(out), schur_backend="hahn", schur_dtype="float64", threads=1,
               coefficient_variance="1/(number of orbits * orbit size)",
               eigensolver="scipy.linalg.eigh, driver=evr", fresh_tables=True,
               acceptance_tolerance=1e-8)
    conversions.configure_public_source(cfg)
    write_json(out / "config.json", cfg)
    hashes = source_snapshot(out)
    public_reference = conversions.load_public_reference("anschuetz_public_original", cfg) if "anschuetz_public_original" in args.methods else None
    records = []
    with threadpool_limits(limits=1), (out / "runs.jsonl").open("w") as stream:
        write_json(out / "environment.json", dict(environment_record(), source_sha256=hashes))
        for n in sizes:
            available = [method for method in dict.fromkeys(args.methods)
                         if method != conversions.PUBLIC_METHOD or n <= args.public_max_n]
            if not available:
                continue
            for instance in range(args.instances):
                seed = args.seed + 1009 * n + instance
                input_file = out / "inputs" / f"n{n}_i{instance}.json"
                if args.input_directory:
                    folder = args.input_directory
                    folder = folder / "inputs" if (folder / "inputs").is_dir() else folder
                    archived = folder / input_file.name
                    payload = json.loads(archived.read_text())
                    if payload["n"] != n or payload["instance"] != instance:
                        raise ValueError("Archived input identity does not match its filename")
                    seed = payload["seed"]
                    coefficients = benchmark_cases.decode_mapping(payload["coefficients"])
                    input_file.write_bytes(archived.read_bytes())
                else:
                    coefficients = benchmark_cases.make_case(n, "general", seed, locality=0)
                    write_json(input_file, dict(n=n, instance=instance, seed=seed,
                        coefficients=benchmark_cases.encode_mapping(coefficients)))
                dense = dense_reference.dense_operator(n, coefficients) if n <= args.dense_max_n else None
                bases = dense_reference.prepare(n) if dense is not None else None
                reference = ([basis.conj() @ dense @ basis.T for basis in bases] if dense is not None
                             else anschuetz_optimized.pauli_to_schur(n, coefficients))
                methods = available
                # Rotate the first method across inputs.
                shift = instance % len(methods)
                methods = methods[shift:] + methods[:shift]
                for method in methods:
                    print(f"n={n} instance={instance} method={method}", flush=True)
                    for repeat in range(-args.warmup, args.repeats):
                        stages, blocks, eigensystems = timed_sample(n, method, coefficients, cfg, public_reference)
                        metrics = eigensystem_checks(blocks, eigensystems)
                        metrics.update(reference_checks(n, coefficients, blocks, eigensystems, reference, dense, bases))
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
                        if n == example_n and instance == 0 and method == available[0] and repeat == 0:
                            np.savez(out / "spectrum_example.npz", n=n,
                                **{f"values_{k}": values for k, (values, _) in enumerate(eigensystems)},
                                **{f"vectors_{k}": vectors for k, (_, vectors) in enumerate(eigensystems)})
    write_csv(out / "runs.csv", records)
    write_json(out / "validation.json", dict(timed_samples=len(records), all_valid=all(row["valid"] for row in records),
        maxima={key: max(row[key] for row in records if row[key] is not None) for key in metrics
                if any(row[key] is not None for row in records)},
        note="Dense checks through dense_max_n; larger cases use a second conversion, Pauli round trip and eigenpair residuals."))
    print(f"Saved {len(records)} valid timed samples to {out}", flush=True)


if __name__ == "__main__":
    main()
