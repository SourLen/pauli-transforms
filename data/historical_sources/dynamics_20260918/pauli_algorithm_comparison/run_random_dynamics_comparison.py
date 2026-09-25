"""Fresh Pauli-to-Schur conversion and complete curves on common random inputs.

Input generation, reference construction, validation and serialization are
untimed. Each measured trial constructs a fresh shared bank, converts H/rho/M,
diagonalizes H and evaluates all requested times with the same solver.
Run from Code with ``.venv/bin/python -m
pauli_algorithm_comparison.run_random_dynamics_comparison --output DIR``.
Prepared with Codex assistance in September 2026.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from . import application_conversions as conversions, benchmark_cases
from . import physical_models as pm, random_dynamics_inputs as inputs_module
from .common import specht_multiplicities

METHODS = conversions.CONVERSION_METHODS
PHASES = ("setup_seconds", "conversion_seconds", "eigensolve_seconds", "curve_seconds", "total_seconds")
SOURCE_NAMES = ("run_random_dynamics_comparison.py", "random_dynamics_inputs.py",
                "application_conversions.py", "physical_models.py", "benchmark_cases.py",
                "anschuetz_optimized.py", "anschuetz_public.py", "schur_full_ed.py",
                "schur_separated.py", "separated.py", "schur_hahn.py", "schur_factorial.py", "common.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def times_for(points, tmax):
    if points < 1 or not np.isfinite(tmax) or tmax <= 0:
        raise ValueError("positive point count and finite positive tmax required")
    return np.linspace(0., tmax, points)


def curve_error(values, reference):
    values, reference = np.asarray(values), np.asarray(reference)
    if (values.shape != reference.shape or values.size == 0
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(reference))):
        return float("inf")
    return float(np.max(np.abs(values - reference)))


def timed_sample(n, method, inputs, times, cfg, public_reference=None):
    start = time.perf_counter()
    convert, tables = conversions.prepare(n, method, inputs, cfg, reference=public_reference)
    prepared_at = time.perf_counter()
    blocks = tuple(convert(n, mapping, tables) for mapping in inputs)
    converted_at = time.perf_counter()
    spectrum = pm.diagonalize_blocks(blocks[0])
    spectral_at = time.perf_counter()
    values = pm.expectation_time_series(n, spectrum, blocks[1], blocks[2], times)
    finished_at = time.perf_counter()
    phases = dict(setup_seconds=prepared_at - start,
                  conversion_seconds=converted_at - prepared_at,
                  eigensolve_seconds=spectral_at - converted_at,
                  curve_seconds=finished_at - spectral_at,
                  total_seconds=finished_at - start)
    return blocks, spectrum, values, phases


def trial_diagnostics(n, inputs, blocks, spectrum, values, reference_blocks, reference_curve, observable_scale):
    metrics = inputs_module.state_diagnostics(n, blocks[1])
    metrics.update(inputs_module.spectrum_diagnostics(blocks[0], spectrum))
    metrics.update(h_hermiticity_error=inputs_module.operator_skew(blocks[0]),
                   observable_hermiticity_error=inputs_module.operator_skew(blocks[2]),
                   h_block_error=inputs_module.relative_blocks(blocks[0], reference_blocks[0]),
                   state_block_error=inputs_module.relative_blocks(blocks[1], reference_blocks[1], specht_multiplicities(n)),
                   observable_block_error=inputs_module.relative_blocks(blocks[2], reference_blocks[2]))
    f0 = 2**n * sum(benchmark_cases.pauli_orbit_size(n, key) * coefficient * inputs[2].get(key, 0.)
                    for key, coefficient in inputs[1].items())
    error = curve_error(values, reference_curve)
    metrics.update(scaled_curve_error=error / observable_scale,
                   initial_expectation_error=float(abs(values[0] - f0)) / observable_scale,
                   imaginary_expectation_error=float(np.max(np.abs(np.imag(values)))) / observable_scale)
    return metrics, error


def load_input(out, n, instance):
    input_file = Path(out) / "inputs" / f"n{n}_i{instance}.json"
    payload = json.loads(input_file.read_text())
    mappings = tuple(benchmark_cases.decode_mapping(payload[name]) for name in ("hamiltonian", "state", "observable"))
    reference_file = input_file.with_suffix(".npz")
    with np.load(reference_file, allow_pickle=False) as saved:
        blocks = tuple([saved[f"{prefix}_{k}"] for k in range(n // 2 + 1)] for prefix in ("h", "rho", "m"))
        dense = tuple(saved[f"dense_{prefix}"] for prefix in ("h", "rho", "m")) if "dense_h" in saved else None
    return mappings, blocks, dense, payload, input_file, reference_file


def worker(job, cfg):
    n, instance, method = job["n"], job["instance"], job["method"]
    inputs, references, dense, payload, input_file, reference_file = load_input(cfg["output"], n, instance)
    times = times_for(job["points"], cfg["tmax"])
    spectrum = pm.diagonalize_blocks(references[0])
    reference_curve = pm.expectation_time_series(n, spectrum, references[1], references[2], times)
    scale = max(1., max(float(np.linalg.norm(block, 2)) for block in references[2]))
    reference_checks = {"reference_" + key: value for key, value in
                        inputs_module.spectrum_diagnostics(references[0], spectrum).items()}
    reference_checks["reference_imaginary_error"] = float(np.max(np.abs(reference_curve.imag))) / scale
    if dense is not None:
        reference_checks["full_space_curve_error"] = curve_error(
            reference_curve, inputs_module.full_space_curve(dense, times)) / scale
    inputs_module.require_valid(reference_checks, cfg["acceptance_tolerance"], "Reference curve failed")
    public_reference = conversions.load_public_reference(method, cfg)
    identity = dict(job, seed=payload["seed"], input_sha256=sha256(input_file),
                    reference_sha256=sha256(reference_file), observable_scale=scale,
                    acceptance_tolerance=cfg["acceptance_tolerance"])
    rows, curves = [], []
    for repetition in range(-cfg["warmup"], cfg["repeats"]):
        blocks, actual_spectrum, values, phases = timed_sample(n, method, inputs, times, cfg, public_reference)
        diagnostics, error = trial_diagnostics(n, inputs, blocks, actual_spectrum, values,
                                               references, reference_curve, scale)
        valid = all(np.isfinite(value) and 0 <= value <= cfg["acceptance_tolerance"]
                    for value in diagnostics.values())
        if repetition < 0 and not valid:
            raise ArithmeticError(f"Warmup failed: {diagnostics}")
        if repetition >= 0:
            rows.append(dict(identity, repeat=repetition, valid=bool(valid),
                             status="ok" if valid else "failed", max_abs_error=error,
                             **phases, **diagnostics, **reference_checks))
        if repetition == cfg["repeats"] - 1:
            curves = [dict(job, time=float(t), value_real=float(v.real), value_imag=float(v.imag),
                           reference_real=float(r.real), reference_imag=float(r.imag))
                      for t, v, r in zip(times, values, reference_curve, strict=True)]
    return dict(rows=rows, curves=curves)


def accepted(row):
    if row.get("valid") is not True or row.get("status") != "ok":
        return False
    keys = PHASES + tuple(key for key in row if key.endswith(("_error", "_mass", "_residual")))
    try:
        if any(not np.isfinite(float(row[key])) or float(row[key]) < 0 for key in keys):
            return False
        tolerance = float(row.get("acceptance_tolerance", 1e-8))
        diagnostics = tuple(key for key in keys if key not in PHASES and key != "max_abs_error")
        if any(float(row[key]) > tolerance for key in diagnostics):
            return False
        return bool(np.isclose(sum(row[field] for field in PHASES[:-1]), row["total_seconds"], rtol=1e-10, atol=1e-12))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def summarize(rows, expected_samples=None):
    summaries = []
    group_fields = ("kind", "n", "points", "method")
    for key in sorted({tuple(row[field] for field in group_fields) for row in rows}):
        selected = [row for row in rows if tuple(row[field] for field in group_fields) == key]
        valid = all(accepted(row) for row in selected)
        if expected_samples is not None:
            valid = valid and len(selected) == expected_samples
        entry = dict(zip(group_fields, key))
        entry.update(valid=valid, samples=len(selected),
                     instances=len({row["instance"] for row in selected}),
                     statuses=sorted({row["status"] for row in selected}))
        if valid:
            for field in PHASES:
                quantiles = np.percentile([row[field] for row in selected], [25, 50, 75])
                entry.update({field + suffix: float(value) for suffix, value in
                              zip(("_q25", "_median", "_q75"), quantiles, strict=True)})
            for field in ("max_abs_error", "scaled_curve_error"):
                entry[field] = max(row[field] for row in selected)
        summaries.append(entry)
    return summaries


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {name: sha256(folder / name) for name in SOURCE_NAMES}


def prepare_saved_input(out, n, instance, cfg):
    seed = cfg["seed"] + 1009 * n + instance
    inputs, generated_state = inputs_module.make_inputs(n, seed)
    refs, dense, metrics = inputs_module.independent_reference(
        n, inputs, generated_state, dense_max_n=cfg["dense_max_n"], tolerance=cfg["acceptance_tolerance"])
    rho = inputs[1]
    imaginary_norm = sum(benchmark_cases.pauli_orbit_size(n, key) * abs(value.imag)**2 for key, value in rho.items())
    whole_norm = sum(benchmark_cases.pauli_orbit_size(n, key) * abs(value)**2 for key, value in rho.items())
    metrics["state_coefficient_imaginary_error"] = float(np.sqrt(imaginary_norm / max(whole_norm, np.finfo(float).tiny)))
    inputs_module.require_valid(metrics, cfg["acceptance_tolerance"], "Serialized input failed")
    path = Path(out) / "inputs" / f"n{n}_i{instance}.json"
    write_json(path, dict(n=n, instance=instance, seed=seed,
        **{name: benchmark_cases.encode_mapping(mapping) for name, mapping in
           zip(("hamiltonian", "state", "observable"), inputs, strict=True)}))
    arrays = {f"{prefix}_{k}": block for prefix, blocks in zip(("h", "rho", "m"), refs, strict=True)
              for k, block in enumerate(blocks)}
    if dense is not None:
        arrays.update({f"dense_{prefix}": matrix for prefix, matrix in zip(("h", "rho", "m"), dense, strict=True)})
    np.savez(path.with_suffix(".npz"), **arrays)
    return dict(n=n, instance=instance, seed=seed, valid=True, **metrics)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path)
    result.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 5, 8, 12, 16, 20])
    result.add_argument("--methods", nargs="+", choices=METHODS, default=list(conversions.DEFAULT_CONVERSION_METHODS))
    result.add_argument("--points", type=int, default=241)
    result.add_argument("--tmax", type=float, default=12.)
    result.add_argument("--instances", type=int, default=3)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--seed", type=int, default=20260918)
    result.add_argument("--dense-max-n", type=int, default=5)
    result.add_argument("--sweep-n", type=int, default=12)
    result.add_argument("--sweep-points", type=int, nargs="*", default=[])
    result.add_argument("--timeout", type=float, default=360.)
    result.add_argument("--public-max-n", type=int, default=5)
    result.add_argument("--anschuetz-source")
    result.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        cfg = json.loads(args.config.read_text())
        with threadpool_limits(limits=1):
            print(json.dumps(worker(json.loads(args.worker.read_text()), cfg), allow_nan=False))
        return
    if args.output is None:
        raise SystemExit("--output is required")
    if (min(args.n + [args.sweep_n, args.points, args.instances, args.repeats] + args.sweep_points) < 1
            or args.warmup < 0 or args.dense_max_n < 0 or args.public_max_n < 0
            or not np.isfinite(args.timeout) or args.timeout <= 0):
        raise SystemExit("Invalid size, repetition, or timeout parameter")
    times_for(args.points, args.tmax)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory is not empty; choose a new campaign directory")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "jobs", "source_snapshot"):
        (out / name).mkdir()
    cfg = {key: value for key, value in vars(args).items() if key not in ("worker", "config")}
    cfg.update(output=str(out), schur_backend="hahn", schur_dtype="float64", threads=1,
        acceptance_tolerance=1e-8,
        input_family="Independent general Gaussian Pauli orbit H,M with variance 1/(number of orbits * orbit size); independent complex Wishart state blocks of dimensions d by 2d, rho_k=(d/2^n) GGdagger/Tr(GGdagger)",
        input_contract="Saved common Pauli coordinates for H,rho,M; state inverse conversion and all input generation excluded from timing",
        timing_contract="Fresh shared tables + three conversions + common numpy eigh block solver + complete expectation curve; imports, inputs, validation and I/O excluded",
        curve_stage="Includes one-time energy-basis contractions and evaluation of all requested times",
        output_contract="Complex expectation values at all requested times; no density trajectories",
        acceptance_contract="max_t |f-f_ref|/max(1,||M||_op)<=1e-8, plus trace, block-relative positivity/Hermiticity, conversion and solver checks",
        eigensolver_roundoff="Shared existing solver checks Hermiticity before averaging H with Hdagger; no state repair or output normalization",
        reference_contract="Independent Appendix-E H,M and original generated state; inverse state conversion checked by Appendix E and full matrices through dense_max_n; full expm curves through dense_max_n")
    conversions.configure_public_source(cfg)
    write_json(out / "config.json", cfg)
    before = source_hashes()
    folder = Path(__file__).resolve().parent
    for name in SOURCE_NAMES:
        (out / "source_snapshot" / name).write_bytes((folder / name).read_bytes())
    with threadpool_limits(limits=1):
        environment = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), processor=platform.processor(),
            cpu_model=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                            if line.startswith("model name")), None),
            threadpools=threadpool_info(), source_sha256=before,
            utc_started=datetime.now(timezone.utc).isoformat())
        write_json(out / "environment.json", environment)
        sizes = sorted(set(args.n + ([args.sweep_n] if args.sweep_points else [])))
        input_checks = []
        for n in sizes:
            for instance in range(args.instances):
                print(f"Preparing independent input/reference n={n} instance={instance}", flush=True)
                input_checks.append(prepare_saved_input(out, n, instance, cfg))
        write_json(out / "input_validation.json", dict(all_valid=True, inputs=input_checks))
    input_hashes = {path.name: sha256(path) for path in sorted((out / "inputs").iterdir())}
    write_json(out / "input_hashes.json", input_hashes)
    jobs = []
    workloads = [("size", n, args.points) for n in sorted(set(args.n))]
    workloads += [("times", args.sweep_n, points) for points in sorted(set(args.sweep_points))]
    for kind, n, points in workloads:
        for instance in range(args.instances):
            for method in dict.fromkeys(args.methods):
                if method != conversions.PUBLIC_METHOD or n <= args.public_max_n:
                    jobs.append(dict(kind=kind, n=n, points=points, instance=instance, method=method))
    random.Random(args.seed).shuffle(jobs)
    write_json(out / "job_order.json", jobs)
    invocation = [sys.executable, "-m", "pauli_algorithm_comparison.run_random_dynamics_comparison"]
    invocation += sys.argv[1:] if argv is None else list(argv)
    (out / "reproduce.sh").write_text("#!/bin/sh\n# Choose a new output directory before rerunning.\ncd "
        + shlex.quote(str(folder.parent)) + "\n" + shlex.join(invocation) + "\n")
    rows, curves = [], []
    expected_samples = args.instances * args.repeats
    for index, job in enumerate(jobs):
        name = f"{index:04d}_{job['kind']}_n{job['n']}_L{job['points']}_i{job['instance']}_{job['method']}"
        job_file = out / "jobs" / f"{name}.json"
        write_json(job_file, job)
        print(f"{index + 1}/{len(jobs)} {name}", flush=True)
        command = [sys.executable, "-m", "pauli_algorithm_comparison.run_random_dynamics_comparison",
                   "--worker", str(job_file), "--config", str(out / "config.json")]
        try:
            result = subprocess.run(command, cwd=folder.parent, capture_output=True, text=True,
                                    timeout=args.timeout, env=os.environ.copy())
            (out / "jobs" / f"{name}.log").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"Worker exit {result.returncode}: {result.stderr[-4000:]}")
            payload = json.loads(result.stdout)
            new_rows = payload["rows"]
            curves.extend(payload["curves"])
        except subprocess.TimeoutExpired:
            new_rows = [dict(job, repeat=None, valid=False, status="timeout",
                             detail=f"Worker exceeded {args.timeout:g}s including imports, validation, warmup and repetitions")]
        except Exception:
            new_rows = [dict(job, repeat=None, valid=False, status="error", detail=traceback.format_exc())]
        rows.extend(new_rows)
        with (out / "runs.jsonl").open("a") as stream:
            for row in new_rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_csv(out / "runs.csv", rows)
        write_csv(out / "curves.csv", curves)
        write_json(out / "summary.json", summarize(rows, expected_samples))
        print(f"  {len(new_rows)} records; accepted={all(accepted(row) for row in new_rows)}", flush=True)
    after = source_hashes()
    inputs_unchanged = input_hashes == {path.name: sha256(path) for path in sorted((out / "inputs").iterdir())}
    public_unchanged = cfg.get("anschuetz_public_source") == conversions.public_source_snapshot(cfg)
    valid_rows = [row for row in rows if accepted(row)]
    metric_names = sorted({key for row in valid_rows for key in row if key.endswith(("_error", "_mass", "_residual"))})
    validation = dict(all_valid=bool(rows) and before == after and inputs_unchanged and public_unchanged
                      and all(entry["valid"] for entry in summarize(rows, expected_samples)),
        jobs=len(jobs), timed_samples=len(valid_rows), records=len(rows), source_unchanged=before == after,
        inputs_unchanged=inputs_unchanged, public_source_unchanged=public_unchanged,
        source_sha256_after=after, utc_completed=datetime.now(timezone.utc).isoformat(),
        maxima={key: max(row[key] for row in valid_rows if key in row) for key in metric_names},
        note="Full expm checks apply only through dense_max_n; larger checks compare independent conversions, known Wishart blocks and original-block eigensolver residuals.")
    write_json(out / "validation.json", validation)
    print(f"Saved {len(valid_rows)} accepted timings; all_valid={validation['all_valid']}", flush=True)
    if not validation["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
