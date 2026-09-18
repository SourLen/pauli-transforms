"""Fresh H/state/observable conversions and complete expectation curves.

Each trial builds shared conversion tables, converts all three inputs, solves
with numpy.linalg.eigh, and evaluates the full curve. References and checks are
outside timers. Defaults reproduce the thesis size and time-grid campaigns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import traceback
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import application_conversions as conversions, benchmark_cases
from . import physical_models as pm, random_dynamics_inputs as inputs_module
from .common import specht_multiplicities
from .benchmark_helpers import environment_record, source_snapshot, sha256, write_json, write_csv

METHODS = conversions.CONVERSION_METHODS
PHASES = ("setup_seconds", "conversion_seconds", "eigensolve_seconds", "curve_seconds", "total_seconds")


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



def prepare_saved_input(out, n, instance, cfg):
    seed = cfg["seed"] + 1009 * n + instance
    archived = None
    if cfg.get("input_directory"):
        folder = Path(cfg["input_directory"])
        folder = folder / "inputs" if (folder / "inputs").is_dir() else folder
        archived = folder / f"n{n}_i{instance}.json"
        payload = json.loads(archived.read_text())
        if payload["n"] != n or payload["instance"] != instance:
            raise ValueError("Archived input identity does not match its filename")
        seed = payload["seed"]
        inputs = tuple(benchmark_cases.decode_mapping(payload[name])
                       for name in ("hamiltonian", "state", "observable"))
        with np.load(archived.with_suffix(".npz"), allow_pickle=False) as saved:
            generated_state = [saved[f"rho_{k}"] for k in range(n // 2 + 1)]
    else:
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
    if archived is not None:
        path.write_bytes(archived.read_bytes())
    arrays = {f"{prefix}_{k}": block for prefix, blocks in zip(("h", "rho", "m"), refs, strict=True)
              for k, block in enumerate(blocks)}
    if dense is not None:
        arrays.update({f"dense_{prefix}": matrix for prefix, matrix in zip(("h", "rho", "m"), dense, strict=True)})
    np.savez(path.with_suffix(".npz"), **arrays)
    return dict(n=n, instance=instance, seed=seed, valid=True, **metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 5, 8, 12, 16, 20])
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(conversions.DEFAULT_CONVERSION_METHODS))
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--tmax", type=float, default=12.)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--dense-max-n", type=int, default=5)
    parser.add_argument("--sweep-n", type=int, default=12)
    parser.add_argument("--sweep-points", type=int, nargs="*", default=[1, 8, 32, 128, 512, 2048],
                        help="Time-grid sweep; pass this flag with no values to omit it")
    parser.add_argument("--public-max-n", type=int, default=5)
    parser.add_argument("--anschuetz-source")
    parser.add_argument("--input-directory", type=Path,
                        help="Read exact saved JSON/NPZ inputs from a prior campaign")
    parser.add_argument("--smoke", action="store_true", help="Small size and time-grid checks")
    args = parser.parse_args(argv)
    if args.smoke:
        args.n, args.instances, args.repeats, args.warmup = [2, 3], 1, 1, 0
        args.points, args.sweep_n, args.sweep_points = 5, 3, [1, 8]
    if (min(args.n + [args.sweep_n, args.points, args.instances, args.repeats] + args.sweep_points) < 1
            or min(args.warmup, args.seed, args.dense_max_n, args.public_max_n) < 0):
        parser.error("Positive sizes/repetitions and nonnegative warmup/validation sizes required")
    times_for(args.points, args.tmax)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; choose a new campaign directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir()
    cfg = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    cfg.update(output=str(out), schur_backend="hahn", schur_dtype="float64", threads=1,
        acceptance_tolerance=1e-8,
        coefficient_variance="1/(number of orbits * orbit size)",
        state_ensemble="rho_k=(d/2^n) GGdagger/Tr(GGdagger), G complex Gaussian d by 2d",
        eigensolver="numpy.linalg.eigh", fresh_shared_tables=True,
        timed_stages=list(PHASES[:-1]))
    conversions.configure_public_source(cfg)
    write_json(out / "config.json", cfg)
    hashes = source_snapshot(out)
    rows, curves = [], []
    workloads = [("size", n, args.points) for n in sorted(set(args.n))]
    workloads += [("times", args.sweep_n, points) for points in sorted(set(args.sweep_points))]
    jobs = [dict(kind=kind, n=n, points=points, instance=instance, method=method)
            for kind, n, points in workloads for instance in range(args.instances)
            for method in dict.fromkeys(args.methods)
            if method != conversions.PUBLIC_METHOD or n <= args.public_max_n]
    if not jobs:
        parser.error("No methods are available at the requested sizes")
    random.Random(args.seed).shuffle(jobs)
    write_json(out / "job_order.json", jobs)
    with threadpool_limits(limits=1):
        write_json(out / "environment.json", dict(environment_record(), source_sha256=hashes))
        input_checks = []
        for n in sorted({job["n"] for job in jobs}):
            for instance in range(args.instances):
                print(f"Preparing input n={n}, instance={instance}", flush=True)
                input_checks.append(prepare_saved_input(out, n, instance, cfg))
        write_json(out / "input_validation.json", dict(all_valid=True, inputs=input_checks))
        write_json(out / "input_hashes.json", {path.name: sha256(path) for path in sorted((out / "inputs").iterdir())})
        with (out / "runs.jsonl").open("w") as stream:
            for index, job in enumerate(jobs):
                print(f"{index + 1}/{len(jobs)} {job}", flush=True)
                try:
                    payload = worker(job, cfg)
                    new_rows = payload["rows"]
                    curves.extend(payload["curves"])
                except Exception:
                    new_rows = [dict(job, repeat=None, valid=False, status="error", detail=traceback.format_exc())]
                rows.extend(new_rows)
                for row in new_rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    write_csv(out / "runs.csv", rows)
    write_csv(out / "curves.csv", curves)
    summary = summarize(rows, args.instances * args.repeats)
    write_json(out / "summary.json", summary)
    valid_rows = [row for row in rows if accepted(row)]
    metric_names = sorted({key for row in valid_rows for key in row
                           if key.endswith(("_error", "_mass", "_residual"))})
    validation = dict(all_valid=bool(rows) and all(entry["valid"] for entry in summary),
        jobs=len(jobs), timed_samples=len(valid_rows), records=len(rows),
        maxima={key: max(row[key] for row in valid_rows if key in row) for key in metric_names},
        note="Full-space checks only through dense_max_n; larger cases use independent conversion routes and original state blocks.")
    write_json(out / "validation.json", validation)
    print(f"Saved {len(valid_rows)} accepted timings; all_valid={validation['all_valid']}", flush=True)
    if not validation["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
