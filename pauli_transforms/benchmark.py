"""General and fixed-locality conversion benchmarks used in the thesis."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .benchmark_helpers import read_jsonl, sha256, source_snapshot, write_csv, write_json

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def worker(job_file):
    """One process per method/input/cache mode; numerical checks are untimed."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    from . import anschuetz_optimized, anschuetz_public, chang, dense_reference, transforms
    from .benchmark_cases import banded_blocks, collective_reference, decode_mapping, validate_schur

    job = json.loads(Path(job_file).read_text())
    folder = Path(job_file).parent
    n, method, mode = job["n"], job["method"], job["cache_mode"]
    cfg = job.pop("config")
    if method == "anschuetz_public_original" and mode != "cold":
        raise ValueError("the public routine has no cached interface")
    coefficients = decode_mapping(json.loads(Path(job.pop("input_path")).read_text())["coefficients"])
    public = anschuetz_public.load(cfg["anschuetz_source"]) if method == "anschuetz_public_original" else None
    module = {"separated": transforms, "anschuetz_optimized": anschuetz_optimized,
              "chang": chang, "anschuetz_public_original": anschuetz_public}[method]

    def prepare():
        if method == "chang":
            return chang.prepare(n, job["locality"])
        if method == "anschuetz_public_original":
            return public
        return module.prepare(n)

    with threadpool_limits(limits=1):
        write_json(folder / "phase.json", "reference")
        reference_start = time.perf_counter()
        reference = (collective_reference(n, coefficients) if cfg["comparison"] == "chang" else
                     dense_reference.pauli_to_schur(n, coefficients) if n <= 5 else None)
        reference_seconds = time.perf_counter() - reference_start
        validation_kind = ("independent_collective_pauli_recurrence" if cfg["comparison"] == "chang" else
                           "independent_dense_singlet_dicke_projection" if n <= 5 else
                           "pauli_schur_pauli_roundtrip")
        write_json(folder / "phase.json", "cache_setup")
        cache_start = time.perf_counter()
        cached = prepare() if mode == "cached" else None
        cache_build_seconds = time.perf_counter() - cache_start if mode == "cached" else 0.
        with (folder / "samples.jsonl").open("w") as stream:
            for repeat in range(-cfg["warmup"], cfg["repeats"]):
                write_json(folder / "phase.json", "timed_conversion")
                started = time.perf_counter()
                tables = cached if mode == "cached" else prepare()
                prepared = time.perf_counter()
                raw = module.pauli_to_schur(n, coefficients, tables)
                applied = time.perf_counter()
                result = banded_blocks(raw, job["locality"]) if cfg["comparison"] == "chang" and method == "separated" else raw
                finished = time.perf_counter()
                write_json(folder / "phase.json", "validation")
                validation_start = time.perf_counter()
                metrics = validate_schur(n, raw, coefficients, reference, cfg["atol"], cfg["rtol"])
                if result is not raw:
                    metrics["valid"] &= validate_schur(n, result, coefficients, reference, cfg["atol"], cfg["rtol"])["valid"]
                if repeat < 0 and not metrics["valid"]:
                    raise ArithmeticError(f"Warmup validation failed: {metrics}")
                if repeat >= 0:
                    # Public loading and hash verification occur before the timer.
                    # All original block construction occurs in apply_seconds.
                    setup_seconds = prepared-started if mode == "cold" and public is None else 0.
                    row = dict(job, repeat=repeat, preparation_seconds=0.,
                               setup_seconds=setup_seconds, apply_seconds=applied-prepared,
                               output_seconds=finished-applied, elapsed_seconds=setup_seconds+finished-prepared,
                               cache_build_seconds=cache_build_seconds, reference_seconds=reference_seconds,
                               validation_seconds=time.perf_counter()-validation_start,
                               status="ok" if metrics["valid"] else "incorrect", dtype="complex128",
                               threads=1, validation_kind=validation_kind, **metrics)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()


def run_job(job, folder):
    """Retain completed samples and an explicit limit record if a worker stops."""
    folder.mkdir()
    write_json(folder / "job.json", job)
    env = dict(os.environ, **{name: "1" for name in THREAD_VARIABLES})
    with (folder / "worker.log").open("w") as log:
        try:
            result = subprocess.run([sys.executable, "-m", "pauli_transforms.benchmark",
                                     "--worker", str(folder / "job.json")], env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=job["config"]["timeout"])
            status = "worker_error" if result.returncode else None
        except subprocess.TimeoutExpired:
            status = "timeout"
    samples = folder / "samples.jsonl"
    rows = read_jsonl(samples, interrupted=bool(status)) if samples.exists() else []
    if status:
        phase_file = folder / "phase.json"
        try:
            failure_phase = json.loads(phase_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            failure_phase = "unknown"
        rows.append({k: v for k, v in job.items() if k not in ("config", "input_path")} |
                    dict(status=status, valid=False, repeat=-1,
                         failure_phase=failure_phase))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=("anschuetz", "chang"), default="anschuetz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--localities", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--anschuetz-source", type=Path, help="pinned external utils.py or checkout")
    parser.add_argument("--input-directory", type=Path, help="reuse archived inputs instead of generating them")
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--smoke", action="store_true", help="two small sizes, one input and one repetition")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.worker)
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.smoke:
        args.n = [2, 4] if args.comparison == "anschuetz" else [4, 6]
        args.instances = args.repeats = 1
    if args.n is None:
        args.n = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20] if args.comparison == "anschuetz" else [4, 8, 12, 16, 20, 24, 32, 40]
    args.n = sorted(set(args.n))
    args.localities = sorted(set(args.localities))
    if (min(args.n + args.localities + [args.instances, args.repeats]) < 1 or args.warmup < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("sizes/counts/timeout must be positive and warmup nonnegative")
    if args.comparison == "chang" and max(args.localities) > min(args.n):
        parser.error("locality cannot exceed a requested size")
    seed = args.seed if args.seed is not None else 20260917 if args.comparison == "anschuetz" else 20260914
    if seed < 0:
        parser.error("seed must be nonnegative")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; choose a new directory")
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    from .benchmark_helpers import environment_record
    from .benchmark_cases import decode_mapping, encode_mapping, make_case
    from . import anschuetz_public
    if args.anschuetz_source:
        anschuetz_public.load(args.anschuetz_source)  # Fail before creating a campaign if the pin is wrong.
    methods = ["separated", "anschuetz_optimized"] if args.comparison == "anschuetz" else ["separated", "chang"]
    if args.anschuetz_source and args.comparison == "anschuetz":
        methods.append("anschuetz_public_original")
    cfg = dict(comparison=args.comparison, n_values=args.n, localities=args.localities,
               instances=args.instances, repeats=args.repeats, warmup=args.warmup, seed=seed,
               methods=methods, threads=1, schur_backend="hahn", dtype="complex128",
               timeout=args.timeout, atol=1e-9, rtol=1e-8,
               anschuetz_public_max_n=5,
               anschuetz_source=str(args.anschuetz_source.resolve()) if args.anschuetz_source else None,
               input_directory=str(args.input_directory.resolve()) if args.input_directory else None)
    if args.anschuetz_source:
        cfg["public_commit"] = anschuetz_public.COMMIT
        cfg["public_source_sha256"] = anschuetz_public.UTILS_SHA256
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir()
    (output / "jobs").mkdir()
    write_json(output / "config.json", cfg)
    source_snapshot(output)
    with threadpool_limits(limits=1):
        write_json(output / "environment.json", environment_record())
    family = "general" if args.comparison == "anschuetz" else "fixed_weight"
    order = random.Random(seed)
    rows = []
    for n in args.n:
        for locality in ([0] if family == "general" else args.localities):
            for instance in range(args.instances):
                case_id = f"n{n}_{family}_l{locality}_i{instance}"
                input_seed = seed + 1009*n + 104729*instance + 37*locality
                input_path = output / "inputs" / (case_id + ".json")
                if args.input_directory:
                    payload = json.loads((args.input_directory / input_path.name).read_text())
                    if (payload["n"], payload["family"], payload["locality"]) != (n, family, locality):
                        raise ValueError("archived input metadata does not match requested case")
                    coefficients = decode_mapping(payload["coefficients"])
                    input_seed = payload["seed"]
                else:
                    coefficients = make_case(n, family, input_seed, locality)
                    payload = dict(n=n, family=family, locality=locality, seed=input_seed,
                                   normalization="common coefficient of each individual Pauli word",
                                   coefficients=encode_mapping(coefficients))
                write_json(input_path, payload)
                jobs = [(method, mode) for method in methods for mode in ("cold", "cached")
                        if method != "anschuetz_public_original" or mode == "cold"]
                order.shuffle(jobs)
                for method, mode in jobs:
                    job_id = f"{case_id}_{method}_{mode}"
                    job = dict(job_id=job_id, comparison=args.comparison, method=method, n=n,
                               family=family, locality=locality, instance=instance, input_seed=input_seed,
                               input_path=str(input_path), input_hash=sha256(input_path), cache_mode=mode,
                               output_contract="dense_schur_blocks" if family == "general" else "banded_schur_CSR", config=cfg)
                    print(job_id, flush=True)
                    if method == "anschuetz_public_original" and n > 5:
                        samples = [{k: v for k, v in job.items() if k not in ("config", "input_path")} |
                                   dict(status="configured_size_limit", valid=False, repeat=-1)]
                    else:
                        samples = run_job(job, output / "jobs" / job_id)
                    rows.extend(samples)
                    with (output / "runs.jsonl").open("w") as stream:
                        for row in rows:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_csv(output / "runs.csv", rows)
    failures = [row for row in rows if not row["valid"] and row["status"] != "configured_size_limit"]
    write_json(output / "validation.json", dict(valid=sum(row["valid"] for row in rows), failures=len(failures)))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
