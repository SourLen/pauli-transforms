"""Reproducible, isolated benchmarks of the local thesis comparison kernels.

Run via scripts/run_vs_*.sh or python -m pauli_algorithm_comparison.benchmark.
Each worker handles one method/input/cache mode and checkpoints every sample.
Imports, reference calculations, and file writing are outside measured phases.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
METHODS = {
    "anschuetz": ("separated", "anschuetz_public", "anschuetz_optimized",
                  "anschuetz_tensor_network"),
    "chang": ("separated", "chang"),
    "spencer": ("separated", "spencer"),
    "georges": ("separated", "georges"),
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break  # A worker may have been killed while writing its last sample.
            raise
    return rows


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def array_bytes(value):
    """Numerical array storage only; process peak RSS is recorded separately."""
    seen = set()

    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if hasattr(item, "indptr"):
            return item.data.nbytes + item.indices.nbytes + item.indptr.nbytes
        if hasattr(item, "nbytes"):
            return item.nbytes
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(v) for v in item)
        return 0
    return int(visit(value))


def estimate_memory(n, method, comparison, expanded):
    """Conservative allocation guard, supplemented by the worker RSS limit."""
    from math import comb
    D = comb(n + 3, 3)
    estimate = 512 * D
    if method == "anschuetz_tensor_network":
        # Full F storage plus count-MPO/CG contraction workspaces. Even when
        # storing selected columns, the network visits every Pauli count.
        from .anschuetz_tensor_network import estimate_workspace_bytes
        estimate += 16 * D * D + estimate_workspace_bytes(n) + 64 * 1024 ** 2
    elif method.startswith("anschuetz"):
        estimate += 3 * 16 * D * D
    if method == "separated":
        estimate += 24 * sum((m + 1) ** 2 for m in range(n + 1))
        if comparison in ("anschuetz", "chang"):
            estimate += 24 * sum((min(r, n - s) + 1) ** 2
                                 for r in range(n + 1) for s in range(r, n + 1))
    if comparison == "georges":
        # Dense validation also uses an expanded reference for both methods.
        estimate += (112 if method == "georges" else 48) * (4 ** n)
    elif expanded and comparison == "spencer":
        estimate += 1024 * D
    return estimate


def base_row(job):
    return {key: job[key] for key in ("job_id", "comparison", "method", "n", "family",
                                     "locality", "instance", "input_seed", "cache_mode",
                                     "input_hash", "pauli_support", "output_contract")}


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    cfg = job["config"]
    for name in THREAD_VARS:
        os.environ[name] = str(cfg["threads"])
    import numpy as np
    from . import anschuetz_optimized, anschuetz_public, anschuetz_tensor_network
    from . import chang, georges, schur_full_ed
    from . import schur_separated, separated, spencer
    from .benchmark_cases import (OrbitRowOracle, array_errors, banded_blocks, collective_reference,
                                  decode_mapping, expanded_array, expanded_dictionary, mapping_errors,
                                  pauli_orbit_size, validate_schur)

    folder = Path(job_path).parent
    samples_path = folder / "samples.jsonl"
    samples_path.write_text("")
    n, method = job["n"], job["method"]
    schur_task = job["comparison"] in ("anschuetz", "chang")
    source = json.loads(Path(job["input_path"]).read_text())
    coefficients = decode_mapping(source["coefficients"])
    entries = decode_mapping(source["entries"]) if "entries" in source else None
    atol, rtol = cfg["atol"], cfg["rtol"]

    def phase(name):
        write_json(folder / "phase.json", {"phase": name})

    def timed(function):
        start = time.perf_counter_ns()
        try:
            value = function()
        except Exception as exc:
            exc.benchmark_seconds = (time.perf_counter_ns() - start) * 1e-9
            raise
        return value, (time.perf_counter_ns() - start) * 1e-9

    phase("reference")
    start = time.perf_counter()
    reference = None
    if job["comparison"] == "chang":
        reference = collective_reference(n, coefficients)
        validation_kind = "independent_collective_pauli_recurrence"
    elif schur_task and n <= cfg["dense_reference_max_n"]:
        reference = schur_full_ed.pauli_to_schur(n, coefficients)
        validation_kind = "independent_dense_singlet_dicke_projection"
    elif schur_task:
        validation_kind = "pauli_schur_pauli_roundtrip"
    elif job["comparison"] == "georges":
        reference = expanded_array(n, coefficients)
        validation_kind = "planted_full_pauli_array"
    else:
        reference = expanded_dictionary(n, coefficients)
        validation_kind = "planted_full_sparse_pauli_dictionary"
    reference_seconds = time.perf_counter() - start
    public = anschuetz_public.load(cfg["anschuetz_source"]) if method == "anschuetz_public" else None

    def prepare_tables():
        if method == "separated":
            return schur_separated.prepare(n) if schur_task else separated.prepare(n)
        if method == "anschuetz_public":
            return anschuetz_public.prepare(n, public)
        if method == "anschuetz_optimized":
            return anschuetz_optimized.prepare(n)
        if method == "anschuetz_tensor_network":
            return anschuetz_tensor_network.prepare(
                n, max_memory_bytes=int(cfg["max_memory_mb"] * 1024 ** 2))
        if method == "chang":
            return chang.prepare(n, job["locality"])
        return None

    reusable = method in ("separated", "anschuetz_public", "anschuetz_optimized",
                          "anschuetz_tensor_network", "chang")
    cached_tables, cache_build_seconds = None, 0.0
    if job["cache_mode"] == "cached" and reusable:
        phase("cache_setup")
        cached_tables, cache_build_seconds = timed(prepare_tables)

    def one_sample(repeat, warmup=False):
        row = base_row(job)
        row.update(repeat=repeat, algorithm_seed=job["input_seed"] + 97531 * (repeat + 1),
                   dtype="complex128", threads=cfg["threads"], status="pending", valid=False,
                   preparation_seconds=0.0, setup_seconds=0.0, apply_seconds=0.0, output_seconds=0.0,
                   cache_build_seconds=cache_build_seconds, validation_kind=validation_kind,
                   reference_seconds=reference_seconds, input_generation_seconds=source["generation_seconds"],
                   input_contract="pauli_orbit_coefficients" if schur_task else "entry_orbit_coefficients",
                   query_count=None, row_queries=None, entry_queries=None, returned_entries=None,
                   delta=cfg["delta"] if method == "spencer" else None,
                   backend={"anschuetz_public": "pinned_public_F_columns",
                            "anschuetz_optimized": "constrained_Appendix_E_DP",
                            "anschuetz_tensor_network": "Appendix_C_CG_Pauli_count_MPO",
                            "separated": "numpy_Krawtchouk_Hahn" if schur_task else "numpy_Krawtchouk",
                            "chang": "Algorithm_1_corrected_normalization_CSR",
                            "georges": "numpy_FWHT", "spencer": "pinned_sparse_row_oracle"}[method])
        oracle = None
        try:
            phase("setup")
            tables = cached_tables
            if job["cache_mode"] == "cold" and reusable:
                tables, row["setup_seconds"] = timed(prepare_tables)
            row["table_array_bytes"] = array_bytes(tables)
            phase("input_preparation")
            if method == "georges":
                from .common import entries_to_dense
                matrix, row["preparation_seconds"] = timed(lambda: entries_to_dense(n, entries))
            if method == "spencer":
                oracle, row["preparation_seconds"] = timed(
                    lambda: OrbitRowOracle(n, entries, max_queries=cfg["max_queries"]))

            def apply():
                if method == "separated":
                    return (schur_separated.pauli_to_schur(n, coefficients, tables) if schur_task
                            else separated.decompose(n, entries, tables))
                if method == "anschuetz_public":
                    return anschuetz_public.pauli_to_schur(n, coefficients, tables=tables)
                if method == "anschuetz_optimized":
                    return anschuetz_optimized.pauli_to_schur(n, coefficients, tables)
                if method == "anschuetz_tensor_network":
                    return anschuetz_tensor_network.pauli_to_schur(
                        n, coefficients, tables,
                        max_memory_bytes=int(cfg["max_memory_mb"] * 1024 ** 2))
                if method == "chang":
                    return chang.pauli_to_schur(n, coefficients, tables)
                if method == "georges":
                    return georges.decompose(matrix, backend="numpy")
                return spencer.decompose(oracle, job["pauli_support"], delta=cfg["delta"],
                                         seed=row["algorithm_seed"], tol=cfg["tol"])

            phase("application")
            raw, row["apply_seconds"] = timed(apply)
            completed = True
            if method == "spencer":
                raw, completed = raw
            row["completed"] = bool(completed)

            def format_output():
                if job["comparison"] == "chang" and method == "separated":
                    return banded_blocks(raw, job["locality"])
                if job["comparison"] == "georges":
                    if job["output_contract"] == "expanded_pauli_array":
                        return expanded_array(n, raw) if method == "separated" else raw
                    if method == "georges":
                        from .common import compress_pauli
                        return compress_pauli(n, raw)
                if job["comparison"] == "spencer" and method == "separated":
                    return expanded_dictionary(n, raw, tol=cfg["tol"])
                return raw

            phase("output_formatting")
            formatted, row["output_seconds"] = timed(format_output)
            row["elapsed_seconds"] = sum(row[name] for name in
                                          ("preparation_seconds", "setup_seconds", "apply_seconds", "output_seconds"))
            row["output_array_bytes"] = array_bytes(formatted)
            if not warmup:
                write_json(folder / "pending.json", row)
            phase("validation")
            started = time.perf_counter()
            if schur_task:
                metrics = validate_schur(n, raw, coefficients, reference, atol, rtol)
                if job["comparison"] == "chang":
                    final_metrics = validate_schur(n, formatted, coefficients, reference, atol, rtol)
                    metrics["valid"] = metrics["valid"] and final_metrics["valid"]
            elif method == "georges":
                metrics = array_errors(raw, reference, atol, rtol)
                if job["output_contract"] == "compressed_pauli_orbits":
                    final_metrics = mapping_errors(formatted, coefficients, atol, rtol,
                                                   weights=lambda key: pauli_orbit_size(n, key))
                    metrics["valid"] = metrics["valid"] and final_metrics["valid"]
            elif job["comparison"] == "spencer":
                metrics = mapping_errors(formatted, reference, atol, rtol)
            elif job["output_contract"] == "expanded_pauli_array":
                metrics = array_errors(formatted, reference, atol, rtol)
            else:
                metrics = mapping_errors(formatted, coefficients, atol, rtol,
                                         weights=lambda key: pauli_orbit_size(n, key))
            row.update(metrics)
            row["validation_seconds"] = time.perf_counter() - started
            row["valid"] = bool(row["valid"] and completed)
            row["status"] = "ok" if row["valid"] else "incorrect"
        except Exception as exc:
            row.update(status="query_limit" if str(exc).startswith("query_limit:") else "error",
                       valid=False, message=f"{type(exc).__name__}: {exc}",
                       failed_phase=json.loads((folder / "phase.json").read_text())["phase"],
                       failed_phase_seconds=getattr(exc, "benchmark_seconds", None))
            field = {"setup": "setup_seconds", "application": "apply_seconds",
                     "input_preparation": "preparation_seconds", "output_formatting": "output_seconds"}.get(row["failed_phase"])
            if field:
                row[field] = None
        if oracle is not None:
            for name in ("query_count", "row_queries", "entry_queries", "returned_entries"):
                row[name] = getattr(oracle, name)
        if not warmup:
            with samples_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            (folder / "pending.json").unlink(missing_ok=True)
        return row

    for warmup in range(cfg["warmup"]):
        one_sample(-warmup - 2, warmup=True)
    for repeat in range(cfg["repeats"]):
        one_sample(repeat)
    phase("finished")


def environment():
    import contextlib
    import io
    import numpy as np
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "versions": versions, "numpy_config": stream.getvalue(),
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARS},
            "timer": "time.perf_counter_ns", "command": sys.argv}


def run_job(job, folder):
    """Enforce a hard wall/RSS budget, even inside a long NumPy/native call."""
    import psutil
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "samples.jsonl").exists() and not (folder / "result.json").exists():
        archive = folder / "previous_attempts" / str(time.time_ns())
        archive.mkdir(parents=True)
        for name in ("samples.jsonl", "worker.log", "phase.json", "pending.json", "job.json"):
            if (folder / name).exists():
                (folder / name).replace(archive / name)
    write_json(folder / "job.json", job)
    start = time.monotonic()
    peak = 0
    terminal = "finished"
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "pauli_algorithm_comparison.benchmark",
                                    "--worker", str(folder / "job.json")], cwd=ROOT.parent,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, monitored.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > job["config"]["max_memory_mb"] * 1024 ** 2:
                    terminal = "memory_limit"
                    process.kill()
                elif time.monotonic() - start > job["config"]["timeout"]:
                    terminal = "timeout"
                    process.kill()
                time.sleep(0.05)
            process.wait()
        except KeyboardInterrupt:
            process.kill()
            process.wait()
            raise
    rows = read_jsonl(folder / "samples.jsonl")
    phase = json.loads((folder / "phase.json").read_text())["phase"] if (folder / "phase.json").exists() else "startup"
    if terminal != "finished" or process.returncode != 0:
        pending = folder / "pending.json"
        row = json.loads(pending.read_text()) if pending.exists() else base_row(job)
        row.setdefault("repeat", -1)
        row.update(status=terminal if terminal != "finished" else "worker_error", valid=False,
                   message=f"worker stopped during {phase}; see jobs/{job['job_id']}/worker.log")
        rows.append(row)
    for row in rows:
        row["worker_peak_rss_bytes"] = peak
    write_json(folder / "result.json", rows)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison", choices=METHODS, default="anschuetz")
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--n-step", type=int, default=2)
    p.add_argument("--n", type=int, nargs="+", help="explicit qubit counts, overriding the range")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120, help="hard seconds per method/input/cache-mode worker")
    p.add_argument("--max-memory-mb", type=float, default=1024, help="allocation guard and absolute worker RSS limit")
    p.add_argument("--max-pauli-terms", type=int, default=2000, help="Spencer expanded-support guard")
    p.add_argument("--max-queries", type=int, default=2000000, help="Spencer query budget per call")
    p.add_argument("--localities", type=int, nargs="+", default=None)
    p.add_argument("--cache-modes", nargs="+", choices=("cold", "cached"), default=None)
    p.add_argument("--output-contract", choices=("compressed", "expanded"), default="compressed",
                   help="Georges comparison output; other comparisons have fixed common outputs")
    p.add_argument("--anschuetz-implementations", nargs="+",
                   choices=("public", "optimized", "tensor_network"),
                   default=["public", "optimized", "tensor_network"],
                   help="F construction methods; tensor_network needs no external source")
    public_source = ROOT / "_external/symmetric_hamiltonians"
    legacy_source = ROOT.parent / "pauli_symmetry_bench/benchmarks/_external/symmetric_hamiltonians"
    if not public_source.exists() and legacy_source.exists():
        public_source = legacy_source
    p.add_argument("--anschuetz-source", default=str(public_source))
    p.add_argument("--dense-reference-max-n", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", action="store_true", help="resume identical settings and source hashes")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.worker:
        worker(args.worker)
        return 0
    import math
    if ((not args.n and (args.n_step <= 0 or args.n_max < args.n_min or args.n_min < 1)) or args.seed < 0 or
            min(args.repeats, args.instances, args.threads, args.max_pauli_terms, args.max_queries) < 1 or args.warmup < 0):
        p.error("require positive counts, nonnegative seed/warmup, and a valid qubit range")
    if (not all(math.isfinite(v) for v in (args.timeout, args.max_memory_mb, args.atol, args.rtol, args.tol, args.delta))
            or min(args.timeout, args.max_memory_mb) <= 0 or min(args.atol, args.rtol, args.tol) < 0
            or not 0 < args.delta < 1 or args.dense_reference_max_n < 0):
        p.error("invalid timeout, memory budget, tolerance, delta, or reference cutoff")
    n_values = sorted(set(args.n or range(args.n_min, args.n_max + 1, args.n_step)))
    if not n_values or min(n_values) < 1:
        p.error("provide at least one positive qubit count")
    localities = sorted(set(args.localities or ([2, 4] if args.comparison == "chang" else [1])))
    if min(localities) < 0:
        p.error("locality must be nonnegative")
    modes = list(dict.fromkeys(args.cache_modes or (["cold", "cached"] if args.comparison in ("anschuetz", "chang") else ["cold"])))
    for name in THREAD_VARS:
        os.environ[name] = str(args.threads)
    from .benchmark_cases import encode_mapping, make_case, pauli_orbit_size
    from . import anschuetz_public, separated
    from .plot_benchmarks import summarize, plot
    methods = list(METHODS[args.comparison])
    if args.comparison == "anschuetz":
        methods = ["separated"] + ["anschuetz_" + value for value in dict.fromkeys(args.anschuetz_implementations)]
    output = (args.output_dir or ROOT / "results" / (args.comparison + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    cfg = {key: value for key, value in vars(args).items() if key not in ("output_dir", "resume", "worker", "no_plots")}
    cfg.update(n_values=n_values, localities=localities, cache_modes=modes, methods=methods,
               anschuetz_source=str(Path(args.anschuetz_source).expanduser().resolve()), schema="thesis-comparison-v1")
    hashes = {path.name: sha256(path) for path in sorted(ROOT.glob("*.py"))}
    public_path = Path(cfg["anschuetz_source"])
    public_path = public_path / "utils.py" if public_path.is_dir() else public_path
    hashes["public_anschuetz_utils.py"] = sha256(public_path) if public_path.is_file() else None
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            p.error(f"output directory is not empty: {output}; use a new directory or --resume")
        if (json.loads((output / "config.json").read_text()) != cfg or
                json.loads((output / "source_hashes.json").read_text()) != hashes):
            p.error("resume requires identical configuration and source hashes")
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    (output / "jobs").mkdir(exist_ok=True)
    write_json(output / "config.json", cfg)
    write_json(output / "source_hashes.json", hashes)
    current_environment = environment()
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", current_environment)
    else:
        previous_environment = json.loads((output / "environment.json").read_text())
        for key in ("python", "platform", "cpu", "versions", "numpy_config", "thread_environment"):
            if previous_environment[key] != current_environment[key]:
                p.error(f"resume environment differs in {key}; use a new output directory")
    family = {"anschuetz": "general", "chang": "fixed_weight", "spencer": "local", "georges": "general"}[args.comparison]
    all_rows = []
    # Reproducible ordering mitigates always running the thesis method first.
    import random
    order = random.Random(args.seed)
    for n in n_values:
        for locality in (localities if family != "general" else [0]):
            for instance in range(args.instances):
                input_seed = args.seed + 1009 * n + 104729 * instance + 37 * locality
                case_id = f"n{n}_{family}_l{locality}_i{instance}"
                input_path = output / "inputs" / (case_id + ".json")
                # Guard the parent too: it holds the polynomial input table.
                guarded = None
                if 512 * math.comb(n + 3, 3) > args.max_memory_mb * 1024 ** 2:
                    guarded = "input_memory_guard"
                elif locality > n:
                    guarded = "inapplicable_locality"
                coefficients = None
                if not guarded:
                    if input_path.exists():
                        from .benchmark_cases import decode_mapping
                        coefficients = decode_mapping(json.loads(input_path.read_text())["coefficients"])
                    else:
                        started = time.perf_counter()
                        coefficients = make_case(n, family, input_seed, locality)
                        record = {"n": n, "family": family, "locality": locality, "seed": input_seed,
                                  "normalization": "common coefficient of each individual Pauli word",
                                  "coefficients": encode_mapping(coefficients)}
                        if args.comparison in ("spencer", "georges"):
                            record["entries"] = encode_mapping(separated.inverse(n, coefficients))
                        record["generation_seconds"] = time.perf_counter() - started
                        write_json(input_path, record)
                support = sum(pauli_orbit_size(n, key) for key, value in (coefficients or {}).items() if value)
                jobs = [(method, mode) for method in methods for mode in modes]
                order.shuffle(jobs)
                for method, mode in jobs:
                    job_id = f"{case_id}_{method}_{mode}"
                    folder = output / "jobs" / job_id
                    contract = {"anschuetz": "dense_schur_blocks", "chang": "banded_schur_CSR",
                                "spencer": "expanded_sparse_pauli_dictionary",
                                "georges": "expanded_pauli_array" if args.output_contract == "expanded" else "compressed_pauli_orbits"}[args.comparison]
                    job = dict(job_id=job_id, comparison=args.comparison, method=method, n=n,
                               family=family, locality=locality, instance=instance, input_seed=input_seed,
                               cache_mode=mode, input_path=str(input_path), pauli_support=support,
                               input_hash=sha256(input_path) if input_path.exists() else None,
                               output_contract=contract, config=cfg)
                    problem = guarded
                    if not problem and estimate_memory(n, method, args.comparison, args.output_contract == "expanded") > args.max_memory_mb * 1024 ** 2:
                        problem = "allocation_guard"
                    if args.comparison == "spencer" and support > args.max_pauli_terms:
                        problem = "pauli_support_guard"
                    if method == "anschuetz_public" and not public_path.is_file():
                        problem = "missing_public_source"
                    if (folder / "result.json").exists() and args.resume:
                        rows = json.loads((folder / "result.json").read_text())
                        if any(row.get("input_hash") != job["input_hash"] for row in rows):
                            p.error(f"saved input changed for {job_id}; use a new output directory")
                        print(f"resume {job_id}", flush=True)
                    elif problem:
                        row = base_row(job)
                        row.update(repeat=-1, status=problem, valid=False, message="see configured limits and BENCHMARKS.md")
                        rows = [row]
                        folder.mkdir(parents=True, exist_ok=True)
                        write_json(folder / "result.json", rows)
                        print(f"skip {job_id}: {problem}", flush=True)
                    else:
                        print(f"run {job_id}", flush=True)
                        rows = run_job(job, folder)
                        print("  " + ", ".join(sorted({row["status"] for row in rows})), flush=True)
                    all_rows.extend(rows)
                    with (output / "runs.jsonl").open("w") as stream:
                        for row in all_rows:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    write_csv(output / "runs.csv", all_rows)
                    write_csv(output / "summary.csv", summarize(all_rows))
    if not args.no_plots:
        plot(output)
    print(f"Results: {output}", flush=True)
    return 0 if any(row.get("valid") for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
