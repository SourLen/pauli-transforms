"""Compare matrix-unit -> Schur-block conversion, as in Appendix C.2.

Each preparation sample runs in a new subprocess, excluding Python/package
imports, input loading, validation and persistence. All algorithms receive the
same dictionary of literal orbit entries. No eigensolver is timed.
"""
import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from math import isfinite
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

from .benchmark_helpers import source_snapshot

ROOT = Path(__file__).resolve().parent
METHODS = ('permqit', 'factorial_float64', 'hahn_float64')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    # Import every timed implementation before either timer, including native
    # permqit: its constructor otherwise imports substantial dependencies.
    if args.worker == 'permqit':
        from permqit.representation.isomorphism import EndSnAlgebraIsomorphism
        from permqit.representation.young_tableau import SSYT
    from . import permqit_adapter, schur_factorial, schur_hahn
    from .permqit_comparison_reference import integer_kernel_blocks, block_errors
    source = json.loads(Path(args.input).read_text())
    n = source['n']
    coefficients = {tuple(row[:3]): complex(*row[3:]) for row in source['entries']}
    method = args.worker
    if method == 'permqit':
        prep, apply = permqit_adapter.prepare, permqit_adapter.orbit_to_schur
    elif method == 'factorial_float64':
        prep = schur_factorial.prepare
        apply = schur_factorial.orbit_to_schur
    elif method == 'hahn_float64':
        prep, apply = schur_hahn.prepare, schur_hahn.orbit_to_schur
    else:
        raise ValueError(method)
    row = {'method': method, 'n': n, 'trial': source['trial'], 'seed': source['seed'],
           'input_sha256': sha(args.input), 'status': 'error'}
    try:
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            start = time.perf_counter_ns()
            prepared = prep(n)
            prep_s = (time.perf_counter_ns() - start)*1e-9
            start = time.perf_counter_ns()
            blocks = apply(n, coefficients, prepared)
            first_s = (time.perf_counter_ns() - start)*1e-9
            # One first application above; independently retained warm timings.
            warm = []
            for _ in range(args.app_repeats):
                start = time.perf_counter_ns()
                repeated = apply(n, coefficients, prepared)
                warm.append((time.perf_counter_ns() - start)*1e-9)
            reference = integer_kernel_blocks(n, coefficients)
            errors = block_errors(n, blocks, reference)
            repeated_errors = block_errors(n, repeated, reference)
            if method == 'permqit':
                numeric_bytes = prepared.numeric_map_bytes
                nnz = prepared.nnz
            elif method.startswith('factorial'):
                numeric_bytes = sum(getattr(prepared, field.name).nbytes
                                    for field in fields(prepared)
                                    if isinstance(getattr(prepared, field.name), np.ndarray))
                nnz = None
            else:
                numeric_bytes = sum(matrix.nbytes for _, matrix in prepared.values())
                nnz = None
            row.update(status='ok' if errors['passed'] and repeated_errors['passed'] else 'accuracy_failed',
                       preparation_s=prep_s, first_application_s=first_s,
                       fresh_total_s=prep_s+first_s, warm_application_s=warm,
                       warm_median_s=statistics.median(warm), errors=errors,
                       repeated_errors=repeated_errors, numeric_table_bytes=numeric_bytes,
                       normalized_map_nnz=nnz, threadpools=pools)
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    dump(args.output, row)


def verified_upstream(upstream):
    """Check the installed native source against the clean, pinned checkout."""
    import permqit
    from .permqit_adapter import UPSTREAM_COMMIT
    upstream = Path(upstream).resolve()
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError('unexpected upstream checkout commit')
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)
    if dirty:
        raise RuntimeError('upstream checkout is modified')
    installed = Path(permqit.__file__).parent
    source_root = upstream/'src/permqit'
    upstream_files = sorted(source_root.rglob('*.py'))
    if not upstream_files or {p.relative_to(installed) for p in installed.rglob('*.py')} != {
            p.relative_to(source_root) for p in upstream_files}:
        raise RuntimeError('installed permqit source file set differs from the pinned checkout')
    mismatches = [str(p.relative_to(source_root)) for p in upstream_files
                  if sha(p) != sha(installed/p.relative_to(source_root))]
    if mismatches:
        raise RuntimeError(f'installed source differs: {mismatches}')
    return {'permqit': importlib.metadata.version('permqit'), 'upstream_commit': commit,
            'installed_source_verified': True, 'verified_upstream_python_files': len(upstream_files),
            'upstream_python_sha256': {str(p.relative_to(upstream)): sha(p) for p in upstream_files}}


def environment_and_sources(destination, args):
    import numpy as np
    import scipy
    upstream = verified_upstream(args.permqit_source) if 'permqit' in args.methods else {}
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (destination/'requirements.lock').write_text(freeze)
    return {'date_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'executable': sys.executable, 'platform': platform.platform(), 'cpu': cpu,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'threads': {name: os.environ.get(name) for name in THREADS},
            'source_sha256': source_snapshot(destination),
            'longdouble_mantissa_bits': np.finfo(np.longdouble).nmant+1, **upstream}


def campaign(args):
    from .permqit_comparison_reference import random_entries
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('inputs', 'workers', 'logs'):
        (out/name).mkdir()
    meta = environment_and_sources(out, args)
    meta.update(sizes=args.sizes, methods=args.methods, trials=args.trials, seed=args.seed,
                app_repeats=args.app_repeats, timeout_s=args.timeout,
                input_family='complex Gaussian coefficients in HS-normalized matrix-unit orbit basis; HS norm one',
                reference='exact-integer kernel sums with long-double final roots and accumulation',
                gate={'weighted_hs_relative': 1e-8, 'entrywise_atol': 1e-9, 'entrywise_rtol': 1e-8},
                timing='fresh subprocess per preparation; imports/input/reference excluded; packing/unpacking included',
                memory='numeric buffers of retained factors/kernels; permqit includes raw+normalized CSR, not Python objects or peak RSS')
    dump(out/'manifest.json', meta)
    jobs = []
    for n in args.sizes:
        for trial in range(args.trials):
            seed = args.seed + 1000*n + trial
            path = out/'inputs'/f'n{n}_trial{trial}.json'
            if args.input_directory:
                from .permqit_comparison_reference import keys
                import numpy as np
                folder = Path(args.input_directory)
                folder = folder/'inputs' if (folder/'inputs').is_dir() else folder
                archived = folder/path.name
                payload = json.loads(archived.read_text())
                entries = payload['entries']
                if (payload['n'] != n or payload['trial'] != trial
                        or len(entries) != len(keys(n))
                        or any(len(row) != 5 or any(type(x) is not int for x in row[:3]) for row in entries)
                        or {tuple(row[:3]) for row in entries} != set(keys(n))
                        or not np.isfinite(np.asarray(entries, dtype=float)).all()):
                    raise ValueError(f'invalid archived input: {archived}')
                path.write_bytes(archived.read_bytes())
            else:
                coefficients = random_entries(n, seed)
                dump(path, {'n': n, 'trial': trial, 'seed': seed,
                            'entries': [[*key, z.real, z.imag] for key, z in coefficients.items()]})
            for method in args.methods:
                jobs.append((n, trial, method, path))
    random.Random(args.seed).shuffle(jobs)
    rows = []
    for i, (n, trial, method, path) in enumerate(jobs, 1):
        name = f'n{n}_{method}_trial{trial}'
        result = out/'workers'/f'{name}.json'
        command = [sys.executable, '-m', 'pauli_transforms.run_permqit_comparison',
                   '--worker', method, '--input', str(path), '--output', str(result),
                   '--app-repeats', str(args.app_repeats)]
        with (out/'logs'/f'{name}.log').open('w') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False)
                if result.exists() and process.returncode == 0:
                    row = json.loads(result.read_text())
                else:
                    row = {'n': n, 'trial': trial, 'method': method, 'status': 'worker_failed',
                           'returncode': process.returncode}
            except subprocess.TimeoutExpired:
                row = {'n': n, 'trial': trial, 'method': method, 'status': 'timeout'}
        rows.append(row)
        with (out/'measurements.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        print(f'{i}/{len(jobs)} {name}: {row["status"]}', flush=True)
    summary = []
    for n in args.sizes:
        for method in args.methods:
            group = [r for r in rows if r['n'] == n and r['method'] == method]
            measured = [r for r in group if 'fresh_total_s' in r]
            row = {'n': n, 'method': method, 'passed': sum(r['status'] == 'ok' for r in group),
                   'trials': len(group)}
            for metric in ('preparation_s', 'first_application_s', 'fresh_total_s', 'warm_median_s'):
                if measured:
                    row[metric] = statistics.median(r[metric] for r in measured)
            if measured:
                row.update(max_weighted_hs_relative=max(r['errors']['weighted_hs_relative'] for r in measured),
                           numeric_table_bytes=measured[0]['numeric_table_bytes'])
            summary.append(row)
    dump(out/'summary.json', summary)
    with (out/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in summary for k in row)))
        writer.writeheader()
        writer.writerows(summary)
    changed = [name for name, digest in meta['source_sha256'].items() if sha(ROOT/name) != digest]
    dump(out/'integrity_after.json', {'changed_sources': changed, 'unchanged': not changed})
    if changed:
        raise RuntimeError(f'sources changed during run: {changed}')
    if 'permqit' in args.methods:
        upstream_after = verified_upstream(args.permqit_source)
        if upstream_after != {key: meta[key] for key in upstream_after}:
            raise RuntimeError('upstream sources changed during run')
    if any(row['status'] != 'ok' for row in rows):
        raise RuntimeError('campaign contains failed trials; see measurements.jsonl and logs/')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[2,4,6,8,10,12,16,20])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--app-repeats', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--permqit-source', type=Path, default=Path('_external/permqit'))
    parser.add_argument('--input-directory', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--input', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke:
        args.sizes, args.trials, args.app_repeats = [2, 4], 1, 2
    if args.app_repeats < 1 or args.trials < 1 or any(n < 1 for n in args.sizes):
        parser.error('sizes, trials and app-repeats must be positive')
    if args.timeout <= 0 or not isfinite(args.timeout):
        parser.error('--timeout must be positive and finite')
    if len(set(args.methods)) != len(args.methods) or len(set(args.sizes)) != len(args.sizes):
        parser.error('methods and sizes must not contain duplicates')
    if args.worker and not args.input:
        parser.error('--input is required for a worker')
    for name in THREADS:
        os.environ[name] = '1'
    os.environ['PERMQIT_USE_GPU'] = 'false'
    (worker if args.worker else campaign)(args)


if __name__ == '__main__':
    main()
