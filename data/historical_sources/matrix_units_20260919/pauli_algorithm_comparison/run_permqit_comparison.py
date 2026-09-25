"""Standalone matrix-unit -> Schur-block campaign; no thesis exports.

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
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

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
    elif method in ('factorial_float64', 'factorial_longdouble'):
        dtype = np.longdouble if method.endswith('longdouble') else np.float64
        prep = lambda n: schur_factorial.prepare(n, dtype=dtype)
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


def environment_and_sources(destination):
    import numpy as np
    import scipy
    import permqit
    from .permqit_adapter import UPSTREAM_COMMIT
    upstream = ROOT/'_external/permqit'
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError('unexpected upstream checkout commit')
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)
    if dirty:
        raise RuntimeError('upstream checkout is modified')
    installed = Path(permqit.__file__).parent
    upstream_files = list((upstream/'src/permqit').rglob('*.py'))
    mismatches = [str(p.relative_to(upstream/'src/permqit')) for p in upstream_files
                  if not (installed/p.relative_to(upstream/'src/permqit')).exists()
                  or sha(p) != sha(installed/p.relative_to(upstream/'src/permqit'))]
    if mismatches:
        raise RuntimeError(f'installed source differs: {mismatches}')
    source_names = ('permqit_adapter.py', 'permqit_comparison_reference.py',
                    'run_permqit_comparison.py', 'test_permqit_adapter.py',
                    'schur_factorial.py', 'schur_hahn.py', 'common.py', 'schur_full_ed.py',
                    'requirements-permqit.txt')
    snapshots = destination/'sources'
    snapshots.mkdir()
    for name in source_names:
        (snapshots/name).write_bytes((ROOT/name).read_bytes())
    # Preserve the exact unmodified upstream source and license for this pilot.
    subprocess.run(['git', '-C', str(upstream), 'archive', '--format=tar.gz',
                    '-o', str(snapshots/'permqit-upstream.tar.gz'), commit], check=True)
    cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')), platform.processor())
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (destination/'requirements.lock').write_text(freeze)
    return {'date_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'executable': sys.executable, 'platform': platform.platform(), 'cpu': cpu,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'permqit': importlib.metadata.version('permqit'), 'upstream_commit': commit,
            'installed_source_verified': True, 'verified_upstream_python_files': len(upstream_files),
            'threads': {name: os.environ.get(name) for name in THREADS},
            'source_sha256': {name: sha(ROOT/name) for name in source_names},
            'upstream_python_sha256': {str(p.relative_to(upstream)): sha(p) for p in upstream_files},
            'longdouble_mantissa_bits': np.finfo(np.longdouble).nmant+1}


def campaign(args):
    from .permqit_comparison_reference import random_entries
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'workers', 'logs'):
        (out/name).mkdir()
    meta = environment_and_sources(out)
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
            coefficients = random_entries(n, seed)
            path = out/'inputs'/f'n{n}_trial{trial}.json'
            dump(path, {'n': n, 'trial': trial, 'seed': seed,
                        'entries': [[*key, z.real, z.imag] for key, z in coefficients.items()]})
            for method in args.methods:
                jobs.append((n, trial, method, path))
    random.Random(args.seed).shuffle(jobs)
    rows = []
    for i, (n, trial, method, path) in enumerate(jobs, 1):
        name = f'n{n}_{method}_trial{trial}'
        result = out/'workers'/f'{name}.json'
        command = [sys.executable, '-m', 'pauli_algorithm_comparison.run_permqit_comparison',
                   '--worker', method, '--input', str(path), '--output', str(result),
                   '--app-repeats', str(args.app_repeats)]
        with (out/'logs'/f'{name}.log').open('w') as log:
            try:
                process = subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False)
                if result.exists():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    parser.add_argument('--sizes', type=int, nargs='+', default=[2,4,6,8,10,12,16,20])
    parser.add_argument('--methods', nargs='+', choices=(*METHODS, 'factorial_longdouble'), default=list(METHODS))
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--app-repeats', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--worker', choices=(*METHODS, 'factorial_longdouble'))
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.app_repeats < 1 or args.trials < 1 or any(n < 1 for n in args.sizes):
        parser.error('sizes, trials and app-repeats must be positive')
    if not args.worker and not args.out:
        parser.error('--out is required for a campaign')
    for name in THREADS:
        os.environ[name] = '1'
    os.environ['PERMQIT_USE_GPU'] = 'false'
    (worker if args.worker else campaign)(args)


if __name__ == '__main__':
    main()
