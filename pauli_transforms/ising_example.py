"""Regenerate the thesis Ising spectrum and transverse-magnetization curves."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import physical_models as pm, transforms
from .benchmark_helpers import environment_record, source_snapshot, write_csv, write_json
from .random_dynamics_inputs import relative_blocks, require_valid, state_diagnostics


def compute(n, times, *, g=1., h=.5, p=.6):
    """Convert Pauli inputs and independently check the collective-spin answer."""
    tables = transforms.prepare(n, backend="hahn")
    inputs = (pm.ising_pauli(n, g, h), pm.product_x_pauli(n, p), {(1, 0, 0): 1 / n})
    blocks = tuple(transforms.pauli_to_schur(n, mapping, tables) for mapping in inputs)
    references = (pm.ising_collective_blocks(n, g, h), pm.product_x_collective_blocks(n, p),
                  pm.collective_observable_blocks(n, "x"))
    spectrum = pm.diagonalize_blocks(blocks[0])
    reference_spectrum = pm.diagonalize_blocks(references[0])
    values = pm.expectation_time_series(n, spectrum, blocks[1], blocks[2], times)
    expected = pm.expectation_time_series(n, reference_spectrum, references[1], references[2], times)
    checks = {name + "_block_error": relative_blocks(actual, reference)
              for name, actual, reference in zip(("h", "state", "observable"), blocks, references, strict=True)}
    checks.update(state_diagnostics(n, blocks[1]))
    checks["curve_error"] = float(np.max(np.abs(values - expected)))
    checks["spectrum_error"] = max(float(np.max(np.abs(a - b))) / max(1., float(np.max(np.abs(b))))
                                   for a, b in zip(spectrum.energies, reference_spectrum.energies, strict=True))
    require_valid(checks, 1e-8, f"Ising n={n}")
    return spectrum, values, expected, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[8, 20, 40])
    parser.add_argument("--spectrum-n", type=int, default=12)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--tmax", type=float, default=12.)
    parser.add_argument("--g", type=float, default=1.)
    parser.add_argument("--h", type=float, default=.5)
    parser.add_argument("--p", type=float, default=.6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.n, args.spectrum_n, args.points = [2, 3], 3, 5
    if min(args.n + [args.spectrum_n]) < 2 or args.points < 1 or not np.isfinite(args.tmax) or args.tmax <= 0:
        parser.error("n>=2, positive points and finite positive tmax required")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", dict(vars(args), output=str(out), threads=1,
                                        model="H=-(g sum_{i<j} ZiZj+h sum_i Xi)/n; rho=[(I+pX)/2]^tensor(n); M=sum_i Xi/n"))
    hashes = source_snapshot(out)
    rows, checks = [], []
    times = np.linspace(0., args.tmax, args.points)
    with threadpool_limits(limits=1):
        write_json(out / "environment.json", dict(environment_record(), source_sha256=hashes))
        for n in sorted(set(args.n + [args.spectrum_n])):
            print(f"Ising n={n}", flush=True)
            spectrum, values, expected, diagnostics = compute(n, times, g=args.g, h=args.h, p=args.p)
            checks.append(dict(n=n, **diagnostics))
            if n == args.spectrum_n:
                np.savez(out / "spectrum.npz", n=n,
                         **{f"values_{k}": energy for k, energy in enumerate(spectrum.energies)})
            if n in args.n:
                rows.extend(dict(n=n, time=float(t), value_real=float(v.real), value_imag=float(v.imag),
                                 reference_real=float(r.real), reference_imag=float(r.imag))
                            for t, v, r in zip(times, values, expected, strict=True))
    write_csv(out / "selected_curves.csv", rows)
    write_json(out / "validation.json", dict(all_valid=True, tolerance=1e-8, checks=checks))


if __name__ == "__main__":
    main()
