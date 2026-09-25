#!/usr/bin/env python3
"""Regenerate the retained Ising numerical values with archived kernels.

This is a new verification command, not the original measurement/plot driver.
It writes new diagnostics and does not modify the retained observations.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parent
from run import verify

manifest = json.loads((ROOT / "manifest.json").read_text())
verify({"ising_20260918": manifest["campaigns"]["ising_20260918"]})
sys.path.insert(0, str(ROOT / "ising_20260918"))

import numpy as np
from threadpoolctl import threadpool_limits
from pauli_algorithm_comparison import physical_models as pm, schur_separated as conversion
from pauli_algorithm_comparison.plot_ising_example import spectrum_and_small_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = ROOT.parents[1] / "data/thesis/ising"
    config = json.loads((data / "config.json").read_text())
    with (data / "selected_curves.csv").open(newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    rows = []
    with threadpool_limits(limits=1):
        example, small_checks = spectrum_and_small_checks(config["spectrum_n"])
        with np.load(data / "spectrum.npz") as saved:
            spectrum_error = max(float(np.max(np.abs(example[key]-saved[key])))
                                 for key in saved.files if key.startswith("values_"))
        for n in config["n"]:
            selected = sorted((row for row in saved_rows if int(row["n"]) == n),
                              key=lambda row: float(row["time"]))
            times = np.array([float(row["time"]) for row in selected])
            assert len(times) == config["points"]
            np.testing.assert_array_equal(times, np.linspace(0, config["tmax"], config["points"]))
            saved = np.array([complex(float(row["value_real"]), float(row["value_imag"]))
                              for row in selected])
            tables = conversion.prepare(n, backend="hahn")
            ham = conversion.pauli_to_schur(n, pm.ising_pauli(n, config["g"], config["h"]), tables)
            state = conversion.pauli_to_schur(n, pm.product_x_pauli(n, config["p"]), tables)
            observable = conversion.pauli_to_schur(n, {(1, 0, 0): 1/n}, tables)
            spectrum = pm.diagonalize_blocks(ham)
            values = pm.expectation_time_series(n, spectrum, state, observable, times)
            error = float(np.max(np.abs(values-saved)))
            rows.append({"n": n, "points": len(times), "max_absolute_error": error,
                         "passed": error <= 1e-10})
    result = {"kind": "regeneration with archived kernels, not new timing data",
              "python": platform.python_version(), "numpy": np.__version__,
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "archive_manifest_sha256": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
              "dtype": "complex128/float64", "absolute_tolerance": 1e-10,
              "spectrum_n": config["spectrum_n"], "spectrum_max_absolute_error": spectrum_error,
              "small_checks": small_checks, "curves": rows,
              "all_passed": spectrum_error <= 1e-10 and all(row["passed"] for row in rows)
                            and all(row["valid"] for row in small_checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
