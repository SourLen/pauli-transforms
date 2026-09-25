#!/usr/bin/env python3
"""Exhaustive small-system check of the published Muñoz coordinate dictionary."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np

from test_munoz_dictionary import check_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_n <= 6:
        parser.error("max-n must be between 1 and 6 for this bounded check")
    inputs = ["scripts/check_munoz_dictionary.py", "tests/test_munoz_dictionary.py",
              "pauli_transforms/krawtchouk.py", "pauli_transforms/common.py"]
    rows = [row for n in range(1, args.max_n+1) for row in check_size(n)]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_publication": "https://doi.org/10.1007/s11128-018-2045-0",
        "source_equations": [14, 17, 41, 42, 43],
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "measured_source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in inputs},
        "python": platform.python_version(), "numpy": np.__version__,
        "sizes": list(range(1, args.max_n+1)),
        "input_family": "every unnormalized Pauli orbit, coefficient one",
        "direction": "Pauli orbits to literal matrix entries",
        "reference": "literal tensor products and independent binary sums",
        "reference_precision": "exact integer binomials and Gaussian integers representable in complex128",
        "backend": "Krawtchouk, supplied tables and fresh sparse path",
        "dtype": "complex128", "seed": None,
        "tolerance": {"absolute": 0, "relative": 0},
        "orbit_count": len(rows),
        "matrix_entries_checked": sum(row["matrix_entries"] for row in rows),
        "f_representatives_checked": sum(row["f_orbit_representatives"] for row in rows),
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "observations": rows},
                                      indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
