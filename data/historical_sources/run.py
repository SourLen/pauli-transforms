#!/usr/bin/env python3
"""Verify and run a selected historical driver in an isolated temporary package.

Example:
  python data/historical_sources/run.py conversion_20260921 -- \
    --comparison anschuetz --anschuetz-implementations optimized --n 2 \
    --instances 1 --repeats 1 --warmup 0 --no-plots --output-dir /tmp/new-run

Arguments after -- belong to the original driver. Timing outputs are new
measurements and must be kept separate from the retained thesis data.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def verify(campaigns):
    checked = 0
    for name, record in campaigns.items():
        folder = ROOT / name / "pauli_algorithm_comparison"
        hashes = dict(record["files"])
        hashes.update({path: info["sha256"] for path, info in
                       record["supplemental_dependencies"].items()})
        hashes["__init__.py"] = record["package_marker_sha256"]
        for path, expected in hashes.items():
            target = folder / path
            if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"historical source hash mismatch: {name}/{path}")
            checked += 1
    return checked


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for path, expected in manifest.get("adapters", {}).items():
        target = ROOT / path
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"archive adapter hash mismatch: {path}")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign", nargs="?", choices=manifest["campaigns"])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--permqit-source", type=Path,
                        help="clean pinned permqit checkout, needed by its original driver")
    args, driver_args = parser.parse_known_args()
    if driver_args[:1] == ["--"]:
        driver_args = driver_args[1:]
    selected = ({args.campaign: manifest["campaigns"][args.campaign]}
                if args.campaign else manifest["campaigns"])
    checked = verify(selected)
    if args.verify:
        print(json.dumps({"verified_files": checked, "campaigns": list(selected)}, indent=2))
        return
    if not args.campaign:
        parser.error("select a campaign, or use --verify")
    record = selected[args.campaign]
    if record["runner"] == "benchmark":
        if any(value in driver_args for value in ("tensor_network", "spencer", "georges", "public")):
            parser.error("this archive supports only the selected Anschuetz and Chang conversions")
        if "--comparison" not in driver_args and not any(value in driver_args for value in ("-h", "--help")):
            parser.error("state --comparison explicitly for the original generic driver")
        if "anschuetz" in driver_args and "--anschuetz-implementations" not in driver_args:
            parser.error("select --anschuetz-implementations optimized and/or public_original explicitly")
    with tempfile.TemporaryDirectory(prefix="pauli-historical-") as temporary:
        runtime = Path(temporary)
        package = runtime / "pauli_algorithm_comparison"
        shutil.copytree(ROOT / args.campaign / "pauli_algorithm_comparison", package)
        if args.permqit_source:
            source = args.permqit_source.resolve(strict=True)
            (package / "_external").mkdir()
            (package / "_external" / "permqit").symlink_to(source, target_is_directory=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "adapters"), str(runtime)])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-m", "pauli_algorithm_comparison." + record["runner"], *driver_args]
        completed = subprocess.run(command, env=env)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
