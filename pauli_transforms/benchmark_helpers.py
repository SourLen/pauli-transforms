"""Small file and environment helpers shared by the benchmark commands."""

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def read_jsonl(path, *, interrupted=False):
    rows = []
    lines = Path(path).read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if interrupted and index == len(lines)-1 and not line.endswith("\n"):
                break
            raise
    return rows


def environment_record():
    from threadpoolctl import threadpool_info
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return dict(utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                platform=platform.platform(), cpu=cpu,
                versions={p: version(p) for p in ("numpy", "scipy", "matplotlib", "threadpoolctl")},
                threadpools=threadpool_info(), command=sys.argv,
                thread_environment={k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS")})


def source_snapshot(output):
    """Record hashes of the code used for this run; the repository supplies sources."""
    hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(Path(output) / "source_hashes.json", hashes)
    return hashes
