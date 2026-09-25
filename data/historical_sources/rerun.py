#!/usr/bin/env python3
"""Rerun a retained timing configuration with its selected archived sources.

Use --smoke for one small input and one repetition per selected method.
Use --show to print the command without running measurements.
"""

import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATASETS = {
    "direct": ("direct_20260917", "direct/config.json"),
    "fixed_locality": ("fixed_locality_20260914", "fixed_locality/config.json"),
    "current/direct-baseline": ("direct_20260917", "current/direct/baseline_config.json"),
    "current/direct-extension": ("conversion_20260921", "current/direct/extension_config.json"),
    "current/fixed_locality/low": ("conversion_20260921", "current/fixed_locality/low/config.json"),
    "current/fixed_locality/high": ("conversion_20260921", "current/fixed_locality/high/config.json"),
    "current/fixed_weight_cache/low": ("fixed_weight_cache_20260923", "current/fixed_weight_cache/low/config.json"),
    "current/fixed_weight_cache/high": ("fixed_weight_cache_20260923", "current/fixed_weight_cache/high/config.json"),
    "spectral": ("spectral_20260917", "spectral/config.json"),
    "dynamics": ("dynamics_20260918", "dynamics/config.json"),
    "matrix_units": ("matrix_units_20260919", "matrix_units/manifest.json"),
}


def declared_options(source):
    """Read the original CLI declarations without importing algorithm code."""
    options = {}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            values = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            for option in values:
                if isinstance(option, str) and option.startswith("--"):
                    action = next((kw.value.value for kw in node.keywords
                                   if kw.arg == "action" and isinstance(kw.value, ast.Constant)), None)
                    options[option[2:].replace("-", "_")] = option, action
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anschuetz-source", type=Path)
    parser.add_argument("--permqit-source", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    campaign, config_path = DATASETS[args.dataset]
    cfg = json.loads((REPO / "data/thesis" / config_path).read_text())
    metadata = json.loads((ROOT / "manifest.json").read_text())["campaigns"][campaign]
    runner = metadata["runner"]
    declared = declared_options((ROOT / campaign / "pauli_algorithm_comparison" / (runner + ".py")).read_text())
    output = args.output.resolve()
    if output.exists():
        parser.error("choose a new output directory to preserve existing observations")
    for key in ("output", "output_dir", "out"):
        if key in declared:
            cfg[key] = str(output)
    if runner == "benchmark":
        cfg["no_plots"] = True
        if cfg["comparison"] == "anschuetz":
            cfg["anschuetz_implementations"] = [method.removeprefix("anschuetz_")
                                                for method in cfg["methods"] if method.startswith("anschuetz_")]
    if args.dataset == "matrix_units":
        cfg["timeout"] = cfg["timeout_s"]
    if args.anschuetz_source:
        cfg["anschuetz_source"] = str(args.anschuetz_source.resolve(strict=True))
    elif "anschuetz_public_original" in cfg.get("methods", ()):
        parser.error("this configuration needs --anschuetz-source at the documented pinned revision")
    else:
        cfg.pop("anschuetz_source", None)
    if args.smoke:
        for key in ("n", "sizes"):
            if key in cfg:
                cfg[key] = [min(cfg[key])]
        cfg.update(instances=1, repeats=1, warmup=0, trials=1, app_repeats=1)
        if args.dataset == "dynamics":
            cfg.update(points=3, tmax=0.2, sweep_n=min(cfg["n"]), sweep_points=[3])
    command = [sys.executable, str(ROOT / "run.py"), campaign]
    if args.permqit_source:
        command += ["--permqit-source", str(args.permqit_source.resolve(strict=True))]
    command.append("--")
    ignored = {"worker", "resume", "config", "input"}
    for key, (option, action) in declared.items():
        if key not in cfg or key in ignored or cfg[key] is None:
            continue
        value = cfg[key]
        if action == "store_true":
            if value:
                command.append(option)
        elif action == "store_false":
            if not value:
                command.append(option)
        else:
            command.append(option)
            command.extend(map(str, value if isinstance(value, list) else [value]))
    print(shlex.join(command), flush=True)
    if not args.show:
        raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
