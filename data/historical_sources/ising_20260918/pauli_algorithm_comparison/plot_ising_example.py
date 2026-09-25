"""A verified Ising spectrum and retained magnetization curves, without timings.

The spectrum is newly evaluated through the thesis conversion and checked against
collective-spin blocks. Magnetization values are copied from a recorded campaign
and checked again, not replaced with a newly sampled curve.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import dynamics_baselines, physical_models, schur_separated
from .common import specht_multiplicities
from .plot_benchmarks import PlotSettings, save_figure, sha256
from .plot_spectral_comparison import draw_spectrum


SETTINGS = PlotSettings(width=6.25, height=2.9, font_size=9,
                       line_width=1.25, marker_size=4, dpi=300, xscale="linear")
PARAMETERS = dict(g=1.0, h=0.5, p=0.6)
TOLERANCE = 1e-10


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_product(factors):
    result = np.array([[1.0]], dtype=complex)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def explicit_inputs(n, g, h, p):
    """Independent tensor-product construction for the small-space checks."""
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1.0, -1.0])
    local_x = [tensor_product([x if site == i else identity for site in range(n)])
               for i in range(n)]
    local_z = [tensor_product([z if site == i else identity for site in range(n)])
               for i in range(n)]
    hamiltonian = -h / n * sum(local_x)
    for i in range(n):
        for j in range(i + 1, n):
            hamiltonian -= g / n * (local_z[i] @ local_z[j])
    state = tensor_product([(identity + p * x) / 2] * n)
    return hamiltonian, state, sum(local_x) / n


def spectrum_and_small_checks(n_example):
    example = dict(n=n_example)
    checks = []
    times = np.array([0.0, 0.13, 0.8, 2.7, 12.0])
    for n in sorted({1, 2, 3, 4, 5, n_example}):
        coefficients = physical_models.ising_pauli(n, PARAMETERS["g"], PARAMETERS["h"])
        converted = schur_separated.pauli_to_schur(n, coefficients, backend="hahn")
        reference = physical_models.ising_collective_blocks(n, PARAMETERS["g"], PARAMETERS["h"])
        spectrum = physical_models.diagonalize_blocks(converted)
        record = dict(n=n, collective_block_max_abs_error=max(
            float(np.max(np.abs(actual - expected)))
            for actual, expected in zip(converted, reference, strict=True)))
        record["eigenpair_max_abs_residual"] = max(
            float(np.max(np.abs(block @ basis - basis * energies)))
            for block, basis, energies in zip(converted, spectrum.vectors, spectrum.energies, strict=True))
        if n == n_example:
            example.update({f"values_{k}": values for k, values in enumerate(spectrum.energies)})
        if n <= 5:
            h_full, rho_full, m_full = explicit_inputs(n, **PARAMETERS)
            energies, vectors = np.linalg.eigh(h_full)
            expanded = np.sort(np.concatenate([
                np.repeat(values, mu) for values, mu in
                zip(spectrum.energies, specht_multiplicities(n), strict=True)]))
            record["full_spectrum_max_abs_error"] = float(np.max(np.abs(expanded - energies)))
            state = schur_separated.pauli_to_schur(n, physical_models.product_x_pauli(n, PARAMETERS["p"]), backend="hahn")
            observable = schur_separated.pauli_to_schur(n, {(1, 0, 0): 1 / n}, backend="hahn")
            curve = physical_models.expectation_time_series(n, spectrum, state, observable, times)
            full_curve = []
            for time in times:
                evolution = (vectors * np.exp(-1j * time * energies)) @ vectors.conj().T
                full_curve.append(np.trace(m_full @ evolution @ rho_full @ evolution.conj().T))
            record["full_curve_max_abs_error"] = float(np.max(np.abs(curve - full_curve)))
            record["state_trace_error"] = float(abs(physical_models.block_trace(n, state) - 1))
        record["valid"] = all(value <= TOLERANCE for key, value in record.items() if key != "n")
        checks.append(record)
    return example, checks


def selected_curves(source, sizes):
    cfg = json.loads((source / "config.json").read_text())
    if any(cfg[name] != value for name, value in PARAMETERS.items()):
        raise ValueError("The retained campaign does not use the requested Ising parameters")
    records = [json.loads(line) for line in (source / "runs.jsonl").read_text().splitlines()]
    with (source / "curves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected, checks = [], []
    times_expected = np.asarray(cfg["times"])
    for n in sizes:
        group = sorted((row for row in rows if int(row["n"]) == n and row["method"] == "thesis"),
                       key=lambda row: float(row["time"]))
        trials = [row for row in records if row["n"] == n and row["method"] == "thesis"]
        if len(trials) != cfg["repeats"] or not all(row["valid"] and row["status"] == "ok" for row in trials):
            raise ValueError(f"The saved thesis trials at n={n} did not all pass")
        times = np.array([float(row["time"]) for row in group])
        if not np.array_equal(times, times_expected):
            raise ValueError(f"The retained curve at n={n} has missing, duplicate or changed times")
        values = np.array([complex(float(row["value_real"]), float(row["value_imag"])) for row in group])
        saved_reference = np.array([complex(float(row["reference_real"]), float(row["reference_imag"])) for row in group])
        hamiltonian, state, observable = dynamics_baselines.prepare_direct(n, **PARAMETERS)
        spectrum = physical_models.diagonalize_blocks(hamiltonian)
        reference = physical_models.expectation_time_series(n, spectrum, state, observable, times)
        record = dict(n=n, points=len(group), retained_valid_trials=len(trials),
                      saved_reference_max_abs_error=float(np.max(np.abs(values - saved_reference))),
                      fresh_reference_max_abs_error=float(np.max(np.abs(values - reference))),
                      imaginary_max_abs=float(np.max(np.abs(values.imag))),
                      initial_value_error=float(abs(values[0] - PARAMETERS["p"])),
                      minimum=float(np.min(values.real)), final=float(values[-1].real))
        record["valid"] = all(record[name] <= TOLERANCE for name in [
            "saved_reference_max_abs_error", "fresh_reference_max_abs_error", "imaginary_max_abs", "initial_value_error"])
        selected.extend(group)
        checks.append(record)
    return selected, checks


def make_figure(example, rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(SETTINGS.width, SETTINGS.height), layout="constrained")
    draw_spectrum(axes[0], example)
    axes[0].set_title(rf"(a) Energy levels, $n={int(example['n'])}$", fontsize=SETTINGS.font_size)
    axes[0].set_ylabel(r"Energy $E$")
    axes[0].tick_params(axis="x", labelsize=8)
    styles = [dict(color="#4477AA", linestyle="-"),
              dict(color="#EE6677", linestyle="--"),
              dict(color="#228833", linestyle="-.")]
    for n, style in zip(sorted({int(row["n"]) for row in rows}), styles, strict=True):
        group = [row for row in rows if int(row["n"]) == n]
        axes[1].plot([float(row["time"]) for row in group],
                     [float(row["value_real"]) for row in group],
                     label=rf"$n={n}$", linewidth=SETTINGS.line_width, **style)
    axes[1].set_title("(b) Transverse magnetization", fontsize=SETTINGS.font_size)
    axes[1].set_xlabel(r"Time $t$")
    axes[1].set_ylabel(r"$\langle M\rangle_t$")
    axes[1].set_xlim(0, 12)
    axes[1].set_xticks([0, 3, 6, 9, 12])
    axes[1].set_ylim(-.07, .65)
    axes[1].set_yticks([0, .2, .4, .6])
    axes[1].legend(frameon=False, loc="upper center", fontsize=8, bbox_to_anchor=(.59, 1))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved physical dynamics campaign")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--spectrum-n", type=int, default=12)
    args = parser.parse_args(argv)
    source, results, figures = (path.resolve() for path in (args.source, args.results, args.figures))
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        example, spectrum_checks = spectrum_and_small_checks(args.spectrum_n)
        rows, curve_checks = selected_curves(source, [8, 20, 40])
    validation = dict(tolerance=TOLERANCE, spectrum_checks=spectrum_checks,
                      retained_curve_checks=curve_checks,
                      all_passed=all(row["valid"] for row in spectrum_checks + curve_checks))
    write_json(results / "validation.json", validation)
    if not validation["all_passed"]:
        raise ArithmeticError("At least one Ising illustration check failed")
    np.savez(results / "spectrum.npz", **example)
    with (results / "selected_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    rc = {"font.size": SETTINGS.font_size, "axes.labelsize": SETTINGS.font_size,
          "xtick.labelsize": SETTINGS.font_size, "ytick.labelsize": SETTINGS.font_size,
          "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
          "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "thesis-ising-example",
          "savefig.bbox": None}
    caption = ("Fully connected Ising model H=-(g sum_{i<j} ZiZj+h sum_i Xi)/n, with g=1, h=1/2. "
               f"(a) All representative eigenvalues for n={args.spectrum_n}; the indicated multiplicities apply to each level. "
               "Nearly degenerate levels can overlap at this scale. "
               "(b) Expectation of M=sum_i Xi/n from rho(0)=[(I+0.6X)/2]^tensor(n), using the "
               "retained thesis-conversion curves for n=8,20,40 on t=0,...,12 in steps of 0.05. "
               "No dynamics timing measurements were rerun.")
    with plt.style.context(["science", "ieee", "bright", "no-latex"]), matplotlib.rc_context(rc):
        artifact = save_figure(make_figure(example, rows), figures, "ising_example", caption, SETTINGS)
    code = Path(__file__).resolve().parent
    manifest = dict(parameters=PARAMETERS, spectrum_n=args.spectrum_n, trajectory_sizes=[8, 20, 40],
                    dynamics_source=str(source), results=str(results), figures=[artifact],
                    source_sha256={name: sha256(source / name) for name in
                        ["config.json", "curves.csv", "runs.jsonl", "validation.json", "selection_provenance.json"]},
                    code_sha256={name: sha256(code / name) for name in
                        [Path(__file__).name, "physical_models.py", "dynamics_baselines.py", "schur_separated.py", "plot_benchmarks.py", "plot_spectral_comparison.py"]},
                    result_sha256={name: sha256(results / name) for name in
                        ["spectrum.npz", "selected_curves.csv", "validation.json"]},
                    matplotlib=matplotlib.__version__, numpy=np.__version__, settings=asdict(SETTINGS),
                    note="Spectrum newly computed; selected CSV rows retained unchanged; no new timing samples.")
    write_json(results / "provenance.json", manifest)
    write_json(figures / "ising_example_manifest.json", manifest)
    print(json.dumps(validation, indent=2))
    print(figures / "ising_example.pdf")


if __name__ == "__main__":
    main()
