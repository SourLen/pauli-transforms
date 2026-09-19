"""Recreate the seven current thesis benchmark figures from saved data.

Run ``python -m pauli_transforms.plot --data data/thesis --output figures``.
The input root can contain any of direct/, fixed_locality/, spectral/,
dynamics/, ising/ and matrix_units/. No benchmark timings run here.
Use --supplementary to also export the former main-text general-conversion
plot and the conversion-accuracy plot, which are no longer in the manuscript.
"""

import argparse
import csv
import hashlib
from itertools import cycle
import json
from math import comb
from pathlib import Path

import numpy as np


STYLES = {
    "separated": ("Our conversion", "#4477AA", "o", "-"),
    "thesis": ("Our conversion", "#4477AA", "o", "-"),
    "anschuetz_optimized": ("Anschuetz (optimized)", "#228833", "^", "-."),
    "anschuetz_public_original": ("Anschuetz (public)", "#EE6677", "s", "--"),
    "chang": ("Chang et al.", "#EE6677", "s", "--"),
    "factorial_float64": ("Shared factors", "#4477AA", "o", "-"),
    "hahn_float64": ("Hahn recurrence", "#228833", "^", "-."),
    "permqit": ("permqit", "#EE6677", "s", "--"),
}
PUBLIC = "anschuetz_public_original"
ERROR_FLOOR = 1e-17


def style(method):
    label, color, marker, line = STYLES[method]
    return dict(label=label, color=color, marker=marker, linestyle=line,
                linewidth=1.25, markersize=4, markerfacecolor="white",
                markeredgewidth=.8)


def read_campaign(path):
    cfg = json.loads((path / "config.json").read_text())
    rows = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"No measurements in {path}")
    for row in rows:
        if row.get("status", "ok") == "configured_size_limit":
            if row["method"] != PUBLIC or row["n"] <= cfg["anschuetz_public_max_n"]:
                raise ValueError("Unexpected configured size limit")
        elif row.get("valid") is not True or row.get("status", "ok") != "ok":
            raise ValueError(f"Failed measurement in {path}: {row}")
    return cfg, rows


def point_statistics(rows, cfg, field, axis, coordinates, **selection):
    """Quartiles and maximum for each complete, finite group of trials."""
    expected = {(i, r) for i in range(cfg["instances"]) for r in range(cfg["repeats"])}
    result = []
    for coordinate in coordinates:
        group = [row for row in rows if row[axis] == coordinate
                 and all(row.get(key) == value for key, value in selection.items())]
        identities = [(row["instance"], row["repeat"]) for row in group]
        if len(identities) != len(expected) or set(identities) != expected:
            raise ValueError(f"Incomplete/duplicate trials: {selection}, {axis}={coordinate}")
        if not all(row.get("valid") is True and row.get("status", "ok") == "ok" for row in group):
            raise ValueError(f"Failed trials: {selection}, {axis}={coordinate}")
        values = np.asarray([row[field] for row in group], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Invalid {field}: {selection}, {axis}={coordinate}")
        low, median, high = np.percentile(values, [25, 50, 75])
        result.append(dict(x=int(coordinate), q25=float(low), median=float(median),
                           q75=float(high), maximum=float(values.max()), samples=len(group)))
    return result


def draw_runtime(ax, rows, cfg, field, axis, coordinates, method, **selection):
    points = point_statistics(rows, cfg, field, axis, coordinates, method=method, **selection)
    appearance = style(method)
    ax.plot(coordinates, [p["median"] for p in points], **appearance)
    ax.fill_between(coordinates, [p["q25"] for p in points], [p["q75"] for p in points],
                    color=appearance["color"], alpha=.15, linewidth=0)
    return points


def label_runtime(ax, title, xlabel=r"Number of qubits, $n$"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.margins(x=.04, y=.14)


def shared_legend(fig):
    labels = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(label, handle)
    fig.legend(labels.values(), labels.keys(), loc="outside upper center",
               ncols=2, frameon=False, handlelength=2.2)


def direct_figures(path, *, fixed=False, supplementary=False):
    import matplotlib.pyplot as plt
    cfg, rows = read_campaign(path)
    sizes = cfg.get("n_values") or cfg["n"]
    localities = cfg["localities"] if fixed else [0]
    methods = ["separated", "chang"] if fixed else ["separated", "anschuetz_optimized"]
    variants = [("fixed_locality" if fixed else "general_conversion", methods, False)]
    if not fixed:
        # The public API has no cached operation.
        public_methods = methods + ([PUBLIC] if PUBLIC in cfg["methods"] else [])
        if PUBLIC in cfg["methods"]:
            if not supplementary:
                variants.clear()
            variants.append(("general_conversion_public", public_methods, False))
        if supplementary:
            variants.append(("conversion_accuracy", public_methods, True))
    for name, selected, accuracy in variants:
        fig, axes = plt.subplots(len(localities), 2, squeeze=False,
                                 figsize=(6.25, 5.65 if fixed else 3.55), layout="constrained")
        stats = {}
        for index, locality in enumerate(localities):
            for col, mode in enumerate(("cold", "cached")):
                ax = axes[index, col]
                for method in selected:
                    if method == PUBLIC and mode == "cached":
                        continue
                    ns = [n for n in sizes if method != PUBLIC or n <= cfg["anschuetz_public_max_n"]]
                    field = "relative_error" if accuracy else "elapsed_seconds"
                    selection = dict(locality=locality, cache_mode=mode)
                    if accuracy:
                        points = point_statistics(rows, cfg, field, "n", ns, method=method, **selection)
                        ax.plot(ns, [max(ERROR_FLOOR, p["maximum"]) for p in points], **style(method))
                    else:
                        points = draw_runtime(ax, rows, cfg, field, "n", ns, method, **selection)
                    stats[f"{method}/ell{locality}/{mode}"] = points
                title = "Setup included" if mode == "cold" else "Reusable data cached"
                if fixed:
                    title = rf"$\ell={locality}$: " + title
                label_runtime(ax, title)
                ax.set_xlim(0, max(sizes) * 1.04)
                if accuracy:
                    ax.set_ylabel("Maximum relative error")
                    ax.axhline(cfg.get("rtol", 1e-8), color=".5", linestyle=":", linewidth=.8)
                if PUBLIC in selected and mode == "cached":
                    ax.text(.03, .02, "Public API: no cached operation", transform=ax.transAxes,
                            fontsize=7, color=".35")
        shared_legend(fig)
        yield name, fig, stats


def read_matrix_unit_campaign(path):
    """Validate the distinct fresh-process / repeated-application protocol."""
    cfg = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "measurements.jsonl").read_text().splitlines()
            if line.strip()]
    if (cfg["trials"] < 1 or cfg["app_repeats"] < 1 or not cfg["sizes"] or not cfg["methods"]
            or len(set(cfg["sizes"])) != len(cfg["sizes"])
            or len(set(cfg["methods"])) != len(cfg["methods"])
            or set(cfg["methods"]) - {"factorial_float64", "hahn_float64", "permqit"}):
        raise ValueError("Invalid matrix-unit campaign configuration")
    expected = {(method, n, trial) for method in cfg["methods"]
                for n in cfg["sizes"] for trial in range(cfg["trials"])}
    identities = [(r["method"], r["n"], r["trial"]) for r in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Incomplete/duplicate matrix-unit trials")
    input_records = {}
    for n in cfg["sizes"]:
        for trial in range(cfg["trials"]):
            raw = (path / "inputs" / f"n{n}_trial{trial}.json").read_bytes()
            payload = json.loads(raw)
            if payload["n"] != n or payload["trial"] != trial:
                raise ValueError("Matrix-unit input identity disagrees with its filename")
            input_records[n, trial] = hashlib.sha256(raw).hexdigest(), payload["seed"]
    for row in rows:
        if row["status"] != "ok":
            raise ValueError("Failed matrix-unit trial")
        for key in ("errors", "repeated_errors"):
            errors = row[key]
            values = np.asarray([errors["weighted_hs_relative"], errors["max_abs"]])
            if (errors["passed"] is not True or not np.isfinite(values).all()
                    or np.any(values < 0)
                    or errors["weighted_hs_relative"] > cfg["gate"]["weighted_hs_relative"]):
                raise ValueError("Failed matrix-unit accuracy check")
        warm = row["warm_application_s"]
        times = np.asarray([row[key] for key in (
            "preparation_s", "first_application_s", "fresh_total_s", "warm_median_s")] + warm)
        if (len(warm) != cfg["app_repeats"] or not np.isfinite(times).all() or np.any(times < 0)
                or row["warm_median_s"] != float(np.median(warm))
                or row["fresh_total_s"] != row["preparation_s"] + row["first_application_s"]):
            raise ValueError("Inconsistent matrix-unit timing record")
        if (row["input_sha256"], row["seed"]) != input_records[row["n"], row["trial"]]:
            raise ValueError("Matrix-unit trial does not match its saved input")
    return cfg, rows


def matrix_unit_figure(path):
    import matplotlib.pyplot as plt
    cfg, rows = read_matrix_unit_campaign(path)
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.0), layout="constrained")
    stats = {}
    for method in ("factorial_float64", "hahn_float64", "permqit"):
        if method not in cfg["methods"]:
            continue
        stats[method] = {}
        for field in ("preparation_s", "fresh_total_s", "warm_median_s"):
            points = []
            for n in cfg["sizes"]:
                values = [r[field] for r in rows if r["method"] == method and r["n"] == n]
                low, median, high = np.percentile(values, [25, 50, 75])
                points.append(dict(x=n, q25=float(low), median=float(median),
                                   q75=float(high), samples=len(values)))
            stats[method][field] = points
            if field == "preparation_s":
                continue
            ax = axes[0 if field == "fresh_total_s" else 1]
            appearance = style(method)
            ax.plot(cfg["sizes"], [p["median"] for p in points], **appearance)
            ax.fill_between(cfg["sizes"], [p["q25"] for p in points],
                            [p["q75"] for p in points], color=appearance["color"],
                            alpha=.15, linewidth=0)
    for ax, title in zip(axes, ("(a) Preprocessing included", "(b) Reusable data cached")):
        label_runtime(ax, title)
        ax.set_xlim(0, max(cfg["sizes"]) * 1.04)
    shared_legend(fig)
    return "matrix_unit_schur", fig, stats


def draw_spectrum(ax, example):
    n = int(example["n"])
    ticks = []
    for k in range(n // 2 + 1):
        values = example[f"values_{k}"]
        if values.shape != (n - 2 * k + 1,) or not np.isfinite(values).all():
            raise ValueError("Invalid representative spectrum")
        ax.hlines(values, k - .27, k + .27, color="#4477AA", linewidth=1.25)
        multiplicity = comb(n, k) - (comb(n, k - 1) if k else 0)
        ticks.append(rf"${k}$" + "\n" + rf"$\times {multiplicity}$")
    ax.set_xticks(range(n // 2 + 1), ticks)
    ax.set_xlabel(r"Block $k$; multiplicity below each label")
    ax.set_ylabel("Eigenvalue")
    ax.margins(x=.04, y=.06)


def spectral_figure(path):
    import matplotlib.pyplot as plt
    cfg, rows = read_campaign(path)
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9), layout="constrained")
    stats = {}
    for method in ("thesis", "anschuetz_optimized"):
        stats[method] = draw_runtime(axes[0], rows, cfg, "total_seconds", "n", cfg["n"], method)
    label_runtime(axes[0], "(a) Complete eigensystems")
    axes[0].set_xticks([n for n in cfg["n"] if n not in (3, 4)])
    axes[0].legend(frameon=False, fontsize=8)
    with np.load(path / "spectrum_example.npz") as example:
        draw_spectrum(axes[1], example)
    axes[1].set_title("(b) One representative spectrum")
    return "spectral_comparison", fig, stats


def dynamics_figures(path):
    import matplotlib.pyplot as plt
    cfg, rows = read_campaign(path)
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9), layout="constrained")
    stats = {}
    for col, (kind, axis, coordinates) in enumerate([
        ("size", "n", cfg["n"]), ("times", "points", cfg["sweep_points"])
    ]):
        for method in ("thesis", "anschuetz_optimized"):
            stats[f"{kind}/{method}"] = draw_runtime(
                axes[col], rows, cfg, "total_seconds", axis, coordinates, method, kind=kind)
        title = rf"(a) $L={cfg['points']}$ times" if col == 0 else rf"(b) $n={cfg['sweep_n']}$ qubits"
        label_runtime(axes[col], title, r"Number of qubits, $n$" if col == 0 else r"Requested times, $L$")
        axes[col].set_xticks([n for n in coordinates if col == 1 or n not in (3, 4)])
        if col == 1:
            axes[col].set_xscale("log")
            axes[col].set_xticks(coordinates, [str(n) for n in coordinates])
            axes[col].minorticks_off()
        axes[col].legend(frameon=False, fontsize=8)
    yield "random_dynamics_comparison", fig, stats

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9), layout="constrained")
    stats = {}
    stages = [("setup_seconds", "Fresh tables", "#4477AA", "o", "-"),
              ("conversion_seconds", "Three conversions", "#EE6677", "s", "--"),
              ("eigensolve_seconds", "Eigensystems", "#CCBB44", "^", ":"),
              ("curve_seconds", "Basis changes + curve", "#66CCEE", "x", "-.")]
    for field, label, color, marker, line in stages:
        points = point_statistics(rows, cfg, field, "n", cfg["n"], method="thesis", kind="size")
        axes[0].plot(cfg["n"], [p["median"] for p in points], label=label, color=color,
                     marker=marker, linestyle=line, linewidth=1.25, markersize=4)
        axes[0].fill_between(cfg["n"], [p["q25"] for p in points], [p["q75"] for p in points],
                             color=color, alpha=.15, linewidth=0)
        stats[field] = points
    for method in ("thesis", "anschuetz_optimized"):
        points = point_statistics(rows, cfg, "max_abs_error", "n", cfg["n"], method=method, kind="size")
        axes[1].plot(cfg["n"], [max(ERROR_FLOOR, p["maximum"]) for p in points], **style(method))
        stats[method] = points
    label_runtime(axes[0], "(a) Stages of our method")
    label_runtime(axes[1], "(b) Agreement of full curves")
    axes[1].set_ylabel("Maximum absolute error")
    for ax in axes:
        ax.set_xticks([n for n in cfg["n"] if n not in (3, 4)])
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 5)
    yield "random_dynamics_diagnostics", fig, stats


def ising_figure(path):
    import matplotlib.pyplot as plt
    cfg = json.loads((path / "config.json").read_text())
    validation = json.loads((path / "validation.json").read_text())
    flags = [validation[key] for key in ("all_valid", "all_passed") if key in validation]
    tolerance = float(validation["tolerance"])
    if not flags or any(flag is not True for flag in flags) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("The Ising calculation has no successful validation record")
    with np.load(path / "spectrum.npz") as saved:
        example = dict(saved)
    if int(example["n"]) != cfg["spectrum_n"]:
        raise ValueError("The Ising spectrum does not match the configured size")
    with (path / "selected_curves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    curves = {}
    for n in cfg["n"]:
        group = sorted([r for r in rows if int(r["n"]) == n], key=lambda r: float(r["time"]))
        times = np.asarray([float(r["time"]) for r in group])
        values = np.asarray([complex(float(r["value_real"]), float(r["value_imag"])) for r in group])
        reference = np.asarray([complex(float(r["reference_real"]), float(r["reference_imag"])) for r in group])
        if len(group) != cfg["points"] or not np.array_equal(times, np.linspace(0, cfg["tmax"], cfg["points"])):
            raise ValueError(f"Incomplete Ising curve at n={n}")
        if (not np.isfinite(values).all() or not np.isfinite(reference).all()
                or np.max(np.abs(values.imag)) > tolerance
                or np.max(np.abs(values - reference)) > tolerance
                or abs(values[0] - cfg["p"]) > tolerance):
            raise ValueError(f"Invalid Ising curve at n={n}")
        curves[n] = times, values
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9), layout="constrained")
    draw_spectrum(axes[0], example)
    axes[0].set_title(rf"(a) Energy levels, $n={int(example['n'])}$")
    axes[0].set_ylabel(r"Energy $E$")
    stats = {}
    for n, color, line in zip(cfg["n"], cycle(["#4477AA", "#EE6677", "#228833"]), cycle(["-", "--", "-."])):
        times, values = curves[n]
        axes[1].plot(times, values.real, label=rf"$n={n}$", color=color, linestyle=line, linewidth=1.25)
        stats[str(n)] = dict(points=len(times), initial=float(values[0].real),
                             minimum=float(values.real.min()), final=float(values[-1].real))
    axes[1].set(title="(b) Transverse magnetization", xlabel=r"Time $t$",
                ylabel=r"$\langle M\rangle_t$", xlim=(0, cfg["tmax"]))
    axes[1].set_xticks(np.linspace(0, cfg["tmax"], 5))
    axes[1].margins(y=.08)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.59, 1))
    return "ising_example", fig, stats


def export(data, output, formats=("pdf", "svg", "png"), *, supplementary=False):
    """Export current manuscript figures and their unrounded plotted statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    rc = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
          "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
          "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .6}
    def save(result):
        name, fig, statistics = result
        try:
            for extension in formats:
                fig.savefig(output / f"{name}.{extension}", dpi=300)
            figures[name] = statistics
        finally:
            plt.close(fig)
    with matplotlib.rc_context(rc):
        for name in ("direct", "fixed_locality"):
            if (data / name).is_dir():
                for result in direct_figures(data / name, fixed=name == "fixed_locality",
                                             supplementary=supplementary):
                    save(result)
        if (data / "matrix_units").is_dir():
            save(matrix_unit_figure(data / "matrix_units"))
        if (data / "spectral").is_dir():
            save(spectral_figure(data / "spectral"))
        if (data / "dynamics").is_dir():
            for result in dynamics_figures(data / "dynamics"):
                save(result)
        if (data / "ising").is_dir():
            save(ising_figure(data / "ising"))
    if not figures:
        raise ValueError("Input must contain direct/, fixed_locality/, matrix_units/, spectral/, dynamics/ or ising/")
    (output / "plotted_statistics.json").write_text(json.dumps(figures, indent=2, allow_nan=False) + "\n")
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/thesis"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--supplementary", action="store_true",
                        help="also export the former general-conversion and accuracy figures")
    args = parser.parse_args(argv)
    figures = export(args.data, args.output, supplementary=args.supplementary)
    print(f"Exported {len(figures)} figures to {args.output}")


if __name__ == "__main__":
    main()
