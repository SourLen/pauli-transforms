"""Figures from the saved general-eigensystem campaign; no solver is run."""

import argparse
from dataclasses import asdict
import json
from math import comb
from pathlib import Path

import numpy as np

from .plot_benchmarks import PlotSettings, line_style, save_figure, sha256


SETTINGS = PlotSettings(width=6.25, height=2.9, font_size=9,
                       line_width=1.25, marker_size=4, dpi=300, xscale="linear")
SERIES = {
    "thesis": dict(label="Our conversion", color_index=0, marker="o", linestyle="-"),
    "anschuetz_optimized": dict(label="Anschuetz (optimized)", color_index=2, marker="v", linestyle="-."),
    "anschuetz_public_original": dict(label="Anschuetz (public)", color_index=4, marker="x", linestyle="--"),
}


def draw_curve(ax, records, method, field, style, cfg, settings):
    expected = cfg["instances"] * cfg["repeats"]
    ns, medians, lows, highs = [], [], [], []
    for n in cfg["n"]:
        group = [row for row in records if row["n"] == n and row["method"] == method]
        if method == "anschuetz_public_original" and n > cfg["public_max_n"]:
            continue
        values = [row[field] for row in group]
        valid = len(group) == expected and all(row["valid"] for row in group)
        low, median, high = np.percentile(values, [25, 50, 75]) if valid else (np.nan,) * 3
        ns.append(n)
        medians.append(median)
        lows.append(low)
        highs.append(high)
    ax.plot(ns, medians, **style)
    ax.fill_between(ns, lows, highs, color=style["color"], alpha=settings.band_alpha, linewidth=0)


def runtime_figure(records, cfg, *, include_public=False, example=None, settings=SETTINGS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(settings.width, settings.height), layout="constrained")
    methods = list(SERIES) if include_public else ["thesis", "anschuetz_optimized"]
    for method in methods:
        draw_curve(axes[0], records, method, "total_seconds", line_style(SERIES[method], settings), cfg, settings)
    if example is not None:
        draw_spectrum(axes[1], example)
        axes[1].set_title("(b) One representative spectrum", fontsize=settings.font_size)
        axes[0].set_title("(a) Complete eigensystems", fontsize=settings.font_size)
        axes[0].set_xlabel(r"Number of qubits, $n$")
        axes[0].set_ylabel("Time (s)")
        axes[0].set_yscale("log")
        axes[0].set_xticks([2, 5, 8, 12, 16, 20])
        axes[0].minorticks_off()
        axes[0].margins(x=.04, y=.17)
        axes[0].legend(loc="best", frameon=False, fontsize=8, handlelength=2.0)
        return fig
    # The second panel concerns one conversion route and holds its prepared
    # blocks fixed for the two solver requests. It does not compare different
    # physical problems or imply that block construction can be omitted.
    staged = [dict(row, construction_seconds=row["setup_seconds"] + row["conversion_seconds"]) for row in records]
    stages = [
        ("construction_seconds", "Conversion, incl. setup", 0, "o", "-"),
        ("eigensolve_seconds", "All eigenpairs", 1, "s", "--"),
        ("lowest_eigensolve_seconds", "Lowest pair in each block", 3, "^", ":"),
    ]
    for field, label, color, marker, linestyle in stages:
        style = line_style(dict(label=label, color_index=color, marker=marker, linestyle=linestyle), settings)
        draw_curve(axes[1], staged, "thesis", field, style, cfg, settings)
    for ax, title in zip(axes, ["(a) Complete eigensystems", "(b) Stages of our method"]):
        ax.set_title(title, fontsize=settings.font_size)
        ax.set_xlabel(r"Number of qubits, $n$")
        ax.set_ylabel("Time (s)")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 8, 12, 16, 20])
        ax.minorticks_off()
        ax.margins(x=.04, y=.17)
        ax.legend(loc="best", frameon=False, fontsize=8, handlelength=2.0)
    return fig


def draw_spectrum(ax, example):
    n = int(example["n"])
    ticks = []
    for k in range(n // 2 + 1):
        values = example[f"values_{k}"]
        ax.hlines(values, k - .27, k + .27, linewidth=1.25, color="#4477AA")
        multiplicity = comb(n, k) - (comb(n, k - 1) if k else 0)
        ticks.append(rf"${k}$" + "\n" + rf"$\times {multiplicity}$")
    ax.set_xticks(range(n // 2 + 1), ticks)
    ax.set_xlabel(r"Block $k$; multiplicity below each label")
    ax.set_ylabel("Eigenvalue")
    ax.margins(x=.04, y=.06)


def spectrum_figure(example, settings=SETTINGS):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(settings.width, 2.7), layout="constrained")
    draw_spectrum(ax, example)
    ax.set_title(rf"One general Hermitian input, $n={int(example['n'])}$", fontsize=settings.font_size)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    source = args.input.resolve()
    out = (args.output or source / "plots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / "config.json").read_text())
    records = [json.loads(line) for line in (source / "runs.jsonl").read_text().splitlines()]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    rc = {"font.size": SETTINGS.font_size, "axes.labelsize": SETTINGS.font_size,
          "xtick.labelsize": SETTINGS.font_size, "ytick.labelsize": SETTINGS.font_size,
          "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
          "pdf.fonttype": 42, "svg.fonttype": "none"}
    caption = ("General Hermitian permutation-invariant inputs with independent real Gaussian Pauli-orbit coefficients. "
               "(a) Fresh setup, conversion and all representative eigenpairs, using the same dense Hermitian solver. "
               "(b) All representative eigenvalues of one n=12 input; each level occurs with the indicated multiplicity in the full spectrum. "
               "Medians and interquartile ranges over three inputs and five repetitions. The conversion uses float64 Hahn recurrences.")
    figures = []
    with plt.style.context(["science", "ieee", "bright", "no-latex"]), matplotlib.rc_context(rc):
        if (source / "spectrum_example.npz").exists():
            with np.load(source / "spectrum_example.npz") as example:
                figures.append(save_figure(runtime_figure(records, cfg, example=example), out, "spectral_comparison", caption, SETTINGS))
                figures.append(save_figure(runtime_figure(records, cfg, include_public=True, example=example), out, "spectral_comparison_public",
                    caption + " The original public routine is measured only through n=5, without cached tables.", SETTINGS))
                figures.append(save_figure(spectrum_figure(example), out, "representative_spectrum",
                    "All representative eigenvalues of one general n=12 Hermitian input; each level occurs with the indicated multiplicity in the full spectrum.", SETTINGS))
        figures.append(save_figure(runtime_figure(records, cfg), out, "spectral_stages",
            "Complete eigensystem runtime, and our method's construction and eigensolver stages. The selected solver asks for the lowest pair in every already constructed block; no construction cost is avoided. Medians and interquartile ranges across three inputs and five repetitions.", SETTINGS))
    manifest = dict(figures=figures, settings=asdict(SETTINGS), matplotlib=matplotlib.__version__,
                    input_sha256={name: sha256(source / name) for name in ["config.json", "runs.jsonl", "validation.json"]},
                    plotting_source_sha256=sha256(__file__))
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
