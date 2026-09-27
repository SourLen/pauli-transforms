"""Plot the new baseline campaign only, preserving the retained example spectrum."""

import argparse
import json
from pathlib import Path

import run_comparisons as bench
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pauli_transforms.plot import draw_spectrum


STYLES = {
    "thesis": ("Our conversion", "#4477AA", "o", "-"),
    "anschuetz_optimized": ("Anschuetz (optimized)", "#228833", "^", "-."),
    "full_space": ("Full-space diagonalization", "#EE6677", "s", "--"),
    "piqs_blocks": ("PIQS block construction", "#EE6677", "s", "--"),
}
RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
      "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
      "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .6}


def load(path):
    cfg = json.loads((path / "config.json").read_text())
    rows = [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]
    validation = json.loads((path / "validation.json").read_text())
    if not validation["all_valid"] or not validation["source_unchanged"]:
        raise ValueError("Unvalidated campaign")
    if len(rows) != validation["samples"] or any(not row["valid"] for row in rows):
        raise ValueError("Failed or incomplete campaign")
    return cfg, rows


def draw(ax, cfg, rows, method, field):
    label, color, marker, line = STYLES[method]
    sizes = [n for n in cfg["sizes"] if method != "full_space" or n <= cfg["dense_max_n"]]
    expected = {(i, r) for i in range(cfg["instances"]) for r in range(cfg["repeats"])}
    points = []
    for n in sizes:
        selected = [r for r in rows if r["n"] == n and r["method"] == method]
        if (len(selected) != len(expected) or
                {(r["instance"], r["repeat"]) for r in selected} != expected):
            raise ValueError(f"Missing/duplicate trials for {method} n={n}")
        values = np.array([r[field] for r in selected])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Nonpositive or nonfinite timing")
        low, median, high = np.percentile(values, [25, 50, 75])
        points.append(dict(n=n, q25=float(low), median=float(median), q75=float(high), samples=len(values)))
    ax.plot(sizes, [p["median"] for p in points], label=label, color=color,
            marker=marker, linestyle=line, linewidth=1.25, markersize=4,
            markerfacecolor="white", markeredgewidth=.8)
    ax.fill_between(sizes, [p["q25"] for p in points], [p["q75"] for p in points],
                    color=color, alpha=.15, linewidth=0)
    ax.set(xlabel=r"Number of qubits, $n$", ylabel="Time (s)", yscale="log")
    ax.minorticks_off()
    ax.margins(x=.04, y=.12)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--ising", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=bench.ROOT)
    args = parser.parse_args()
    args.spectral, args.ising, args.output = args.spectral.resolve(), args.ising.resolve(), args.output.resolve()
    args.output.mkdir(exist_ok=True, parents=True)
    all_statistics = {}
    with plt.rc_context(RC):
        cfg, rows = load(args.spectral)
        fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.0), layout="constrained")
        statistics = {}
        for method in ("thesis", "anschuetz_optimized", "full_space"):
            statistics[method] = draw(axes[0], cfg, rows, method, "total_seconds")
        axes[0].set_title("(a) Complete eigensystems")
        axes[0].set_xticks([2, 5, 8, 12, 16, 20])
        axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
        with np.load(bench.ROOT / "spectrum_example.npz") as spectrum:
            draw_spectrum(axes[1], spectrum)
        axes[1].set_title("(b) One representative spectrum")
        for ext in ("pdf", "svg", "png"):
            fig.savefig(args.output / f"spectral_comparison.{ext}", dpi=300)
        plt.close(fig)
        all_statistics["spectral_comparison"] = statistics

        cfg, rows = load(args.ising)
        fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.0), layout="constrained")
        statistics = {}
        for method in ("thesis", "piqs_blocks"):
            statistics[method] = {}
            for ax, field in zip(axes, ("prepare_seconds", "total_seconds")):
                statistics[method][field] = draw(ax, cfg, rows, method, field)
        axes[0].set_title("(a) Construct $H$, $\\rho(0)$ and $M$")
        axes[1].set_title("(b) Complete magnetization curve")
        for ax in axes:
            ax.set_xticks([2, 8, 16, 24, 32, 40])
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        for ext in ("pdf", "svg", "png"):
            fig.savefig(args.output / f"ising_runtime_comparison.{ext}", dpi=300)
        plt.close(fig)
        all_statistics["ising_runtime_comparison"] = statistics
    bench.write_json(args.output / "plotted_statistics.json", all_statistics)
    files = [Path(__file__), bench.ROOT / "spectrum_example.npz"]
    for folder in (args.spectral, args.ising):
        files.extend(folder / name for name in ("config.json", "runs.jsonl", "validation.json", "source_hashes.json"))
    bench.write_json(args.output / "figure_manifest.json", {
        "inputs": {str(path.relative_to(bench.ROOT)): bench.sha256(path) for path in files},
        "outputs": {path.name: bench.sha256(path) for path in args.output.glob("*.pdf")},
        "note": "Fresh timing campaign. Existing n=12 illustrative spectrum retained byte-for-byte."})


if __name__ == "__main__":
    main()
