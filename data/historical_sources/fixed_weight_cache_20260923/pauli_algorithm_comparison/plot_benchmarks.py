"""Scientific benchmark figures using Matplotlib and SciencePlots.

Run directly from this folder (no benchmark is rerun):
    python3 plot_benchmarks.py results/verification/chang
    python3 plot_benchmarks.py results/verification/* --font-size 9

EDIT HERE: PlotSettings and SERIES below control the appearance. The drawing
functions runtime_figure, accuracy_figure, and query_figure are ordinary
Matplotlib code. See PLOTTING.md for examples and the statistical conventions.

SciencePlots: https://github.com/garrettj403/SciencePlots
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


# ---- Edit these settings to match the thesis or a journal. ----
@dataclass(frozen=True)
class PlotSettings:
    # Later SciencePlots styles override earlier ones. no-latex is appended
    # automatically unless use_latex=True; MathText needs no TeX installation.
    styles: tuple = ("science", "ieee", "bright")
    use_latex: bool = False
    font_size: float = 8.0             # points at the exported physical size
    panel_width: float = 3.4           # inches per column
    panel_height: float = 2.45         # inches per row, plus figure padding
    width: float | None = None         # optional total figure width, in inches
    height: float | None = None        # optional total figure height, in inches
    line_width: float = 1.0
    marker_size: float = 3.8
    band_alpha: float = 0.15           # interquartile range, not a confidence interval
    legend_columns: int | None = None  # default: one column inside each panel
    xscale: str = "log"                # "linear" is also useful for qubit sweeps
    grid: bool = False
    dpi: int = 600                     # applies to PNG; PDF and SVG are vector
    formats: tuple = ("pdf", "svg", "png")
    error_floor: float = 1e-17         # display only; raw errors are never changed


DEFAULTS = PlotSettings()

# color_index refers to the active SciencePlots palette. Set color="#..." on
# any entry to choose a specific color. Lines and markers also work in grayscale.
SERIES = {
    "separated": dict(label="Thesis route", color_index=0, marker="o", linestyle="-"),
    "anschuetz_public_original": dict(label="Anschuetz (public)", color_index=1, marker="s", linestyle="--"),
    "anschuetz_public": dict(label="Anschuetz (public F, adapter)", color_index=1, marker="s", linestyle="--"),
    "anschuetz_optimized": dict(label="Anschuetz (optimized implementation)", color_index=2, marker="^", linestyle="-."),
    "chang": dict(label="Chang et al.", color_index=1, marker="s", linestyle="--"),
    "spencer": dict(label="Spencer", color_index=1, marker="s", linestyle="--"),
    "georges": dict(label="Georges (NumPy)", color_index=1, marker="s", linestyle="--"),
}
QUERY_SERIES = {
    "row_queries": dict(label="Complete-row queries", color_index=0, marker="o", linestyle="-"),
    "entry_queries": dict(label="Entry queries", color_index=1, marker="s", linestyle="--"),
    "query_count": dict(label="Total queries", color_index=2, marker="^", linestyle="-."),
}
FIGURE_NAMES = {
    "anschuetz": "comparison_general_conversion",
    "chang": "comparison_fixed_locality",
    "spencer": "comparison_spencer",
    "georges": "comparison_georges",
}

# ---- Recorded-data summaries. Keep these independent of visual styling. ----
GROUP_FIELDS = ("comparison", "method", "n", "family", "locality", "cache_mode", "output_contract")
METRICS = ("elapsed_seconds", "preparation_seconds", "setup_seconds", "apply_seconds",
           "output_seconds", "cache_build_seconds", "query_count", "row_queries", "entry_queries",
           "relative_error", "worker_peak_rss_bytes")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in GROUP_FIELDS)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(GROUP_FIELDS, key))
        good = [row for row in group if row.get("status") == "ok" and row.get("valid") is True]
        summary.update(n_success=len(good), n_records=len(group),
                       all_successful=len(good) == len(group),
                       pauli_support=group[0].get("pauli_support"),
                       statuses=";".join(sorted({row["status"] for row in group})))
        for metric in METRICS:
            values = [row[metric] for row in good if row.get(metric) is not None and np.isfinite(row[metric])]
            if values:
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                summary.update({metric + "_median": float(median), metric + "_q25": float(q25),
                                metric + "_q75": float(q75)})
            else:
                summary.update({metric + suffix: None for suffix in ("_median", "_q25", "_q75")})
        result.append(summary)
    return result


def summary_curve(summaries, cfg, method, locality, mode, metric):
    """Return median and quartiles, leaving a gap at any failed/missing size."""
    lookup = {row["n"]: row for row in summaries
              if row["method"] == method and row["locality"] == locality and row["cache_mode"] == mode}
    values = []
    for suffix in ("_median", "_q25", "_q75"):
        column = []
        for n in cfg["n_values"]:
            row = lookup.get(n, {})
            value = row.get(metric + suffix) if row.get("all_successful") is True else None
            column.append(value if value is not None and np.isfinite(value) and value > 0 else np.nan)
        values.append(np.asarray(column, dtype=float))
    return values


# ---- Ordinary Matplotlib helpers: axes, line appearance, and panel legends. ----
def line_style(spec, settings):
    import matplotlib.pyplot as plt
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = spec.get("color", palette[spec["color_index"] % len(palette)])
    return dict(label=spec["label"], color=color, marker=spec["marker"], linestyle=spec["linestyle"],
                linewidth=settings.line_width, markersize=settings.marker_size,
                markerfacecolor="white", markeredgewidth=0.8)


def axes_grid(cfg, settings, ylabel):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter

    localities = cfg["localities"] if cfg["comparison"] in ("chang", "spencer") else [0]
    modes = cfg["cache_modes"]
    nrows, ncols = len(localities), len(modes)
    width = settings.width or settings.panel_width * ncols
    height = settings.height or settings.panel_height * nrows + 0.35
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(width, height), layout="constrained")
    sizes = np.asarray(cfg["n_values"])
    coordinates = np.log(sizes) if settings.xscale == "log" else sizes
    targets = np.linspace(coordinates[0], coordinates[-1], min(6, len(sizes)))
    ticks = sorted({int(sizes[np.argmin(abs(coordinates - target))]) for target in targets})
    panels = []
    for i, locality in enumerate(localities):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            ax.set_xscale(settings.xscale)
            ax.set_yscale("log")
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_locator(NullLocator())  # qubits are integer observations
            ax.set_xlabel(r"Number of qubits, $n$")
            if j == 0:
                ax.set_ylabel(ylabel)
            title_parts = []
            if nrows * ncols > 1:
                title_parts.append(f"({chr(97 + i * ncols + j)})")
            if cfg["comparison"] in ("chang", "spencer"):
                title_parts.append(rf"$\ell={locality}$")
            if title_parts:
                ax.set_title(" ".join(title_parts), loc="left", pad=6)
            if settings.grid:
                ax.grid(axis="y", which="major", color="0.85", linewidth=0.4)
            ax.set_axisbelow(True)
            ax.margins(x=0.06, y=0.12)
            panels.append((ax, locality, mode))
    return fig, panels


def finish_figure(fig, settings):
    from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, LogLocator, NullFormatter

    # Choose enough labeled log ticks even when the observations span much
    # less than a decade. Tick changes never change the plotted observations.
    for ax in fig.axes:
        lower, upper = ax.get_ylim()
        short_range = 0 < lower < upper < 100 * lower
        if short_range:
            ticks = LogLocator(base=10, subs=(1, 2, 5)).tick_values(lower, upper)
            if np.count_nonzero((ticks >= lower) & (ticks <= upper)) < 2:
                ticks = LogLocator(base=10, subs=tuple(range(1, 10))).tick_values(lower, upper)
            visible = ticks[(ticks >= lower) & (ticks <= upper)]
            if len(visible) < 2:
                # An almost constant series can fit between adjacent log ticks.
                # Label positions within that narrow range instead of an empty axis.
                visible = np.geomspace(lower, upper, 3)
            if len(visible) > 6:
                visible = visible[np.linspace(0, len(visible) - 1, 6, dtype=int)]
            ax.yaxis.set_major_locator(FixedLocator(visible))
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(
            base=10, labelOnlyBase=False, minor_thresholds=(float("inf"), float("inf"))))
        minor_ticks = (2, 5) if upper / lower > 1e4 else tuple(range(2, 10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=minor_ticks, numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.legend(loc="best", ncols=settings.legend_columns or 1,
                  handlelength=2.5, columnspacing=1.3, frameon=False)
    fig.align_ylabels()


def runtime_figure(cfg, summaries, settings):
    fig, panels = axes_grid(cfg, settings, "Time per conversion (s)")
    for ax, locality, mode in panels:
        any_data = False
        for method in cfg["methods"]:
            if mode not in cfg.get("method_cache_modes", {}).get(method, cfg["cache_modes"]):
                continue
            median, q25, q75 = summary_curve(summaries, cfg, method, locality, mode, "elapsed_seconds")
            style = line_style(SERIES[method], settings)
            ax.plot(cfg["n_values"], median, **style)
            ax.fill_between(cfg["n_values"], q25, q75, color=style["color"],
                            alpha=settings.band_alpha, linewidth=0)
            any_data |= np.isfinite(median).any()
        if not any_data:
            ax.text(0.5, 0.5, "No fully validated points\nSee runs.csv for status",
                    transform=ax.transAxes, ha="center", va="center")
    finish_figure(fig, settings)
    return fig


def accuracy_figure(cfg, rows, settings):
    fig, panels = axes_grid(cfg, settings, "Relative error (maximum)")
    for ax, locality, mode in panels:
        for method in cfg["methods"]:
            if mode not in cfg.get("method_cache_modes", {}).get(method, cfg["cache_modes"]):
                continue
            worst = []
            for n in cfg["n_values"]:
                errors = [row["relative_error"] for row in rows if row["method"] == method
                          and row["n"] == n and row["locality"] == locality and row["cache_mode"] == mode
                          and row.get("relative_error") is not None and np.isfinite(row["relative_error"])]
                worst.append(max(settings.error_floor, max(errors)) if errors else np.nan)
            ax.plot(cfg["n_values"], worst, **line_style(SERIES[method], settings))
        gate = cfg["rtol"] + cfg["atol"]
        if gate > 0:
            ax.axhline(gate, color="0.4", linestyle=":", linewidth=0.8, label="Validation gate")
    finish_figure(fig, settings)
    return fig


def query_figure(cfg, summaries, settings):
    fig, panels = axes_grid(cfg, settings, "Oracle calls per conversion")
    for ax, locality, mode in panels:
        for metric, spec in QUERY_SERIES.items():
            median, _, _ = summary_curve(summaries, cfg, "spencer", locality, mode, metric)
            ax.plot(cfg["n_values"], median, **line_style(spec, settings))
    finish_figure(fig, settings)
    return fig


# ---- Exports: figures and captions only; the benchmark measurements are read-only. ----
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_caption(cfg):
    contract = {
        "anschuetz": "Common input: general Pauli-orbit coefficients. Common output: all dense Schur blocks.",
        "chang": "Common input: fixed-weight Pauli-orbit averages. Common output: all Schur blocks in CSR form.",
        "spencer": "Common input: entry-orbit coefficients. Common output: an explicit dictionary of Pauli terms.",
        "georges": ("Common input: entry-orbit coefficients. Common output: " +
                    ("a full Pauli coefficient array." if cfg["output_contract"] == "expanded"
                     else "compressed Pauli-orbit coefficients.")),
    }[cfg["comparison"]]
    public_note = (" The public implementation calls the pinned author's complete block-construction "
                   "routine on every trial, with import compatibility fixes only. It has no cached "
                   "curve because the public API exposes no reusable cache."
                   if "anschuetz_public_original" in cfg["methods"] else "")
    if "anschuetz_public_original" in cfg["methods"] and cfg.get("anschuetz_public_max_n"):
        public_note += (f" Public runs above n={cfg['anschuetz_public_max_n']} are deliberately "
                        "outside the configured range, not measured timeouts.")
    if cfg.get("fixed_weight_cache") == "orbit-images":
        public_note += (" All methods prepare the complete exact-weight orbit-image bank on the same "
                        "sparse band pattern and use the same subsequent combination routine. "
                        "First use here includes building every orbit image, not only converting "
                        "one operator with the native method. Temporary construction tables are "
                        "released after preparation. The sparse output retains structural band zeros.")
    return (f"{cfg['comparison'].capitalize()} comparison. Markers show medians; shading spans the "
            f"25th to 75th percentiles, pooling {cfg['repeats']} timed repetitions for each of "
            f"{cfg['instances']} seeded input instances. {cfg['threads']} numerical-library thread(s). "
            f"{contract} First use includes reusable setup; cached application excludes it. Both "
            "include input-representation preparation and output formatting. Input generation and "
            "validation are outside timing. A point is omitted, breaking the curve, if any recorded "
            "trial there failed, was guarded, or timed out. Shading describes observed variability, "
            "not a confidence interval. Connecting lines guide the eye; they are not fitted scaling laws."
            + public_note)


def save_figure(fig, output_dir, stem, caption, settings):
    import matplotlib.pyplot as plt
    artifacts = []
    try:
        for extension in settings.formats:
            path = output_dir / f"{stem}.{extension}"
            metadata = {"CreationDate": None} if extension == "pdf" else ({"Date": None} if extension == "svg" else None)
            # No tight crop: preserve the requested physical dimensions and font sizes.
            fig.savefig(path, dpi=settings.dpi, bbox_inches=None, metadata=metadata)
            artifacts.append({"name": path.name, "sha256": sha256(path)})
        return dict(stem=stem, caption=caption, files=artifacts,
                    size_inches=[float(value) for value in fig.get_size_inches()])
    finally:
        plt.close(fig)


def plot(input_dir, output_dir=None, *, settings=DEFAULTS):
    """Regenerate a run's figures. Existing callers plot(run) remain supported."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401 -- registers the repository's Matplotlib styles
    except ImportError as exc:
        raise ImportError("Install the plotting dependency: python3 -m pip install SciencePlots") from exc

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir or input_dir / "plots").resolve()
    cfg = json.loads((input_dir / "config.json").read_text())
    rows = [json.loads(line) for line in (input_dir / "runs.jsonl").read_text().splitlines() if line]
    summaries = summarize(rows)
    styles = ["default", *settings.styles] + ([] if settings.use_latex else ["no-latex"])
    # Style contexts avoid changing the user's global Matplotlib settings.
    rc = {"font.size": settings.font_size, "axes.labelsize": settings.font_size,
          "axes.titlesize": settings.font_size, "xtick.labelsize": settings.font_size,
          "ytick.labelsize": settings.font_size, "legend.fontsize": settings.font_size,
          "text.usetex": settings.use_latex, "pdf.fonttype": 42, "ps.fonttype": 42,
          "svg.fonttype": "none", "svg.hashsalt": "thesis-pauli-comparison",
          "savefig.bbox": None, "savefig.dpi": settings.dpi}
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.style.context(styles), matplotlib.rc_context(rc):
        figures = [save_figure(runtime_figure(cfg, summaries, settings), output_dir,
                              FIGURE_NAMES[cfg["comparison"]], runtime_caption(cfg), settings)]
        caption = (f"Largest recorded finite relative error, including failed numerical checks. Values "
                   f"below {settings.error_floor:g} are displayed at {settings.error_floor:g}; raw values "
                   "are unchanged. The dotted line is the relative-error gate; an elementwise gate is "
                   "also applied. Trials without a finite error estimate cannot be positioned on this "
                   "axis; consult runs.csv for every failure and guard status. See validation_kind for "
                   "the reference, including round-trip-only checks of larger general Schur cases.")
        figures.append(save_figure(accuracy_figure(cfg, rows, settings), output_dir, "accuracy", caption, settings))
        if cfg["comparison"] == "spencer":
            caption = ("Median Spencer oracle calls. A row query returns every nonzero entry of a row; "
                       "an entry query reads one entry. Total calls add these distinct access operations; "
                       "they do not represent equal amounts of work. Points require all recorded trials "
                       "there to pass. The thesis route reads a compressed table, so no equivalent "
                       "row-query count is assigned to it. Pauli support is recorded in runs.csv.")
            figures.append(save_figure(query_figure(cfg, summaries, settings), output_dir,
                                       "spencer_queries", caption, settings))
        resolved_series = {key: line_style(spec, settings) for key, spec in {**SERIES, **QUERY_SERIES}.items()}
        fonts = {key: plt.rcParams[key] for key in ("font.family", "font.serif", "mathtext.fontset", "text.usetex")}

    if len(cfg["cache_modes"]) > 1:
        modes = {"cold": "setup included", "cached": "reusable data cached"}
        panel_caption = " Columns, from left to right: " + "; ".join(modes[mode] for mode in cfg["cache_modes"]) + "."
        for figure in figures:
            figure["caption"] += panel_caption

    manifest = {"schema": "thesis-comparison-figures-v2", "data_file": str(input_dir / "runs.jsonl"),
                "data_sha256": sha256(input_dir / "runs.jsonl"),
                "config_sha256": sha256(input_dir / "config.json"),
                "plot_code_sha256": sha256(__file__),
                "matplotlib_version": matplotlib.__version__, "scienceplots_version": version("SciencePlots"),
                "numpy_version": np.__version__, "styles": styles, "settings": asdict(settings),
                "resolved_series": resolved_series, "fonts": fonts, "figures": figures}
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    latex = ["% Requires \\usepackage{graphicx}. Keep this file and the PDFs together.",
             "% For a two-column article, use figure* for figures wider than one column.",
             "% Native physical widths below preserve the chosen font size; adapt to your template."]
    if "pdf" in settings.formats:
        for figure in figures:
            caption = figure["caption"].replace("_", "\\_")
            width, _ = figure["size_inches"]
            latex.extend(["\\begin{figure}[tb]", "  \\centering",
                          f"  \\includegraphics[width={width:g}in]{{{figure['stem']}.pdf}}",
                          f"  \\caption{{{caption}}}", f"  \\label{{fig:{cfg['comparison']}-{figure['stem']}}}",
                          "\\end{figure}", ""])
    (output_dir / "figures.tex").write_text("\n".join(latex) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dirs", type=Path, nargs="+", help="one or more saved run directories")
    parser.add_argument("--output-dir", type=Path, help="alternative figure folder; requires one input directory")
    parser.add_argument("--styles", nargs="+", default=DEFAULTS.styles, help="SciencePlots/Matplotlib styles, in order")
    parser.add_argument("--latex", action=argparse.BooleanOptionalAction, default=DEFAULTS.use_latex,
                        help="use external LaTeX; default is portable MathText")
    parser.add_argument("--width", type=float, default=DEFAULTS.width, help="total figure width in inches")
    parser.add_argument("--height", type=float, default=DEFAULTS.height, help="total figure height in inches")
    parser.add_argument("--panel-width", type=float, default=DEFAULTS.panel_width, help="inches per subplot column")
    parser.add_argument("--panel-height", type=float, default=DEFAULTS.panel_height, help="inches per subplot row")
    parser.add_argument("--font-size", type=float, default=DEFAULTS.font_size, help="text size in points")
    parser.add_argument("--legend-columns", type=int, default=DEFAULTS.legend_columns)
    parser.add_argument("--xscale", choices=("log", "linear"), default=DEFAULTS.xscale)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=DEFAULTS.grid)
    parser.add_argument("--dpi", type=int, default=DEFAULTS.dpi, help="PNG resolution")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=DEFAULTS.formats)
    args = parser.parse_args()
    if args.output_dir and len(args.input_dirs) != 1:
        parser.error("--output-dir requires exactly one input directory")
    for name in ("width", "height", "panel_width", "panel_height", "font_size", "dpi", "legend_columns"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name.replace('_', '-')} must be positive and finite")
    overrides = {name: getattr(args, name) for name in
                 ("width", "height", "panel_width", "panel_height", "font_size", "legend_columns", "xscale", "grid", "dpi")}
    settings = replace(DEFAULTS, **overrides, styles=tuple(args.styles), use_latex=args.latex,
                       formats=tuple(dict.fromkeys(args.formats)))
    for run in args.input_dirs:
        manifest = plot(run, args.output_dir, settings=settings)
        destination = (args.output_dir or run / "plots").resolve()
        print(f"{run.name}: {len(manifest['figures'])} SciencePlots figures saved to {destination}")


if __name__ == "__main__":
    main()
