"""Check the archived observations and reject incomplete benchmark summaries."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from pauli_transforms import plot


DATA = Path(__file__).resolve().parents[1] / "data/thesis"


class PlotTests(unittest.TestCase):
    def test_archive_hashes_and_trial_counts(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        for name, record in manifest["files"].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), record["sha256"], name)
        for name, expected in [("direct", 924), ("fixed_locality", 1344),
                               ("spectral", 300), ("dynamics", 420)]:
            _, rows = plot.read_campaign(DATA / name)
            self.assertEqual(sum(r.get("valid") is True for r in rows), expected)

    def test_failed_missing_duplicate_and_nonfinite_trials_rejected(self):
        cfg, all_rows = plot.read_campaign(DATA / "spectral")
        rows = [r for r in all_rows if r["n"] == 20 and r["method"] == "thesis"]
        for variant in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                plot.point_statistics(variant, cfg, "total_seconds", "n", [20], method="thesis")
        for changes in ({"valid": False}, {"total_seconds": float("nan")},
                        {"total_seconds": -1}, {"status": "timeout"}):
            variant = copy.deepcopy(rows)
            variant[0].update(changes)
            with self.assertRaises(ValueError):
                plot.point_statistics(variant, cfg, "total_seconds", "n", [20], method="thesis")

    def test_eight_figures_and_recorded_medians(self):
        expected_names = {"general_conversion", "general_conversion_public", "conversion_accuracy",
                          "fixed_locality", "spectral_comparison", "random_dynamics_comparison",
                          "random_dynamics_diagnostics", "ising_example"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summaries = plot.export(DATA, output, formats=("pdf",))
            self.assertEqual(set(summaries), expected_names)
            self.assertEqual({p.stem for p in output.glob("*.pdf")}, expected_names)
            self.assertAlmostEqual(summaries["spectral_comparison"]["thesis"][-1]["median"],
                                   0.017852479999419302, places=15)
            self.assertAlmostEqual(summaries["random_dynamics_comparison"]["size/thesis"][-1]["median"],
                                   0.049764150000555674, places=15)
            self.assertAlmostEqual(summaries["general_conversion"]["separated/ell0/cold"][-1]["median"],
                                   0.017120777, places=15)
            self.assertNotIn("anschuetz_public_original/ell0/cached", summaries["general_conversion_public"])
            for stats in summaries["ising_example"].values():
                self.assertEqual(stats["points"], 241)
                self.assertAlmostEqual(stats["initial"], .6, places=12)

    def test_small_grid_without_public_measurements(self):
        cfg, rows = plot.read_campaign(DATA / "direct")
        cfg.pop("n")
        cfg["n_values"] = [2]
        cfg["methods"] = ["separated", "anschuetz_optimized"]
        rows = [r for r in rows if r["n"] == 2 and r["method"] in cfg["methods"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "direct"
            campaign.mkdir()
            (campaign / "config.json").write_text(json.dumps(cfg))
            (campaign / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            figures = plot.export(root, root / "plots", formats=())
            self.assertEqual(set(figures), {"general_conversion", "conversion_accuracy"})
            self.assertEqual(figures["general_conversion"]["separated/ell0/cold"][0]["x"], 2)

    def test_failed_or_changed_ising_data_rejected(self):
        for change in ("failed", "missing", "spectrum_size", "curve_shift", "nonfinite_reference"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ising"
                shutil.copytree(DATA / "ising", path)
                if change == "failed":
                    receipt = json.loads((path / "validation.json").read_text())
                    receipt["all_passed"] = False
                    (path / "validation.json").write_text(json.dumps(receipt))
                elif change == "missing":
                    (path / "validation.json").unlink()
                elif change == "spectrum_size":
                    cfg = json.loads((path / "config.json").read_text())
                    cfg["spectrum_n"] = 14
                    (path / "config.json").write_text(json.dumps(cfg))
                else:
                    with (path / "selected_curves.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    if change == "curve_shift":
                        rows[1]["value_real"] = float(rows[1]["value_real"]) + .1
                    else:
                        rows[1]["reference_real"] = "nan"
                    with (path / "selected_curves.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    plot.ising_figure(path)


if __name__ == "__main__":
    unittest.main()
