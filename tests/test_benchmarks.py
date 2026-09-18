"""Input normalization, independent references, and benchmark timing records."""

import contextlib
import io
import json
from math import comb, factorial
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pauli_transforms import benchmark, benchmark_cases, benchmark_helpers, common
from test_transforms import explicit_pauli_matrix, explicit_schur_blocks


def word_count(n, key):
    w, z, y = key
    counts = (n-w-z, w-y, y, z)
    result = factorial(n)
    for count in counts:
        result //= factorial(count)
    return result


class BenchmarkTests(unittest.TestCase):
    def test_general_input_has_stated_hilbert_schmidt_normalization(self):
        n, seed = 4, 331
        coefficients = benchmark_cases.make_case(n, "general", seed, 0)
        self.assertEqual(len(coefficients), comb(n+3, 3))
        self.assertTrue(all(np.isreal(value) for value in coefficients.values()))
        dense = explicit_pauli_matrix(n, coefficients)
        weighted_norm = sum(word_count(n, key)*abs(value)**2
                            for key, value in coefficients.items())
        gaussian_norm = np.mean(np.random.default_rng(seed).normal(size=len(coefficients))**2)
        self.assertAlmostEqual(np.vdot(dense, dense).real / 2**n, weighted_norm)
        self.assertAlmostEqual(weighted_norm, gaussian_norm)

    def test_fixed_weight_inputs_use_a_fixed_number_of_orbit_averages(self):
        for locality in (2, 4):
            first = benchmark_cases.make_case(5, "fixed_weight", 919, locality)
            second = benchmark_cases.make_case(11, "fixed_weight", 919, locality)
            self.assertEqual(set(first), set(second))
            self.assertEqual(len(first), comb(locality+2, 2))
            for key, value in first.items():
                self.assertEqual(key[0]+key[1], locality)
                amplitude = value*word_count(5, key)*len(first)
                self.assertGreaterEqual(abs(amplitude), .5)
                self.assertLessEqual(abs(amplitude), 1.5)
                self.assertAlmostEqual(amplitude,
                                       second[key]*word_count(11, key)*len(second))

    def test_collective_reference_against_independent_dense_projection(self):
        for n in range(1, 5):
            coefficients = {key: (-.3+.2j)/(index+1)
                            for index, key in enumerate(common.all_orbit_keys(n))}
            expected = explicit_schur_blocks(n, explicit_pauli_matrix(n, coefficients))
            actual = benchmark_cases.collective_reference(n, coefficients)
            for block, reference in zip(actual, expected):
                np.testing.assert_allclose(block.toarray(), reference, atol=2e-12, rtol=2e-12)

    def test_archived_input_reuse_and_separate_timing_stages(self):
        with tempfile.TemporaryDirectory(prefix="pauli_benchmark_test_") as temporary:
            root = Path(temporary)
            saved = root/"saved"
            saved.mkdir()
            coefficients = benchmark_cases.make_case(2, "general", 783, 0)
            coefficients[0, 0, 0] = float(np.nextafter(.3, 1.))
            payload = dict(n=2, family="general", locality=0, seed=783,
                           coefficients=benchmark_cases.encode_mapping(coefficients))
            filename = "n2_general_l0_i0.json"
            benchmark_helpers.write_json(saved/filename, payload)
            output = root/"run"
            with contextlib.redirect_stdout(io.StringIO()):
                status = benchmark.main(["--n", "2", "--instances", "1", "--repeats", "1",
                                         "--warmup", "0", "--input-directory", str(saved),
                                         "--output", str(output)])
            self.assertEqual(status, 0)
            restored = json.loads((output/"inputs"/filename).read_text())
            self.assertEqual(restored, payload)
            self.assertEqual(benchmark_cases.decode_mapping(restored["coefficients"]), coefficients)
            rows = benchmark_helpers.read_jsonl(output/"runs.jsonl")
            self.assertEqual({(row["method"], row["cache_mode"]) for row in rows},
                             {(method, mode) for method in ("separated", "anschuetz_optimized")
                              for mode in ("cold", "cached")})
            for row in rows:
                self.assertTrue(row["valid"])
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["input_seed"], 783)
                self.assertEqual(row["input_hash"], benchmark_helpers.sha256(output/"inputs"/filename))
                self.assertEqual(row["output_contract"], "dense_schur_blocks")
                self.assertEqual(row["validation_kind"], "independent_dense_singlet_dicke_projection")
                self.assertEqual(row["threads"], 1)
                self.assertGreaterEqual(row["reference_seconds"], 0)
                self.assertGreaterEqual(row["validation_seconds"], 0)
                stages = [row[name] for name in
                          ("preparation_seconds", "setup_seconds", "apply_seconds", "output_seconds")]
                self.assertTrue(all(value >= 0 for value in stages))
                self.assertAlmostEqual(sum(stages), row["elapsed_seconds"])
                if row["cache_mode"] == "cached":
                    self.assertGreaterEqual(row["cache_build_seconds"], 0)
                    self.assertEqual(row["setup_seconds"], 0)
            self.assertTrue((output/"source_hashes.json").is_file())
            self.assertTrue((output/"environment.json").is_file())

    def test_fixed_weight_runner_validates_the_common_csr_output(self):
        with tempfile.TemporaryDirectory(prefix="pauli_chang_test_") as temporary:
            output = Path(temporary)/"run"
            with contextlib.redirect_stdout(io.StringIO()):
                status = benchmark.main(["--comparison", "chang", "--n", "2", "--localities", "2",
                                         "--instances", "1", "--repeats", "1", "--warmup", "0",
                                         "--output", str(output)])
            self.assertEqual(status, 0)
            rows = benchmark_helpers.read_jsonl(output/"runs.jsonl")
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertTrue(row["valid"])
                self.assertEqual(row["output_contract"], "banded_schur_CSR")
                self.assertEqual(row["validation_kind"], "independent_collective_pauli_recurrence")

    def test_timeout_preserves_completed_samples(self):
        with tempfile.TemporaryDirectory(prefix="pauli_timeout_test_") as temporary:
            folder = Path(temporary)/"job"
            job = dict(job_id="interrupted", method="separated", n=2,
                       cache_mode="cold", config={"timeout": 1.})
            completed = dict(job_id="interrupted", repeat=0, status="ok", valid=True)

            def interrupted(*args, **kwargs):
                (folder/"samples.jsonl").write_text(json.dumps(completed)+"\n")
                raise subprocess.TimeoutExpired("worker", 1.)

            with patch.object(benchmark.subprocess, "run", side_effect=interrupted):
                rows = benchmark.run_job(job, folder)
            self.assertEqual(rows[0], completed)
            self.assertEqual(rows[1]["status"], "timeout")
            self.assertFalse(rows[1]["valid"])

    def test_worker_rejects_a_cached_public_job(self):
        with tempfile.TemporaryDirectory(prefix="pauli_public_mode_test_") as temporary:
            path = Path(temporary)/"job.json"
            benchmark_helpers.write_json(path, dict(n=2, method="anschuetz_public_original",
                                                   cache_mode="cached", config={}))
            with self.assertRaisesRegex(ValueError, "no cached interface"):
                benchmark.worker(path)

    @unittest.skipUnless(os.environ.get("ANSCHUETZ_SOURCE"), "optional pinned public checkout")
    def test_public_comparator_is_fresh_only_with_an_explicit_size_limit(self):
        with tempfile.TemporaryDirectory(prefix="pauli_public_test_") as temporary:
            output = Path(temporary)/"run"
            with contextlib.redirect_stdout(io.StringIO()):
                status = benchmark.main(["--n", "2", "6", "--instances", "1", "--repeats", "1",
                                         "--warmup", "0", "--output", str(output),
                                         "--anschuetz-source", os.environ["ANSCHUETZ_SOURCE"]])
            self.assertEqual(status, 0)
            rows = [row for row in benchmark_helpers.read_jsonl(output/"runs.jsonl")
                    if row["method"] == "anschuetz_public_original"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["cache_mode"] == "cold" for row in rows))
            by_size = {row["n"]: row for row in rows}
            self.assertTrue(by_size[2]["valid"])
            self.assertEqual(by_size[2]["setup_seconds"], 0)
            self.assertEqual(by_size[6]["status"], "configured_size_limit")
            self.assertFalse(by_size[6]["valid"])


if __name__ == "__main__":
    unittest.main()
