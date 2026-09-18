"""Scientific and reporting checks for the random invariant-dynamics campaign."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pauli_transforms import physical_models as pm, random_dynamics_inputs as random_inputs
from pauli_transforms import run_random_dynamics_comparison as campaign
from pauli_transforms.common import block_shapes, specht_multiplicities


def configuration(**overrides):
    cfg = dict(schur_backend="hahn", schur_dtype="float64", acceptance_tolerance=1e-8,
               dense_max_n=5, seed=20260918, tmax=1.7, warmup=0, repeats=1)
    cfg.update(overrides)
    return cfg


class InputPhysicsTests(unittest.TestCase):
    def test_wishart_states_have_the_prescribed_physical_sector_probabilities(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                inputs, state = random_inputs.make_inputs(n, 70 + n)
                weights, sides = specht_multiplicities(n), block_shapes(n)
                for mu, side, block in zip(weights, sides, state, strict=True):
                    self.assertGreater(np.linalg.eigvalsh(block)[0], 0.)
                    self.assertAlmostEqual((mu * np.trace(block)).real, mu * side / 2**n)
                self.assertAlmostEqual(pm.block_trace(n, state).real, 1.)
                regenerated, state2 = random_inputs.make_inputs(n, 70 + n)
                self.assertEqual(inputs, regenerated)
                for a, b in zip(state, state2, strict=True):
                    np.testing.assert_array_equal(a, b)

    def test_independent_appendix_e_and_full_matrices_recover_the_wishart_state(self):
        for n in (2, 3, 5):
            with self.subTest(n=n):
                inputs, state = random_inputs.make_inputs(n, 101 + n)
                refs, dense, metrics = random_inputs.independent_reference(n, inputs, state)
                self.assertLess(max(metrics.values()), 1e-11)
                self.assertEqual(dense[0].shape, (2**n, 2**n))
                self.assertIs(refs[1], state)

    def test_relative_negativity_detects_an_invalid_tiny_sector(self):
        n = 20
        state = [np.eye(side, dtype=complex) / 2**n for side in block_shapes(n)]
        # A relative order-one defect can have small unweighted eigenvalues.
        state[0][0, 0] = -1 / 2**n
        diagnostics = random_inputs.state_diagnostics(n, state)
        self.assertGreater(diagnostics["state_relative_negative_mass"], .1)
        with self.assertRaisesRegex(ArithmeticError, "negative"):
            random_inputs.require_valid(diagnostics, 1e-8, "negative state")

    def test_product_state_control_uses_explicit_pauli_coefficients(self):
        n = 3
        inputs, _ = random_inputs.make_inputs(n, 42)
        state_coefficients = pm.product_state_pauli(n, (.2, -.3, .4))
        from pauli_transforms import dense_reference
        full_state = dense_reference.dense_operator(n, state_coefficients)
        one_site = np.array([[1.4, .2 + .3j], [.2 - .3j, .6]], dtype=complex) / 2
        np.testing.assert_allclose(full_state, np.kron(np.kron(one_site, one_site), one_site), atol=1e-15)
        original = [basis.conj() @ full_state @ basis.T for basis in dense_reference.prepare(n)]
        _, _, checks = random_inputs.independent_reference(n, (inputs[0], state_coefficients, inputs[2]), original)
        self.assertLess(max(checks.values()), 1e-12)


class CurveAndTimingTests(unittest.TestCase):
    def test_both_methods_agree_with_full_exponentials_at_every_requested_time(self):
        for n in (2, 3, 4):
            inputs, state = random_inputs.make_inputs(n, 330 + n)
            refs, dense, _ = random_inputs.independent_reference(n, inputs, state)
            times = np.array([0., .17, .63, 1.7])
            full_curve = random_inputs.full_space_curve(dense, times)
            scale = max(1., np.linalg.norm(dense[2], 2))
            for method in ("thesis", "anschuetz_optimized"):
                with self.subTest(n=n, method=method):
                    blocks, spectrum, values, phases = campaign.timed_sample(n, method, inputs, times, configuration())
                    np.testing.assert_allclose(values, full_curve, rtol=0, atol=1e-11)
                    diagnostics, error = campaign.trial_diagnostics(n, inputs, blocks, spectrum, values, refs, full_curve, scale)
                    self.assertLess(max(diagnostics.values()), 1e-11)
                    self.assertLess(error, 1e-11)
                    self.assertAlmostEqual(sum(phases[key] for key in campaign.PHASES[:-1]), phases["total_seconds"])
                    self.assertTrue(all(value >= 0 for value in phases.values()))

    def test_worker_loads_common_saved_inputs_and_records_full_space_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "inputs").mkdir()
            cfg = configuration(output=tmp)
            campaign.prepare_saved_input(tmp, 3, 0, cfg)
            payload = campaign.worker(dict(kind="size", n=3, points=4, instance=0, method="thesis"), cfg)
            row, = payload["rows"]
            self.assertTrue(campaign.accepted(row))
            self.assertLess(row["full_space_curve_error"], 1e-12)
            self.assertEqual(len(payload["curves"]), 4)
            json.dumps(payload, allow_nan=False)

    def test_curve_check_counts_imaginary_errors_and_rejects_nonfinite_data(self):
        self.assertAlmostEqual(campaign.curve_error([0., 2e-6j], [0., 0.]), 2e-6)
        for actual, expected in (([np.nan], [0.]), ([0.], [np.inf]), ([0.], [0., 0.])):
            self.assertEqual(campaign.curve_error(actual, expected), float("inf"))

    def test_invalid_warmup_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "inputs").mkdir()
            cfg = configuration(output=tmp, warmup=1)
            campaign.prepare_saved_input(tmp, 2, 0, cfg)
            real_sample = campaign.timed_sample
            def inaccurate(*args, **kwargs):
                blocks, spectrum, values, phases = real_sample(*args, **kwargs)
                return blocks, spectrum, values + .01, phases
            with patch.object(campaign, "timed_sample", side_effect=inaccurate):
                with self.assertRaisesRegex(ArithmeticError, "Warmup failed"):
                    campaign.worker(dict(kind="size", n=2, points=3, instance=0, method="thesis"), cfg)

    def test_wrong_phase_sign_and_omitted_multiplicities_fail_curve_check(self):
        n = 4
        inputs, state = random_inputs.make_inputs(n, 934)
        references, dense, _ = random_inputs.independent_reference(n, inputs, state)
        times = np.array([0., .41, 1.7])
        full_curve = random_inputs.full_space_curve(dense, times)
        blocks, spectrum, _, _ = campaign.timed_sample(n, "thesis", inputs, times, configuration())
        wrong_phase = pm.expectation_time_series(n, spectrum, blocks[1], blocks[2], -times)
        divided_state = [block / mu for block, mu in zip(blocks[1], specht_multiplicities(n), strict=True)]
        missing_mu = pm.expectation_time_series(n, spectrum, divided_state, blocks[2], times)
        for corrupt in (wrong_phase, missing_mu):
            diagnostics, _ = campaign.trial_diagnostics(n, inputs, blocks, spectrum, corrupt,
                                                        references, full_curve, max(1., np.linalg.norm(dense[2], 2)))
            self.assertGreater(diagnostics["scaled_curve_error"], 1e-5)


class ReportingTests(unittest.TestCase):
    def row(self, **updates):
        row = dict(kind="size", n=3, points=4, method="thesis", instance=0, repeat=0,
                   valid=True, status="ok", setup_seconds=.1, conversion_seconds=.2,
                   eigensolve_seconds=.3, curve_seconds=.4, total_seconds=1.,
                   max_abs_error=1e-14, scaled_curve_error=1e-14)
        row.update(updates)
        return row

    def test_incomplete_or_failed_groups_have_no_runtime_median(self):
        for rows, expected in (([self.row()], 2),
                               ([self.row(), self.row(repeat=1, valid=False, status="failed")], 2),
                               ([self.row(total_seconds=np.inf)], 1),
                               ([self.row(total_seconds=.5)], 1),
                               ([self.row(scaled_curve_error=1e-3)], 1)):
            summary, = campaign.summarize(rows, expected)
            self.assertFalse(summary["valid"])
            self.assertNotIn("total_seconds_median", summary)

    def test_size_and_time_sweeps_remain_separate(self):
        summaries = campaign.summarize([self.row(), self.row(kind="times")], 1)
        self.assertEqual({row["kind"] for row in summaries}, {"size", "times"})
        self.assertTrue(all(row["valid"] for row in summaries))


if __name__ == "__main__":
    unittest.main()
