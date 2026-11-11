"""Selective signed queries against the exhaustive native collision oracle."""

import importlib.util
import os
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter


@unittest.skipUnless(importlib.util.find_spec("coal"), "requires Coal Python bindings")
class CollisionSubsetTests(unittest.TestCase):
    backend = "placo"

    def make_retargeter(self, robot="x2_omnihand"):
        retargeter = ExternalStateRetargeter(robot=robot, ik_backend=self.backend)
        self.addCleanup(retargeter.close)
        return retargeter

    def test_subset_matches_exhaustive_and_bounds_never_hide_near_pairs(self):
        rng = np.random.default_rng(42)
        for profile in ("x2", "x2_omnihand"):
            r = self.make_retargeter(profile)
            robot, checker = r.robot, r.collision_checker
            limits = {
                joint.get("name"): (float(joint.find("limit").get("lower")), float(joint.find("limit").get("upper")))
                for joint in ET.parse(r.profile.spec.urdf_path).getroot().findall("joint")
                if joint.find("limit") is not None
            }
            for sample in range(24):
                if sample:
                    for name in r.joint_names:
                        robot.set_joint(name, rng.uniform(*limits[name]))
                for threshold in (r.collision_margin, r.collision_trigger):
                    checker.clear_cache()
                    capped = checker.clearances(threshold)
                    selected = np.array(list(checker._subset_witnesses), dtype=int)
                    records = robot.distance_subset(selected[::-1])
                    exact = checker.distances()
                    np.testing.assert_allclose(capped, np.minimum(exact, threshold), atol=1e-5, rtol=0)
                    self.assertTrue(np.all(capped[exact < threshold - 1e-6] < threshold))
                    np.testing.assert_allclose(
                        [record.min_distance for record in records], exact[selected[::-1]], atol=1e-5, rtol=0
                    )
                    for index, record in zip(selected[::-1], records, strict=True):
                        self.assertEqual((record.objA, record.objB), tuple(checker.pair_indices[index]))
                        oracle = checker._witnesses[index]
                        self.assertEqual((record.parentA, record.parentB), (oracle.parentA, oracle.parentB))

    def test_threshold_configuration_and_witness_cache_transitions(self):
        r = self.make_retargeter()
        robot, checker = r.robot, r.collision_checker
        first = robot.state.q.copy()
        with patch.object(robot, "distance_subset", wraps=robot.distance_subset) as query:
            checker.clearances(0.08)
            self.assertEqual(query.call_count, 1)
            checker.clearances(0.08, require_witnesses=True)
            self.assertEqual(query.call_count, 1)
            checker.clearances(0.02)
            self.assertEqual(query.call_count, 1)
            checker.clearances(0.02, require_witnesses=True)
            self.assertEqual(query.call_count, 2)
            checker.clearances(0.08, require_witnesses=True)
            self.assertEqual(query.call_count, 3)
            self.assertFalse(checker._distance_cache)
            self.assertIsNone(checker._reference_distances)
            robot.set_joint("left_shoulder_pitch_joint", 1.7)
            checker.linearized_clearance(0.08)
            robot.state.q = first
            near, jacobian = checker.linearized_clearance(0.08)
            self.assertEqual(query.call_count, 5)
            checker.clear_cache()
            exact = checker.distances()
            oracle_near, oracle_jacobian = checker.linearized_clearance(0.08)
            np.testing.assert_allclose(near, oracle_near, atol=1e-5, rtol=0)
            # Parallel flat faces can have different valid witness subgradients.
            self.assertEqual(jacobian.shape, oracle_jacobian.shape)
            self.assertEqual(len(near), np.count_nonzero(exact < 0.08))
            checker.clear_cache()
            checker.clearances(0.08)
            self.assertEqual(query.call_count, 6)

    def test_subset_indices_are_integers_in_requested_order(self):
        r = self.make_retargeter()
        robot = r.robot
        self.assertEqual(robot.distance_subset([]), [])
        for values in ([0.5], [-0.5], ["0"]):
            with self.assertRaises(TypeError):
                robot.distance_subset(values)
        for values in ([-1], [r.collision_checker.pair_count]):
            with self.assertRaises(ValueError):
                robot.distance_subset(values)
        records = robot.distance_subset([np.int64(1), 0, 1])
        exact = robot.distances()
        for record, index in zip(records, (1, 0, 1), strict=True):
            self.assertEqual((record.objA, record.objB), (exact[index].objA, exact[index].objB))
            self.assertAlmostEqual(record.min_distance, exact[index].min_distance, delta=1e-5)

    def test_guarded_threshold_selects_borderline_pairs(self):
        r = self.make_retargeter()
        checker = r.collision_checker
        bounds = np.full(checker.pair_count, 0.1)
        bounds[:3] = [0.08 - 1e-7, 0.08, 0.08 + 0.5e-6]
        with patch.object(checker, "clearance_bounds", return_value=bounds):
            with patch.object(r.robot, "distance_subset", wraps=r.robot.distance_subset) as query:
                checker.clearances(0.08)
                np.testing.assert_array_equal(query.call_args.args[0], [0, 1, 2])

    def test_capped_motion_references_remain_conservative(self):
        r = self.make_retargeter()
        checker, robot = r.collision_checker, r.robot
        rng = np.random.default_rng(23)
        for pitch in (0.35, 2.0):
            robot.set_joint("left_shoulder_pitch_joint", pitch)
            robot.set_joint("left_shoulder_roll_joint", -0.06)
            checker.clear_cache()
            checker.clearances(0.08)
            self.assertIsNone(checker._reference_distances)
            reference = robot.state.q.copy()
            for _ in range(16):
                robot.state.q = reference.copy()
                robot.state.q[7:] += rng.uniform(-0.005, 0.005, len(reference) - 7)
                bounds = checker.clearance_bounds(np.full(checker.pair_count, 0.02))
                # Query the adapter directly so the exact oracle does not seed
                # or conceal the capped reference under test.
                exact = np.array([d.min_distance for d in robot.distances()])
                self.assertTrue(np.all(bounds <= np.maximum(exact, 0) + 2e-6))
                self.assertTrue(np.all(exact[bounds >= 0.02] >= 0.02 - 1e-6))

    def test_selective_linearization_matches_exhaustive_witnesses(self):
        r = self.make_retargeter()
        checker = r.collision_checker
        for pitch in (0.35, 1.7, 2.0):
            r.robot.set_joint("left_shoulder_pitch_joint", pitch)
            r.robot.set_joint("left_wrist_roll_joint", 0.17)
            r.robot.set_joint("right_wrist_roll_joint", -0.21)
            checker.clear_cache()
            values, jacobian = checker.linearized_clearance(0.08)
            witnesses = checker._subset_witnesses.copy()
            checker.clear_cache()
            exact = checker.distances()
            oracle_values, oracle_jacobian = checker.linearized_clearance(0.08)
            np.testing.assert_allclose(values, oracle_values, atol=1e-5, rtol=0)
            for row, index in enumerate(np.flatnonzero(exact < 0.08)):
                first, second = witnesses[index], checker._witnesses[index]
                a = np.array([first.pointA, first.pointB])
                b = np.array([second.pointA, second.pointB])
                if abs(values[row]) > 1e-3 and np.allclose(a, b, atol=1e-7, rtol=0):
                    np.testing.assert_allclose(jacobian[row], oracle_jacobian[row], atol=2e-4, rtol=0)
                else:
                    # Flat and symmetric surfaces admit different witnesses and
                    # subgradients; both must realize the signed distance.
                    self.assertAlmostEqual(np.linalg.norm(a[1] - a[0]), abs(values[row]), delta=1e-5)
                    self.assertAlmostEqual(np.linalg.norm(b[1] - b[0]), abs(oracle_values[row]), delta=1e-5)

    def test_fallback_and_invalid_results(self):
        r = self.make_retargeter()
        checker = r.collision_checker
        with patch.object(r.robot, "distance_subset", None):
            np.testing.assert_array_equal(checker.clearances(0.08), checker.distances())
        checker.clear_cache()
        with patch.object(checker, "can_bound", False):
            np.testing.assert_array_equal(checker.clearances(0.08), checker.distances())
        checker.clear_cache()
        with patch.object(r.robot, "distance_subset", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                checker.clearances(0.08)
        native_subset = r.robot.distance_subset

        def nonfinite(indices):
            return [replace(record, min_distance=np.nan) for record in native_subset(indices)]

        with patch.object(r.robot, "distance_subset", side_effect=nonfinite):
            with self.assertRaisesRegex(RuntimeError, "non-finite collision distances"):
                checker.clearances(0.08)
        for threshold in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                checker.clearances(threshold)
        r.robot.state.q[7] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            checker.clearances(0.08)

    def test_motion_validation_uses_exhaustive_queries_without_primitive_bounds(self):
        r = self.make_retargeter()
        robot, checker = r.robot, r.collision_checker
        # The bridge fixture supplies a clear deployment posture, so this
        # exercises the positive-floor branch that would otherwise use bounds.
        from test_robojudo_x2 import fixture

        measured, _ = fixture(r)
        for name, value in measured.items():
            robot.set_joint(name, value)
        configuration = robot.state.q.copy()
        checker.clear_cache()
        self.assertGreater(checker.distances().min(), r.collision_margin)
        checker.clear_cache()
        with (
            patch.object(checker, "can_bound", False),
            patch.object(checker, "clearance_bounds", side_effect=AssertionError("unsupported bounds")),
            patch.object(robot, "distance_subset", side_effect=AssertionError("unsupported subset")),
            patch.object(robot, "distances", wraps=robot.distances) as exact,
        ):
            r._validate_motion(configuration, configuration.copy(), configuration.copy())
            self.assertGreater(exact.call_count, 0)


@unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires native LeRobot dependencies")
class LeRobotCollisionSubsetTests(CollisionSubsetTests):
    backend = "lerobot"


class PlacoSubsetFallbackTests(unittest.TestCase):
    def test_missing_python_bindings_preserves_exhaustive_queries(self):
        with patch.dict("sys.modules", {"coal": None}):
            r = ExternalStateRetargeter(robot="x2", ik_backend="placo")
        self.addCleanup(r.close)
        self.assertIsNone(r.robot.distance_subset)
        checker = r.collision_checker
        np.testing.assert_array_equal(checker.clearances(0.08), checker.distances())


if __name__ == "__main__":
    unittest.main()
