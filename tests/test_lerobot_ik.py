"""Optional LeRobot backend checks; enable native tests with X2_TEST_LEROBOT=1."""

import contextlib
import io
import os
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from test_agibot_x2_simulation import ASSETS, FakeXR
from xrobotoolkit_teleop.common.lerobot_ik import LeRobotIKBackend
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController


class BackendSelectionTests(unittest.TestCase):
    def test_default_backend_does_not_load_optional_dependencies(self):
        with (
            patch(
                "xrobotoolkit_teleop.common.lerobot_ik.load_dependencies", side_effect=AssertionError("optional import")
            ),
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient", FakeXR),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            c = AgibotX2TeleopController(str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"))
        self.addCleanup(c.xr_client.close)
        self.assertEqual(c.ik_backend, "placo")
        self.assertIsNone(c.lerobot_ik)
        self.assertIsNotNone(c.collision_checker)

    def test_invalid_backend_fails_before_xr_connection(self):
        with patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient") as xr:
            with self.assertRaisesRegex(ValueError, "ik_backend"):
                AgibotX2TeleopController("unused", "unused", ik_backend="unknown")
            xr.assert_not_called()

    def test_missing_dependencies_fail_before_xr_connection(self):
        with (
            patch("xrobotoolkit_teleop.common.lerobot_ik.load_dependencies", side_effect=ImportError("test")),
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient") as xr,
        ):
            with self.assertRaises(ImportError):
                AgibotX2TeleopController("unused", "unused", ik_backend="lerobot")
            xr.assert_not_called()


@unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires optional Pinocchio/CasADi build")
class LeRobotIKTests(unittest.TestCase):
    def make_controller(self, omnihand=False, **kwargs):
        stem, scene = ("x2_omnihand_ik", "scene_omnihand") if omnihand else ("x2_fixed", "scene")
        with (
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient", FakeXR),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            c = AgibotX2TeleopController(
                str(ASSETS / f"{scene}.xml"), str(ASSETS / f"{stem}.urdf"), ik_backend="lerobot", **kwargs
            )
        self.addCleanup(c.xr_client.close)
        self.addCleanup(c.lerobot_ik.close)
        c.xr_client.grips.update(left_trigger=0.0, right_trigger=0.0)
        return c

    def inputs(self, c):
        measured = {name: c.placo_robot.get_joint(name) for name in c.arm_actuators.values()}
        return measured, c.lerobot_ik.forward_kinematics(measured)

    def test_fk_matches_both_mujoco_models_and_joint_order(self):
        for omni in (False, True):
            with self.subTest(omnihand=omni):
                c = self.make_controller(omni)
                measured, _ = self.inputs(c)
                rng = np.random.default_rng(42)
                for _ in range(10):
                    for name in measured:
                        measured[name] += rng.uniform(-0.03, 0.03)
                        c.mj_data.qpos[c.mj_model.joint(name).qposadr[0]] = measured[name]
                    mujoco.mj_forward(c.mj_model, c.mj_data)
                    for side, pose in c.lerobot_ik.forward_kinematics(measured).items():
                        body = c.mj_data.body(f"{side}_hand_center")
                        np.testing.assert_allclose(pose[:3, 3], body.xpos, atol=1e-5)
                        np.testing.assert_allclose(pose[:3, :3], body.xmat.reshape(3, 3), atol=1e-5)

    def test_solution_matches_independent_numeric_reference(self):
        # Independent finite-difference least squares uses numeric Pinocchio FK,
        # rather than the backend's symbolic error functions or IPOPT graph.
        from scipy.optimize import least_squares

        c = self.make_controller()
        backend = c.lerobot_ik
        measured, targets = self.inputs(c)
        targets["left"][0, 3] += 0.02
        targets["right"][2, 3] += 0.01
        seed = backend._vector(measured)
        data = backend.model.createData()

        def residual(q):
            backend.pin.framesForwardKinematics(backend.model, data, q)
            ep, er = [], []
            for side, fid in backend.frame_ids.items():
                ep.extend(data.oMf[fid].translation - targets[side][:3, 3])
                er.extend(backend.pin.log3(data.oMf[fid].rotation @ targets[side][:3, :3].T))
            return np.r_[np.sqrt(50) * np.array(ep), er, np.sqrt(0.02) * q, np.sqrt(0.1) * (q - seed)]

        reference = least_squares(
            residual, seed, bounds=(backend.lower, backend.upper), ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
        actual = backend.solve(measured, measured, targets, {"left_arm": True, "right_arm": True})
        q = backend._vector(actual)
        np.testing.assert_allclose(q, reference.x, atol=1e-4, rtol=0)
        self.assertLess(abs(np.dot(residual(q), residual(q)) - 2 * reference.cost), 1e-8)
        limits = {
            name: (backend.lower[index], backend.upper[index])
            for name, index in zip(backend.joint_names, backend.indices)
        }
        reordered = LeRobotIKBackend(c.robot_urdf_path, list(reversed(backend.joint_names)), limits)
        self.addCleanup(reordered.close)
        result = reordered.solve(measured, measured, targets, {"left_arm": True, "right_arm": True})
        for name in actual:
            self.assertAlmostEqual(result[name], actual[name], places=7)

    def test_filter_failure_and_reset(self):
        c = self.make_controller()
        b = c.lerobot_ik
        measured, targets = self.inputs(c)
        seed = b._vector(measured)
        active = {"left_arm": True, "right_arm": True}
        samples = [seed + offset for offset in (0.0, 0.01, 0.02, 0.03)]
        for sample in samples:
            with patch.object(b.opti, "solve", return_value=Mock(value=Mock(return_value=sample))):
                result = b.solve(measured, measured, targets, active)
        np.testing.assert_allclose(b._vector(result), seed + 0.01, atol=1e-12)
        with patch.object(b.opti, "solve", return_value=Mock(value=Mock(return_value=samples[-1]))):
            duplicate = b.solve(measured, measured, targets, active)
        np.testing.assert_allclose(b._vector(duplicate), seed + 0.01, atol=1e-12)
        before = b._previous.copy()
        history = np.array(b._history)
        for failure in (RuntimeError("failed"), np.full(14, np.nan)):
            solve = (
                Mock(side_effect=failure)
                if isinstance(failure, Exception)
                else Mock(return_value=Mock(value=Mock(return_value=failure)))
            )
            with patch.object(b.opti, "solve", solve), self.assertRaises(RuntimeError):
                b.solve(measured, measured, targets, active)
            np.testing.assert_array_equal(b._previous, before)
            np.testing.assert_array_equal(np.array(b._history), history)
        b.reset()
        self.assertIsNone(b._previous)
        self.assertEqual(len(b._history), 0)

    def test_inactive_arm_and_joint_bounds_with_unreachable_target(self):
        c = self.make_controller()
        b = c.lerobot_ik
        measured, targets = self.inputs(c)
        targets["left"][0, 3] += 10
        try:
            result = b.solve(measured, measured, targets, {"left_arm": True, "right_arm": False})
        except RuntimeError:
            self.assertIsNone(b._previous)
            return  # A nonconverged unreachable solve is allowed to hold.
        for name, index in zip(b.joint_names, b.indices):
            self.assertGreaterEqual(result[name], b.lower[index])
            self.assertLessEqual(result[name], b.upper[index])
            if name.startswith("right_"):
                self.assertEqual(result[name], measured[name])

    def test_missing_ipopt_is_reported_before_xr_connection(self):
        with (
            patch("casadi.has_nlpsol", return_value=False),
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient") as xr,
        ):
            with self.assertRaisesRegex(ImportError, "IPOPT"):
                AgibotX2TeleopController("unused", "unused", ik_backend="lerobot")
            xr.assert_not_called()

    def test_reengagement_rebases_only_the_changed_arm_filter(self):
        c = self.make_controller()
        b = c.lerobot_ik
        measured, targets = self.inputs(c)
        seed = b._vector(measured)
        left = np.array([i for n, i in zip(b.joint_names, b.indices) if n.startswith("left_")])
        right = np.array([i for n, i in zip(b.joint_names, b.indices) if n.startswith("right_")])

        def solve(raw, positions, previous, active):
            with patch.object(b.opti, "solve", return_value=Mock(value=Mock(return_value=raw))):
                return b.solve(positions, previous, targets, active)

        for offset in (0, 0.01, 0.02, 0.03):
            previous = solve(seed + offset, measured, measured, {"left_arm": True, "right_arm": True})
        raw = seed + 0.04
        raw[left] = b._vector(previous)[left]
        released = solve(raw, measured, previous, {"left_arm": False, "right_arm": True})
        np.testing.assert_allclose(b._vector(released)[left], seed[left] + 0.01, atol=1e-12)
        changed = measured.copy()
        for name in changed:
            if name.startswith("left_"):
                changed[name] += 0.3
        raw = seed + 0.05
        raw[left] = b._vector(changed)[left] + 0.01
        engaged = solve(raw, changed, released, {"left_arm": True, "right_arm": True})
        np.testing.assert_allclose(b._vector(engaged)[left], seed[left] + 0.301, atol=1e-12)
        np.testing.assert_allclose(b._vector(engaged)[right], seed[right] + 0.03, atol=1e-12)

    def test_reengagement_rebases_after_a_failed_release_solve(self):
        c = self.make_controller()
        b = c.lerobot_ik
        measured, targets = self.inputs(c)
        active = {"left_arm": True, "right_arm": True}
        seed = b._vector(measured)
        for offset in (0, 0.01, 0.02, 0.03):
            raw = seed + offset
            with patch.object(b.opti, "solve", return_value=Mock(value=Mock(return_value=raw))):
                previous = b.solve(measured, measured, targets, active)
        history = np.array(b._history)
        with (
            patch.object(b.opti, "solve", side_effect=RuntimeError("release failure")),
            self.assertRaises(RuntimeError),
        ):
            b.solve(measured, previous, targets, {"left_arm": False, "right_arm": True})
        np.testing.assert_array_equal(np.array(b._history), history)
        changed = measured.copy()
        for name in changed:
            if name.startswith("left_"):
                changed[name] += 0.3
        with patch.object(b.opti, "solve", side_effect=RuntimeError("engage failure")), self.assertRaises(RuntimeError):
            b.solve(changed, previous, targets, active)
        np.testing.assert_array_equal(np.array(b._history), history)
        raw = b._vector(changed)
        with patch.object(b.opti, "solve", return_value=Mock(value=Mock(return_value=raw))):
            result = b.solve(changed, previous, targets, active)
        for name in changed:
            if name.startswith("left_"):
                self.assertAlmostEqual(result[name], changed[name])

    def test_controller_bypasses_placo_collision_and_speed_gates(self):
        for omni in (False, True):
            with self.subTest(omnihand=omni):
                c = self.make_controller(omni, arm_velocity_limit=0.0001)
                c.xr_client.grips["left_grip"] = 1.0
                c.step_control()
                held = c.mj_data.ctrl.copy()
                c.xr_client.poses["left_controller"][2] -= 0.03
                with (
                    patch.object(c.solver, "solve", side_effect=AssertionError("Placo tracking called")),
                    patch.object(c, "_validate_collision_motion", side_effect=AssertionError("collision gate called")),
                    patch.object(c, "_validate_combined_commands", side_effect=AssertionError("combined gate called")),
                ):
                    c.step_control()
                self.assertTrue(c._ik_valid)
                self.assertGreater(np.max(np.abs(c.mj_data.ctrl - held)), 0.0001 * c.dt)
                for i, name in c.arm_actuators.items():
                    if name.startswith("right_"):
                        self.assertEqual(c.mj_data.ctrl[i], held[i])

    def test_release_tracking_loss_failure_and_home(self):
        c = self.make_controller()
        c.xr_client.grips["left_grip"] = 1.0
        c.step_control()
        c.xr_client.poses["left_controller"][2] -= 0.02
        c.step_control()
        held = c.mj_data.ctrl.copy()
        for failure in (RuntimeError("test failure"), ValueError("invalid target")):
            with patch.object(c.lerobot_ik, "solve", side_effect=failure):
                c.step_control()
            self.assertFalse(c._ik_valid)
            np.testing.assert_array_equal(c.mj_data.ctrl, held)
        with patch.object(c, "_xr_stream_is_fresh", return_value=False):
            c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        self.assertIsNone(c.lerobot_ik._previous)
        c.xr_client.grips["left_grip"] = 0.0
        c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        c.xr_client.buttons["Y"] = True
        for _ in range(100):
            before = c.mj_data.ctrl.copy()
            c.step_control()
            self.assertLessEqual(np.max(np.abs(c.mj_data.ctrl - before)), c.return_velocity_limit * c.dt + 1e-9)
        np.testing.assert_allclose(c.mj_data.ctrl, c.mj_model.key("home").ctrl, atol=1e-7)
        self.assertIsNone(c.lerobot_ik._previous)


if __name__ == "__main__":
    unittest.main()
