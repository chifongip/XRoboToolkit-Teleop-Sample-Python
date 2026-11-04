"""Headless X2 checks. Run with: python -m unittest discover -s tests -v."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from meshcat import transformations as tf
import placo

from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
from xrobotoolkit_teleop.utils.mujoco_utils import calc_placo_q_from_mujoco_qpos


ASSETS = Path(__file__).resolve().parents[1] / "assets/agibot/x2"


class FakeXR:
    def __init__(self):
        self.grips = {"left_grip": 0.0, "right_grip": 0.0}
        self.poses = {f"{side}_controller": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for side in ("left", "right")}
        self.poses["headset"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.timestamp_ns = 1
        self.advance_timestamps = True
        self.closed = False

    def get_key_value_by_name(self, name):
        return self.grips[name]

    def get_pose_by_name(self, name):
        return self.poses[name].copy()

    def get_timestamp_ns(self):
        if self.advance_timestamps:
            self.timestamp_ns += 10_000_000
        return self.timestamp_ns

    def close(self):
        self.closed = True


class X2SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ASSETS / "scene.xml"))
        cls.robot = placo.RobotWrapper(str(ASSETS / "x2_fixed.urdf"))

    def make_controller(self, **kwargs):
        with (
            patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient", FakeXR),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            controller = AgibotX2TeleopController(str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"), **kwargs)
        self.addCleanup(controller.xr_client.close)
        return controller

    def test_model_structure_and_frame_alignment(self):
        m, robot = self.model, self.robot
        self.assertEqual((m.nq, m.nv, m.nu), (14, 14, 14))
        self.assertEqual(set(robot.joint_names()), {m.joint(i).name for i in range(m.njnt)})
        self.assertTrue(np.all(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
        self.assertEqual(m.nmocap, 2)
        self.assertEqual(len(set(m.actuator_trnid[:, 0])), 14)
        data = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, data, m.key("home").id)
        home = data.qpos.copy()
        rng = np.random.default_rng(5)
        for q in [home] + [
            np.clip(home + rng.uniform(-0.2, 0.2, 14), m.jnt_range[:, 0], m.jnt_range[:, 1]) for _ in range(4)
        ]:
            data.qpos[:] = q
            mujoco.mj_forward(m, data)
            robot.state.q = calc_placo_q_from_mujoco_qpos(m, robot, q)
            robot.update_kinematics()
            for side in ("left", "right"):
                name = f"{side}_wrist_roll_link"
                target = robot.get_T_world_frame(name)
                body = m.body(name).id
                self.assertLess(np.linalg.norm(target[:3, 3] - data.xpos[body]), 1e-4)
                delta = target[:3, :3].T @ data.xmat[body].reshape(3, 3)
                angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
                self.assertLess(angle, 1e-3)

    def test_ik_respects_user_and_urdf_joint_velocity_limits(self):
        limits = {
            joint.get("name"): float(joint.find("limit").get("velocity"))
            for joint in ET.parse(ASSETS / "x2_fixed.urdf").getroot().findall("joint")
            if joint.get("type") != "fixed"
        }
        for requested in (1.5, 4.0, 8.0, 20.0):
            with self.subTest(velocity_limit=requested):
                c = self.make_controller(velocity_limit=requested)
                effective = {name: min(requested, limit) for name, limit in limits.items()}
                for name, limit in effective.items():
                    offset = c.placo_robot.get_joint_v_offset(name)
                    self.assertAlmostEqual(c.placo_robot.model.velocity_limit[offset], limit)
                c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
                c.step_control()
                for side in ("left", "right"):
                    c.xr_client.poses[f"{side}_controller"][:3] = [0.2, 0.1, 0.1]
                    c.xr_client.poses[f"{side}_controller"][3:] = np.roll(tf.quaternion_about_axis(0.7, [1, 0, 0]), -1)
                for _ in range(20):
                    c._update_robot_state()
                    previous = {name: c.placo_robot.get_joint(name) for name in limits}
                    c.step_control()
                    self.assertTrue(c._ik_valid)
                    for name, limit in effective.items():
                        speed = abs(c.placo_robot.get_joint(name) - previous[name]) / c.dt
                        self.assertLessEqual(speed, limit + 1e-6, name)

    def test_invalid_velocity_limit_is_rejected_before_connecting_xr(self):
        for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with (
                self.subTest(velocity_limit=value),
                patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient") as xr,
            ):
                with self.assertRaisesRegex(ValueError, "velocity_limit must be finite and positive"):
                    AgibotX2TeleopController(
                        str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"), velocity_limit=value
                    )
                xr.assert_not_called()

    def test_idle_and_position_actuator_tracking(self):
        c = self.make_controller()
        initial = c.mj_data.qpos.copy()
        fixed_names = ["pelvis", "torso_link", "head_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link"]
        fixed = {n: c.mj_data.xpos[c.mj_model.body(n).id].copy() for n in fixed_names}
        for _ in range(300):
            c.step_control()
        self.assertTrue(np.all(np.isfinite(c.mj_data.qpos)))
        self.assertLess(np.max(np.abs(c.mj_data.qpos - initial)), 0.025)
        np.testing.assert_array_equal(c.mj_data.ctrl, initial)
        self.assertAlmostEqual(c.mj_data.time, 3.0, places=8)
        command = initial.copy()
        for side in ("left", "right"):
            i = c.mj_model.joint(f"{side}_elbow_joint").qposadr[0]
            command[i] -= 0.2
        c.mj_data.ctrl[:] = command
        for _ in range(200):
            c.step_control()
        self.assertLess(np.max(np.abs(c.mj_data.qpos - command)), 0.025)
        self.assertLess(np.max(np.abs(c.mj_data.qvel)), 1e-3)
        for n, pos in fixed.items():
            np.testing.assert_array_equal(c.mj_data.xpos[c.mj_model.body(n).id], pos)

    def test_independent_arms_clutch_and_orientation(self):
        c = self.make_controller()
        xr = c.xr_client
        right_ctrl = c.mj_data.ctrl[7:].copy()
        xr.grips["left_grip"] = 1.0
        c.step_control()
        initial_target = c.effector_task["left_arm"].T_world_frame.copy()
        for i in range(1, 61):
            xr.poses["left_controller"][0] = i * 0.0005
            c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl[7:], right_ctrl)
        self.assertGreater(np.linalg.norm(c.mj_data.ctrl[:7] - np.array([-0.3, 0.2, 0, -1, 0, 0, 0])), 0.01)
        expected = initial_target[:3, 3] + c.R_headset_world @ np.array([0.03, 0, 0])
        np.testing.assert_allclose(c.effector_task["left_arm"].T_world_frame[:3, 3], expected, atol=1e-8)
        for _ in range(50):
            c.step_control()
        actual, _ = c._get_link_pose("left_wrist_roll_link")
        self.assertLess(np.linalg.norm(actual - expected), 0.015)
        xr.grips["left_grip"] = 0
        c.step_control()
        held = c.mj_data.ctrl.copy()
        xr.poses["left_controller"][:3] += 0.4
        for _ in range(20):
            c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        # Re-engagement targets the measured wrist at the new controller reference.
        before, _ = c._get_link_pose("left_wrist_roll_link")
        xr.grips["left_grip"] = 1
        xr.grips["right_grip"] = 1
        c.step_control()
        np.testing.assert_allclose(c.effector_task["left_arm"].T_world_frame[:3, 3], before, atol=1e-8)
        rotation_before = c.effector_task["right_arm"].T_world_frame[:3, :3].copy()
        for i in range(1, 61):
            for side in ("left", "right"):
                xr.poses[f"{side}_controller"][1] += 0.0002
                quat = tf.quaternion_about_axis(i * 0.001, [0, 1, 0])
                xr.poses[f"{side}_controller"][3:] = np.roll(quat, -1)
            c.step_control()
        self.assertFalse(np.allclose(c.mj_data.ctrl[7:], right_ctrl))
        self.assertGreater(np.linalg.norm(c.effector_task["right_arm"].T_world_frame[:3, :3] - rotation_before), 0.01)
        self.assertTrue(np.all(c.mj_data.qpos >= c.mj_model.jnt_range[:, 0] - 1e-3))
        self.assertTrue(np.all(c.mj_data.qpos <= c.mj_model.jnt_range[:, 1] + 1e-3))

    def test_world_frame_targets_ignore_head_motion(self):
        c = self.make_controller()
        xr = c.xr_client
        for side, offset in (("left", -0.2), ("right", 0.2)):
            quat = tf.quaternion_from_euler(0.1, -0.2, offset)
            xr.poses[f"{side}_controller"] = np.r_[[offset, 1.2, -0.3], np.roll(quat, -1)]
        xr.grips.update(left_grip=1.0, right_grip=1.0)
        c.step_control()
        targets = {name: task.T_world_frame.copy() for name, task in c.effector_task.items()}

        for angle in np.linspace(0, 1.0, 20):
            quat = tf.quaternion_from_euler(0.2 * angle, angle, -0.3 * angle)
            xr.poses["headset"] = np.r_[[angle, 1.6, -0.2 * angle], np.roll(quat, -1)]
            c.step_control()
            for name, target in targets.items():
                np.testing.assert_allclose(c.effector_task[name].T_world_frame, target, atol=1e-8)

        # Controller motion uses fixed tracking axes regardless of headset pose.
        rotation = tf.rotation_matrix(0.1, [0, 1, 0])[:3, :3]
        delta_quat = tf.quaternion_about_axis(0.1, [0, 1, 0])
        for side in ("left", "right"):
            pose = xr.poses[f"{side}_controller"]
            pose[:3] += [0.03, 0, 0]
            pose[3:] = np.roll(tf.quaternion_multiply(delta_quat, np.roll(pose[3:], 1)), -1)
        c.step_control()
        for name, target in targets.items():
            expected = target.copy()
            expected[:3, 3] += c.R_headset_world @ np.array([0.03, 0, 0])
            expected[:3, :3] = c.R_headset_world @ rotation @ c.R_headset_world.T @ target[:3, :3]
            np.testing.assert_allclose(c.effector_task[name].T_world_frame, expected, atol=1e-8)

    def test_unreachable_target_remains_bounded(self):
        c = self.make_controller()
        c.xr_client.grips["left_grip"] = 1
        c.step_control()
        c.xr_client.poses["left_controller"][:3] = [10, 10, 10]
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(50):
                c.step_control()
        self.assertTrue(np.all(np.isfinite(c.mj_data.qpos)))
        self.assertTrue(np.all(c.mj_data.ctrl >= c.mj_model.actuator_ctrlrange[:, 0]))
        self.assertTrue(np.all(c.mj_data.ctrl <= c.mj_model.actuator_ctrlrange[:, 1]))
        self.assertTrue(np.all(c.mj_data.qpos >= c.mj_model.jnt_range[:, 0] - 0.01))
        self.assertTrue(np.all(c.mj_data.qpos <= c.mj_model.jnt_range[:, 1] + 0.01))

    def test_failed_ik_and_invalid_tracking_hold_commands(self):
        c = self.make_controller()
        held = c.mj_data.ctrl.copy()
        c.xr_client.grips["left_grip"] = 1
        with (
            patch.object(placo.KinematicsSolver, "solve", side_effect=RuntimeError("injected failure")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        c.xr_client.poses["left_controller"][0] = np.nan
        c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        self.assertIsNone(c.ref_controller_xyz["left_arm"])

    def test_world_frame_control_does_not_require_headset_pose(self):
        c = self.make_controller()
        xr = c.xr_client
        del xr.poses["headset"]
        xr.grips.update(left_grip=1.0, right_grip=1.0)
        c.step_control()
        targets = {name: task.T_world_frame.copy() for name, task in c.effector_task.items()}
        initial_command = c.mj_data.ctrl.copy()
        for i in range(1, 41):
            for side in ("left", "right"):
                xr.poses[f"{side}_controller"][0] = i * 0.0005
            c.step_control()
            self.assertTrue(c._ik_valid)
        self.assertGreater(np.linalg.norm(c.mj_data.ctrl - initial_command), 0.01)
        for name, target in targets.items():
            expected = target[:3, 3] + c.R_headset_world @ np.array([0.02, 0, 0])
            np.testing.assert_allclose(c.effector_task[name].T_world_frame[:3, 3], expected, atol=1e-8)

    def test_stalled_stream_holds_commands_and_rebases_on_recovery(self):
        c = self.make_controller()
        xr = c.xr_client
        xr.grips.update(left_grip=1.0, right_grip=1.0)
        with patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.time.monotonic") as clock:
            clock.return_value = 100.0
            c.step_control()
            xr.poses["left_controller"][0] = 0.15
            c.step_control()
            held = c.mj_data.ctrl.copy()
            xr.advance_timestamps = False

            clock.return_value += c.XR_TRACKING_TIMEOUT_S + 0.01
            for _ in range(10):
                c.step_control()
                self.assertFalse(c._ik_valid)
                np.testing.assert_array_equal(c.mj_data.ctrl, held)
                for name in ("left_arm", "right_arm"):
                    self.assertIsNone(c.ref_ee_xyz[name])
                    self.assertIsNone(c.ref_controller_xyz[name])

            # Releases must still be processed during an outage.
            xr.grips["right_grip"] = 0.0
            c.step_control()
            self.assertFalse(c.active["right_arm"])

            # Tracking resumes at a new pose without replaying the old displacement.
            xr.poses["left_controller"][0] = 0.5
            before, _ = c._get_link_pose("left_wrist_roll_link")
            xr.advance_timestamps = True
            c.step_control()
            self.assertTrue(c._ik_valid)
            np.testing.assert_allclose(c.effector_task["left_arm"].T_world_frame[:3, 3], before, atol=1e-8)
            xr.poses["left_controller"][0] += 0.01
            c.step_control()
            expected = before + c.R_headset_world @ np.array([0.01, 0, 0])
            np.testing.assert_allclose(c.effector_task["left_arm"].T_world_frame[:3, 3], expected, atol=1e-8)

            # Fresh packets with stationary poses must not trip the watchdog.
            clock.return_value += 10.0
            c.step_control()
            self.assertTrue(c._ik_valid)

    def test_cached_startup_and_reset_timestamps_require_new_packets(self):
        c = self.make_controller()
        xr = c.xr_client
        xr.grips["left_grip"] = 1.0
        xr.advance_timestamps = False
        held = c.mj_data.ctrl.copy()
        c.step_control()
        self.assertFalse(c._ik_valid)
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        xr.advance_timestamps = True
        c.step_control()
        self.assertTrue(c._ik_valid)

        for timestamp in (0, 1):
            with self.subTest(timestamp=timestamp):
                xr.advance_timestamps = False
                xr.timestamp_ns = timestamp
                held = c.mj_data.ctrl.copy()
                c.step_control()
                self.assertFalse(c._ik_valid)
                np.testing.assert_array_equal(c.mj_data.ctrl, held)
                self.assertIsNone(c.ref_controller_xyz["left_arm"])
                c.step_control()
                self.assertFalse(c._ik_valid)
                xr.advance_timestamps = True
                before, _ = c._get_link_pose("left_wrist_roll_link")
                c.step_control()
                self.assertTrue(c._ik_valid)
                np.testing.assert_allclose(c.effector_task["left_arm"].T_world_frame[:3, 3], before, atol=1e-8)

    def test_tracking_loss_preserves_other_arm_clutch_release(self):
        for invalid_side, released_side in (("left", "right"), ("right", "left")):
            with self.subTest(invalid_side=invalid_side):
                c = self.make_controller()
                xr = c.xr_client
                xr.grips.update(left_grip=1.0, right_grip=1.0)
                c.step_control()
                held = c.mj_data.ctrl.copy()

                xr.poses[f"{invalid_side}_controller"][0] = np.nan
                xr.grips[f"{released_side}_grip"] = 0.0
                xr.poses[f"{released_side}_controller"][0] += 0.3
                c.step_control()
                name = f"{released_side}_arm"
                self.assertFalse(c.active[name])
                self.assertIsNone(c.ref_ee_xyz[name])
                self.assertIsNone(c.ref_controller_xyz[name])
                np.testing.assert_array_equal(c.mj_data.ctrl, held)

                # Re-engage while the other controller still has invalid tracking.
                before, _ = c._get_link_pose(f"{released_side}_wrist_roll_link")
                xr.grips[f"{released_side}_grip"] = 1.0
                c.step_control()
                np.testing.assert_array_equal(c.mj_data.ctrl, held)
                np.testing.assert_allclose(c.effector_task[name].T_world_frame[:3, 3], before, atol=1e-8)

                xr.poses[f"{invalid_side}_controller"][0] = 0.0
                before, _ = c._get_link_pose(f"{released_side}_wrist_roll_link")
                c.step_control()
                self.assertTrue(c._ik_valid)
                np.testing.assert_allclose(c.effector_task[name].T_world_frame[:3, 3], before, atol=1e-8)

    def test_global_tracking_hold_discards_both_arms_accumulated_motion(self):
        for invalid_side in ("left", "right"):
            with self.subTest(invalid_side=invalid_side):
                c = self.make_controller()
                xr = c.xr_client
                xr.grips.update(left_grip=1.0, right_grip=1.0)
                c.step_control()
                held = c.mj_data.ctrl.copy()
                xr.poses[f"{invalid_side}_controller"][0] = np.nan
                healthy_side = "right" if invalid_side == "left" else "left"
                for i in range(1, 21):
                    pose = xr.poses[f"{healthy_side}_controller"]
                    pose[:3] = [i * 0.01, i * 0.002, 0.0]
                    pose[3:] = np.roll(tf.quaternion_about_axis(i * 0.02, [0, 1, 0]), -1)
                    c.step_control()
                    self.assertFalse(c._ik_valid)
                    np.testing.assert_array_equal(c.mj_data.ctrl, held)

                # Include motion arriving in the same cycle as tracking recovery.
                xr.poses[f"{healthy_side}_controller"][0] += 0.1
                xr.poses[f"{invalid_side}_controller"][0] = 0.4
                measured = {}
                for side in ("left", "right"):
                    xyz, quat = c._get_link_pose(f"{side}_wrist_roll_link")
                    measured[side] = tf.quaternion_matrix(quat)
                    measured[side][:3, 3] = xyz
                c.step_control()
                self.assertTrue(c._ik_valid)
                for side, target in measured.items():
                    np.testing.assert_allclose(c.effector_task[f"{side}_arm"].T_world_frame, target, atol=1e-8)

                # Fresh controller motion still works from the recovered references.
                for side in ("left", "right"):
                    xr.poses[f"{side}_controller"][0] += 0.01
                c.step_control()
                for side, target in measured.items():
                    expected = target.copy()
                    expected[:3, 3] += c.R_headset_world @ np.array([0.01, 0, 0])
                    np.testing.assert_allclose(c.effector_task[f"{side}_arm"].T_world_frame, expected, atol=1e-8)

    def test_viewer_close_and_interrupt_close_xr(self):
        for interrupt in (False, True):
            c = self.make_controller()
            with patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.mj_viewer.launch_passive") as launch:
                viewer = launch.return_value.__enter__.return_value
                if interrupt:
                    viewer.is_running.side_effect = KeyboardInterrupt
                else:
                    viewer.is_running.return_value = False
                c.run()
            self.assertTrue(c.xr_client.closed)
            self.assertTrue(c._stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
