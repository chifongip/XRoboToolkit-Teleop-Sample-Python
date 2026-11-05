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
        self.buttons = {"Y": False}

    def get_button_state_by_name(self, name):
        return self.buttons[name]

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

    def test_default_control_frequency_is_50_hz(self):
        c = self.make_controller()
        self.assertEqual(c.control_frequency, 50.0)
        self.assertAlmostEqual(c.dt, 0.02)
        self.assertAlmostEqual(c.solver.dt, c.dt)
        self.assertEqual(c.physics_substeps, 20)
        c.step_control()
        self.assertAlmostEqual(c.mj_data.time, c.dt, places=12)

    def test_hold_y_returns_arms_with_speed_limits_and_release_holds(self):
        c = self.make_controller(return_velocity_limit=0.25)
        for joint in ("left_elbow_joint", "right_elbow_joint"):
            c.mj_data.qpos[c.mj_model.joint(joint).qposadr[0]] -= 0.1
            actuator = next(i for i, name in c.arm_actuators.items() if name == joint)
            c.mj_data.ctrl[actuator] -= 0.1
        mujoco.mj_forward(c.mj_model, c.mj_data)
        initial = c.mj_data.ctrl.copy()
        home = c.mj_model.key("home").ctrl
        c.xr_client.buttons["Y"] = True
        for step in range(80):
            before = c.mj_data.ctrl.copy()
            c.step_control()
            self.assertTrue(c._ik_valid)
            self.assertLessEqual(np.max(np.abs(c.mj_data.ctrl - before)), 0.25 * c.dt + 1e-12)
            if step == 2:
                c.xr_client.buttons["Y"] = False
                held = c.mj_data.ctrl.copy()
                c.step_control()
                np.testing.assert_array_equal(c.mj_data.ctrl, held)
                self.assertGreater(np.linalg.norm(held - home), 0.05)
                c.xr_client.buttons["Y"] = True
        self.assertLess(np.linalg.norm(c.mj_data.ctrl - home), 1e-6)
        self.assertGreater(np.linalg.norm(c.mj_data.ctrl - initial), 0.1)
        c.xr_client.buttons["Y"] = False
        held = c.mj_data.ctrl.copy()
        c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)

    def test_slow_return_reaches_home_despite_settled_servo_error(self):
        for force_avoidance in (False, True):
            with self.subTest(force_avoidance=force_avoidance):
                c = self.make_controller(arm_velocity_limit=0.1, return_velocity_limit=0.05)
                joint = c.mj_model.joint("left_elbow_joint")
                actuator = next(a for a, name in c.arm_actuators.items() if name == joint.name)
                c.mj_data.qpos[joint.qposadr[0]] -= 0.1
                c.mj_data.ctrl[actuator] -= 0.1
                mujoco.mj_forward(c.mj_model, c.mj_data)
                for _ in range(100):
                    c.step_control()
                errors = [
                    abs(c.mj_data.qpos[c.mj_model.joint(name).qposadr[0]] - c.mj_data.ctrl[a])
                    for a, name in c.arm_actuators.items()
                ]
                self.assertGreater(max(errors), (c.arm_velocity_limit + c.return_velocity_limit) * c.dt)
                c.xr_client.buttons["Y"] = True
                validate = c._validate_collision_motion
                calls = 0

                def check(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if force_avoidance and calls == 1:
                        raise RuntimeError("exercise collision-constrained fallback")
                    return validate(*args, **kwargs)

                with (
                    patch.object(c, "_validate_collision_motion", side_effect=check),
                    patch.object(c, "_solve_return_avoidance", wraps=c._solve_return_avoidance) as avoid,
                ):
                    for _ in range(150):
                        before = c.mj_data.ctrl.copy()
                        c.step_control()
                        self.assertTrue(c._ik_valid)
                        self.assertLessEqual(np.max(np.abs(c.mj_data.ctrl - before)), 0.05 * c.dt + 1e-9)
                        if np.max(np.abs(c.mj_data.ctrl - c.mj_model.key("home").ctrl)) < 1e-6:
                            break
                    else:
                        self.fail("Slow return stalled with steady servo error")
                    if force_avoidance:
                        self.assertGreater(avoid.call_count, 0)

    def test_return_to_default_holds_on_tracking_loss_and_collision(self):
        c = self.make_controller()
        c.xr_client.buttons["Y"] = True
        held = c.mj_data.ctrl.copy()
        with patch.object(c, "_xr_stream_is_fresh", return_value=False):
            c.step_control()
        self.assertFalse(c._ik_valid)
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        c.xr_client.poses["left_controller"][0] = np.nan
        c.step_control()
        self.assertFalse(c._ik_valid)
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        c.xr_client.poses["left_controller"][0] = 0.0
        with patch.object(c, "_validate_collision_motion", side_effect=RuntimeError("blocked")) as check:
            c.step_control()
        self.assertGreater(check.call_count, 0)
        self.assertFalse(c._ik_valid)
        np.testing.assert_array_equal(c.mj_data.ctrl, held)

    def test_return_release_rebases_gripped_controllers(self):
        c = self.make_controller()
        c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
        c.step_control()
        c.xr_client.buttons["Y"] = True
        c.xr_client.poses["left_controller"][0] = 1.0
        c.step_control()
        self.assertIsNone(c.ref_controller_xyz["left_arm"])
        c.xr_client.buttons["Y"] = False
        c.step_control()
        xyz, quat = c._get_link_pose("left_hand_center")
        np.testing.assert_allclose(c.effector_task["left_arm"].T_world_frame[:3, 3], xyz, atol=0.003)

    def test_control_frequency_preserves_exact_physics_periods(self):
        for frequency in (25.0, 60.0, 100.0, 2000.0):
            with self.subTest(frequency=frequency):
                c = self.make_controller(control_frequency=frequency)
                self.assertAlmostEqual(c.dt, 1 / frequency)
                self.assertAlmostEqual(c.solver.dt, c.dt)
                self.assertGreaterEqual(c.physics_substeps, 1)
                self.assertLessEqual(c.mj_model.opt.timestep, 0.001)
                self.assertAlmostEqual(c.physics_substeps * c.mj_model.opt.timestep, c.dt, places=12)
                for _ in range(5):
                    c.step_control()
                self.assertAlmostEqual(c.mj_data.time, 5 / frequency, places=12)

    def test_invalid_control_frequency_rejected_before_xr_connection(self):
        for frequency in (0, -50, np.nan, np.inf, -np.inf, 1e-320):
            with (
                self.subTest(frequency=frequency),
                patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient") as xr,
            ):
                with self.assertRaisesRegex(ValueError, "control_frequency"):
                    AgibotX2TeleopController(
                        str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"), control_frequency=frequency
                    )
                xr.assert_not_called()

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
                name = f"{side}_hand_center"
                target = robot.get_T_world_frame(name)
                body = m.body(name).id
                self.assertLess(np.linalg.norm(target[:3, 3] - data.xpos[body]), 1e-4)
                delta = target[:3, :3].T @ data.xmat[body].reshape(3, 3)
                angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
                self.assertLess(angle, 1e-3)

    def test_rigid_hand_centers_and_markers(self):
        c = self.make_controller()
        for side, expected in (
            ("left", [0.00658717, -0.00936239, -0.0826008]),
            ("right", [0.00671652, 0.00921908, -0.08261105]),
        ):
            name = f"{side}_hand_center"
            frame = c.mj_model.body(name)
            wrist = c.mj_model.body(f"{side}_wrist_roll_link")
            self.assertEqual(frame.parentid[0], wrist.id)
            np.testing.assert_allclose(frame.pos, expected, atol=1e-8)
            np.testing.assert_allclose(c.mj_data.xmat[frame.id], c.mj_data.xmat[wrist.id], atol=1e-12)
            self.assertEqual(c.manipulator_config[f"{side}_arm"]["link_name"], name)
            marker = c.target_mocap_idx[f"{side}_arm"]
            np.testing.assert_allclose(c.mj_data.mocap_pos[marker], c.mj_data.xpos[frame.id], atol=1e-4)

    def test_native_vr_axes_and_xyzw_quaternions_map_to_robot_world(self):
        c = self.make_controller()
        c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
        c.step_control()
        targets = {side: c.effector_task[f"{side}_arm"].T_world_frame.copy() for side in ("left", "right")}
        # Independent physical axis expectations, rather than deriving expected
        # results from the very conversion matrix being tested.
        expected_axes = ([0, -1, 0], [0, 0, 1], [-1, 0, 0])
        angle = np.pi / 6
        for side, initial in targets.items():
            for raw_axis, robot_axis in zip(np.eye(3), expected_axes):
                with self.subTest(side=side, raw_axis=raw_axis):
                    pose = c.xr_client.poses[f"{side}_controller"]
                    pose[:3] = 0.01 * raw_axis
                    # SDK uses xyzw; meshcat and MuJoCo use wxyz.
                    pose[3:] = np.roll(tf.quaternion_about_axis(angle, raw_axis), -1)
                    c.step_control()
                    target = c.effector_task[f"{side}_arm"].T_world_frame
                    np.testing.assert_allclose(target[:3, 3], initial[:3, 3] + 0.01 * np.array(robot_axis), atol=1e-12)
                    rotation = tf.quaternion_matrix(tf.quaternion_about_axis(angle, robot_axis))[:3, :3]
                    np.testing.assert_allclose(target[:3, :3], rotation @ initial[:3, :3], atol=1e-12)

    def test_ik_respects_user_and_urdf_joint_velocity_limits(self):
        limits = {
            joint.get("name"): float(joint.find("limit").get("velocity"))
            for joint in ET.parse(ASSETS / "x2_fixed.urdf").getroot().findall("joint")
            if joint.get("type") != "fixed"
        }
        for requested in (1.5, 4.0, 8.0, 20.0):
            with self.subTest(arm_velocity_limit=requested):
                c = self.make_controller(arm_velocity_limit=requested)
                effective = {name: min(requested, limit) for name, limit in limits.items()}
                for name, limit in effective.items():
                    offset = c.placo_robot.get_joint_v_offset(name)
                    self.assertAlmostEqual(c.placo_robot.model.velocity_limit[offset], limit)
                c.xr_client.grips.update(left_grip=1.0, right_grip=1.0)
                c.step_control()
                # Reach forward/up from the new home so collision rejection
                # does not obscure the velocity-limit assertions.
                for side in ("left", "right"):
                    c.xr_client.poses[f"{side}_controller"][:3] = [0.02, 0.1, -0.1]
                    c.xr_client.poses[f"{side}_controller"][3:] = np.roll(tf.quaternion_about_axis(0.7, [1, 0, 0]), -1)
                for _ in range(20):
                    c._update_robot_state()
                    previous = {name: c.placo_robot.get_joint(name) for name in limits}
                    c.step_control()
                    self.assertTrue(c._ik_valid)
                    for name, limit in effective.items():
                        speed = abs(c.placo_robot.get_joint(name) - previous[name]) / c.dt
                        self.assertLessEqual(speed, limit + 1e-6, name)

    def test_invalid_velocity_caps_are_rejected_before_connecting_xr(self):
        for name in ("arm_velocity_limit", "hand_velocity_limit", "return_velocity_limit"):
            for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
                with (
                    self.subTest(parameter=name, value=value),
                    patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient") as xr,
                ):
                    with self.assertRaisesRegex(ValueError, f"{name} must be finite and positive"):
                        AgibotX2TeleopController(
                            str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"), **{name: value}
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
        self.assertAlmostEqual(c.mj_data.time, 300 * c.dt, places=8)
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
        actual, _ = c._get_link_pose("left_hand_center")
        self.assertLess(np.linalg.norm(actual - expected), 0.015)
        xr.grips["left_grip"] = 0
        c.step_control()
        held = c.mj_data.ctrl.copy()
        xr.poses["left_controller"][:3] += 0.4
        for _ in range(20):
            c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        # Re-engagement targets the measured hand center at the new controller reference.
        before, _ = c._get_link_pose("left_hand_center")
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
            before, _ = c._get_link_pose("left_hand_center")
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
                before, _ = c._get_link_pose("left_hand_center")
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
                before, _ = c._get_link_pose(f"{released_side}_hand_center")
                xr.grips[f"{released_side}_grip"] = 1.0
                c.step_control()
                np.testing.assert_array_equal(c.mj_data.ctrl, held)
                np.testing.assert_allclose(c.effector_task[name].T_world_frame[:3, 3], before, atol=1e-8)

                xr.poses[f"{invalid_side}_controller"][0] = 0.0
                before, _ = c._get_link_pose(f"{released_side}_hand_center")
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
                    xyz, quat = c._get_link_pose(f"{side}_hand_center")
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
