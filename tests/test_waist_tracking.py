"""Waist invariance, freshness, packet validation and real X2 integration."""

import contextlib
import copy
import io
import json
import os
import runpy
import time
import unittest
from dataclasses import replace
from itertools import chain, count, repeat
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from meshcat import transformations as tf

import test_agibot_x2_omnihand as omni_fixture
import test_agibot_x2_simulation as simulation_fixture
from test_robojudo_bridge import feedback_state
from test_robojudo_profiles import MemoryContext
from test_robojudo_x2 import fixture
from xrobotoolkit_teleop.common.waist_tracking import WaistSample, WaistTracking
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD, relative_target, xr_pose_to_matrix


def pose(transform):
    return np.r_[transform[:3, 3], np.roll(tf.quaternion_from_matrix(transform), -1)]


def sample(stamp, waist=None, controllers=None, hip_positions=None):
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return WaistSample(
        identity if waist is None else waist,
        controllers or {s: identity.copy() for s in ("left", "right")},
        0,
        stamp,
        time.monotonic(),
        [[-0.1, 0, 0], [0.1, 0, 0]] if hip_positions is None else hip_positions,
        stamp,
    )


class WaistTests(unittest.TestCase):
    def test_common_motion_and_separate_position_rotation_deltas(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            waist = tf.random_rotation_matrix(rng.random(3))
            waist[:3, 3] = rng.normal(size=3)
            controller = tf.random_rotation_matrix(rng.random(3))
            controller[:3, 3] = rng.normal(size=3)
            motion = tf.random_rotation_matrix(rng.random(3))
            motion[:3, 3] = rng.normal(size=3)
            relative = np.linalg.inv(waist) @ controller
            transformed = np.linalg.inv(motion @ waist) @ (motion @ controller)
            np.testing.assert_allclose(relative, transformed, atol=1e-12)
            hand = np.eye(4)
            hand[:3, 3] = [1, 2, 3]
            np.testing.assert_allclose(
                relative_target(transformed, relative, hand, R_HEADSET_TO_WORLD, 2), hand, atol=1e-12
            )
        reference = np.eye(4)
        changed = tf.rotation_matrix(0.3, [0, 1, 0])
        changed[:3, 3] = [0.1, 0.2, 0.3]
        target = relative_target(changed, reference, hand, R_HEADSET_TO_WORLD, 2)
        np.testing.assert_allclose(target[:3, 3], hand[:3, 3] + 2 * R_HEADSET_TO_WORLD @ changed[:3, 3])
        np.testing.assert_allclose(target[:3, :3], R_HEADSET_TO_WORLD @ changed[:3, :3] @ R_HEADSET_TO_WORLD.T)

    def test_engagement_axes_do_not_depend_on_pelvis_local_yaw_convention(self):
        targets = []
        for pelvis in (np.eye(4), tf.rotation_matrix(0.7, [0, 1, 0])):
            tracking = WaistTracking()
            controllers = {side: [0, 0, 0, 0, 0, 0, 1] for side in ("left", "right")}
            tracking.update(sample(1, pose(pelvis), controllers))
            relative = tracking.update(sample(2, pose(pelvis), controllers))
            hand = np.eye(4)
            tracking.target("left", relative["left"], hand, 2.0)
            moved = tf.rotation_matrix(0.2, [0, 1, 0])
            moved[:3, 3] = [0.1, 0.2, 0.3]
            relative = tracking.update(sample(3, pose(pelvis), {s: pose(moved) for s in controllers}))
            targets.append(tracking.target("left", relative["left"], hand, 2.0))
            # Move pelvis and controllers together; captured axes must stay fixed.
            common = tf.rotation_matrix(0.2, [0, 1, 0])
            common[:3, 3] = [0.01, 0.02, 0.03]
            relative = tracking.update(sample(4, pose(common @ pelvis), {s: pose(common @ moved) for s in controllers}))
            np.testing.assert_allclose(tracking.target("left", relative["left"], hand, 2.0), targets[-1], atol=1e-12)
        np.testing.assert_allclose(targets[0], targets[1], atol=1e-12)
        np.testing.assert_allclose(targets[0][:3, 3], 2 * R_HEADSET_TO_WORLD @ [0.1, 0.2, 0.3])

    def test_regripping_after_turn_preserves_pelvis_relative_control_directions(self):
        tracking = WaistTracking()
        initial_pelvis = tf.rotation_matrix(0.2, [1, 0, 0])
        hand = np.eye(4)
        handed_motion = tf.rotation_matrix(0.15, [1, 2, 3])
        handed_motion[:3, 3] = [0.03, 0.02, -0.05]

        def update(stamp, pelvis, controller):
            return tracking.update(sample(stamp, pose(pelvis), dict.fromkeys(("left", "right"), pose(controller))))

        update(1, initial_pelvis, np.eye(4))
        relative = update(2, initial_pelvis, np.eye(4))
        tracking.target("left", relative["left"], hand, 1)
        relative = update(3, initial_pelvis, handed_motion)
        expected = tracking.target("left", relative["left"], hand, 1)

        tracking.release("left")
        for stamp, angle in enumerate((np.pi / 6, np.pi / 3, np.pi / 2), start=4):
            turn = tf.rotation_matrix(angle, [0, 1, 0])
            turn[:3, 3] = [angle / 10, 0, -angle / 10]
            relative = update(stamp, turn @ initial_pelvis, turn)
            self.assertIsNotNone(relative, tracking.reason)

        new_hand = hand.copy()
        new_hand[:3, 3] = [0.3, -0.1, 0.2]
        for arm in ("left", "right"):
            np.testing.assert_allclose(tracking.target(arm, relative[arm], new_hand, 1), new_hand, atol=1e-12)
        relative = update(7, turn @ initial_pelvis, turn @ handed_motion)
        expected[:3, 3] += new_hand[:3, 3]
        for arm in ("left", "right"):
            np.testing.assert_allclose(tracking.target(arm, relative[arm], new_hand, 1), expected, atol=1e-12)

    def test_first_grip_directions_survive_heading_lean_and_shared_motion(self):
        sideways = tf.rotation_matrix(np.pi / 2, [0, 1, 0])
        tilted = sideways @ tf.rotation_matrix(0.45, [1, 0, 2])
        hips = np.array([[-0.1, 0, 0], [0.1, 0, 0]])
        hand = np.eye(4)

        for initial_pelvis in (np.eye(4), sideways, tilted):
            with self.subTest(pelvis=initial_pelvis):
                tracking = WaistTracking()

                def update(stamp, pelvis, reach):
                    world_hips = hips @ pelvis[:3, :3].T + pelvis[:3, 3]
                    world_hands = {
                        side: pose(pelvis @ tf.translation_matrix(np.array(offset) + reach))
                        for side, offset in (("left", [-0.25, 0.1, -0.3]), ("right", [0.25, 0.1, -0.3]))
                    }
                    return tracking.update(sample(stamp, pose(pelvis), world_hands, world_hips))

                origin = np.zeros(3)
                update(1, initial_pelvis, origin)
                relative = update(2, initial_pelvis, origin)
                for arm in relative:
                    np.testing.assert_allclose(tracking.target(arm, relative[arm], hand, 1), hand, atol=1e-12)

                shared = tf.rotation_matrix(0.3, [2, 1, 3])
                shared[:3, 3] = [0.07, 0.12, 0.04]
                moved_pelvis = shared @ initial_pelvis
                relative = update(3, moved_pelvis, origin)
                for arm in relative:
                    np.testing.assert_allclose(tracking.target(arm, relative[arm], None, 1), hand, atol=1e-12)

                for stamp, reach, expected in (
                    (4, [0.0, 0.0, -0.05], [0.05, 0.0, 0.0]),
                    (5, [-0.05, 0.0, 0.0], [0.0, 0.05, 0.0]),
                ):
                    relative = update(stamp, moved_pelvis, reach)
                    for arm in relative:
                        target = tracking.target(arm, relative[arm], None, 1)
                        np.testing.assert_allclose(target[:3, 3], expected, atol=1e-12)

    def test_missing_lateral_hip_direction_holds_before_first_grip(self):
        tracking = WaistTracking()
        tracking.update(sample(1))
        relative = tracking.update(sample(2, hip_positions=[[0, 1, 0], [0, 0, 0]]))
        self.assertIsNone(relative)
        self.assertIn("lateral pelvis axis", tracking.reason)
        relative = tracking.update(sample(3))
        self.assertIsNotNone(relative)
        tracking.target("left", relative["left"], np.eye(4), 1)
        self.assertIsNotNone(tracking.update(sample(4, hip_positions=[])))

    def test_freshness_rollback_discontinuity_and_recovery(self):
        tracking = WaistTracking()
        self.assertIsNone(tracking.update(sample(1)))
        self.assertIsNotNone(tracking.update(sample(2)))
        tracking.target("left", np.eye(4), np.eye(4), 1.0)
        frozen = sample(2)
        with patch("xrobotoolkit_teleop.common.waist_tracking.time.monotonic", return_value=frozen.observed_at + 0.3):
            self.assertIsNone(tracking.update(frozen))
        self.assertFalse(tracking.has_reference("left"))
        self.assertIsNotNone(tracking.update(sample(3)))
        self.assertIsNone(tracking.update(sample(1)))
        self.assertIsNotNone(tracking.update(sample(2)))
        jumped = sample(3, [2, 0, 0, 0, 0, 0, 1])
        self.assertIsNone(tracking.update(jumped))
        self.assertIn("discontinuity", tracking.reason)
        self.assertIsNotNone(tracking.update(sample(4, jumped.pelvis_pose)))

    def test_fault_recovery_requires_new_packet_timestamp(self):
        tracking = WaistTracking()
        self.assertIsNone(tracking.update(sample(0)))
        self.assertIn("invalid XR packet timestamp", tracking.reason)
        tracking.update(sample(1))
        tracking.update(sample(2))
        self.assertIsNone(tracking.update(None))
        self.assertIsNone(tracking.update(sample(2)))
        self.assertIsNotNone(tracking.update(sample(3)))
        base = sample(4)
        # A body timestamp change by itself is insufficient after a fault.
        tracking.update(None)
        self.assertIsNone(tracking.update(replace(base, xr_timestamp_ns=3)))
        self.assertIsNotNone(tracking.update(sample(4)))

    def test_fresh_packet_after_polling_gap_requires_rebase(self):
        tracking = WaistTracking()
        with patch("time.monotonic", return_value=100.0) as clock:
            tracking.update(sample(1))
            self.assertIsNotNone(tracking.update(sample(2)))
            tracking.target("left", np.eye(4), np.eye(4), 1.0)

            clock.return_value += tracking.timeout_s + 0.01
            self.assertIsNone(tracking.update(sample(3)))
            self.assertIn("sample gap", tracking.reason)
            self.assertFalse(tracking.has_reference("left"))
            self.assertIsNone(tracking.update(sample(3)))
            self.assertIsNotNone(tracking.update(sample(4)))

    def test_invalid_missing_and_quaternion_sign(self):
        for bad in ([0] * 7, [float("nan")] * 7, [0] * 6):
            with self.assertRaises(ValueError):
                xr_pose_to_matrix(bad)
        np.testing.assert_allclose(xr_pose_to_matrix([0, 0, 0, 0, 0, 0, -1]), np.eye(4))
        tracking = WaistTracking()
        tracking.update(sample(1))
        self.assertIsNone(tracking.update(replace(sample(2), controllers={})))
        self.assertIn("left", tracking.reason)
        self.assertIsNone(tracking.update(None))


def make_waist_retargeter(test, robot="x2"):
    retargeter = ExternalStateRetargeter(
        robot=robot, ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"), tracking_reference="waist"
    )
    test.addCleanup(retargeter.close)
    return retargeter


class WaistRetargeterTests(unittest.TestCase):
    def setUp(self):
        self.r = make_waist_retargeter(self)
        self.measured, self.home = fixture(self.r)
        self.previous = self.home.copy()

    def test_waist_path_ignores_head_and_holds_on_loss(self):
        def step(stamp, grip=0.0, data=True, body_stamp=None):
            waist = sample(stamp) if data else None
            if body_stamp is not None:
                waist = replace(waist, body_head_timestamp_ns=body_stamp)
            return self.r.step(
                self.measured,
                self.previous,
                self.home,
                self.r.limits,
                intent="TRACK",
                frame=XRFrame(stamp, {}, {"left": grip, "right": 0.0}, waist=waist),
            )

        step(1)
        step(2)
        step(3)
        self.measured["left_elbow_joint"] -= 0.03
        self.assertIsNone(step(4, 1.0, body_stamp=3))
        self.assertIn("fresh body frame", self.r.hold_reason)
        self.assertFalse(self.r.waist_tracking.has_reference("left"))
        engagement = step(5, 1.0)
        self.assertIsNotNone(engagement, self.r.hold_reason)
        np.testing.assert_allclose(list(engagement.values()), list(self.previous.values()), atol=1e-7, rtol=0)
        before = np.array(self.r.target_poses["left"])
        robot = self.r.robot
        torso = robot.get_T_world_frame(self.r.profile.reference_frame)
        hand_frame = self.r.profile.groups["left"].frame
        measured_hand = np.linalg.inv(torso) @ robot.get_T_world_frame(hand_frame)
        measured_q = robot.state.q.copy()
        try:
            robot.set_joint("left_elbow_joint", self.previous["left_elbow_joint"])
            robot.update_kinematics()
            held_hand = np.linalg.inv(torso) @ robot.get_T_world_frame(hand_frame)
        finally:
            robot.state.q = measured_q
            robot.update_kinematics()
        self.assertGreater(np.linalg.norm(measured_hand - held_hand), 0.01)
        np.testing.assert_allclose(before, measured_hand, atol=1e-12)
        self.assertIsNotNone(step(6, 1.0))
        np.testing.assert_allclose(self.r.target_poses["left"], before)
        self.assertIsNone(step(7, 1.0, False))
        self.assertFalse(self.r.waist_tracking.has_reference("left"))
        self.assertIsNotNone(step(8, 1.0))
        np.testing.assert_allclose(self.r.target_poses["left"], before)

    def test_upstream_timeout_requires_new_waist_data(self):
        def step(xr_stamp, waist_stamp, grip=0.0):
            return self.r.step(
                self.measured,
                self.previous,
                self.home,
                self.r.limits,
                intent="TRACK",
                frame=XRFrame(xr_stamp, {}, {"left": grip, "right": 0.0}, waist=sample(waist_stamp)),
            )

        for stamp in (1, 2, 3):
            step(stamp, stamp)
        self.assertIsNotNone(step(4, 4, 1.0))
        self.assertIsNone(step(0, 5, 1.0))
        self.assertFalse(self.r.waist_tracking.valid)
        self.assertIsNone(step(6, 4, 1.0))
        self.assertIsNotNone(step(7, 7, 1.0))


class WaistSimulationTests(unittest.TestCase):
    make_controller = simulation_fixture.X2SimulationTests.make_controller

    def test_omnihand_simulation_fault_and_home_hold_fingers(self):
        c = omni_fixture.X2OmnihandProTests.make_controller(
            self,
            tracking_reference="waist",
            ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"),
        )
        stamp = [0]

        def read():
            stamp[0] += 1
            return sample(stamp[0])

        c.xr_client.get_waist_sample = read
        c.xr_client.grips["left_trigger"] = 1.0
        c.step_control()
        c.step_control()
        c.step_control()
        fingers = list(c.hand_command_offsets)
        self.assertTrue(np.any(c.mj_data.ctrl[fingers] != 0))
        previous = c.mj_data.ctrl[fingers].copy()
        with patch.object(c.xr_client, "get_waist_sample", side_effect=ValueError("lost")):
            c.step_control()
            np.testing.assert_array_equal(c.mj_data.ctrl[fingers], previous)
            c.xr_client.buttons["Y"] = True
            c.xr_client.poses["left_controller"] = [0] * 7
            c.step_control()
            self.assertTrue(c._ik_valid, c.status)
            np.testing.assert_array_equal(c.mj_data.ctrl[fingers], previous)

    def test_simulator_recovery_waits_for_packet_progress(self):
        c = self.make_controller(tracking_reference="waist")
        c._need_grip_release = False
        c.waist_tracking.update(sample(1))
        c.waist_tracking.update(sample(2))
        c.waist_tracking.invalidate("lost")
        base = sample(3)
        with patch.object(c.xr_client, "get_waist_sample", create=True, side_effect=[sample(2), base]):
            c._update_waist_targets(True, True)
            self.assertFalse(c.waist_tracking.valid)
            c._update_waist_targets(True, True)
            self.assertTrue(c.waist_tracking.valid)

    def test_waist_mode_loss_and_recovery(self):
        c = self.make_controller(
            tracking_reference="waist",
            ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"),
        )
        stamp = [0]
        pelvis = np.eye(4)
        controllers = {side: np.eye(4) for side in ("left", "right")}

        def read():
            stamp[0] += 10_000_000
            return sample(stamp[0], pose(pelvis), {side: pose(value) for side, value in controllers.items()})

        c.xr_client.get_waist_sample = read
        c.xr_client.poses.pop("headset")
        c.step_control()
        c.step_control()
        c.xr_client.grips["left_grip"] = 1.0
        held = c.mj_data.ctrl.copy()
        with patch.object(c.backend, "propose", side_effect=AssertionError("engagement must hold")):
            c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        self.assertEqual(c.status, "tracking")
        c.step_control()
        self.assertTrue(c._ik_valid, c.status)
        left_task = c.effector_task["left_arm"]
        initial_target = left_task.T_world_frame.copy()
        controllers["left"][2, 3] = -0.01
        c.step_control()
        self.assertTrue(c._ik_valid, c.status)
        moved_target = left_task.T_world_frame.copy()
        np.testing.assert_allclose(moved_target[:3, 3] - initial_target[:3, 3], [0.01, 0, 0], atol=1e-9)
        common = tf.rotation_matrix(0.2, [0, 1, 0])
        common[:3, 3] = [0.01, 0.02, 0.03]
        pelvis = common @ pelvis
        controllers = {side: common @ value for side, value in controllers.items()}
        c.step_control()
        np.testing.assert_allclose(left_task.T_world_frame, moved_target, atol=1e-9)

        c.xr_client.grips["left_grip"] = 0.0
        turn = tf.rotation_matrix(0.6, [0, 1, 0])
        turn[:3, 3] = [0.05, 0, -0.05]
        pelvis = turn @ pelvis
        controllers = {side: turn @ value for side, value in controllers.items()}
        c.step_control()
        c.xr_client.grips["left_grip"] = 1.0
        repeated_body_stamp = stamp[0]
        held = c.mj_data.ctrl.copy()

        def repeated_body():
            return replace(read(), body_head_timestamp_ns=repeated_body_stamp)

        with patch.object(c.xr_client, "get_waist_sample", side_effect=repeated_body):
            c.step_control()
        self.assertFalse(c.waist_tracking.has_reference("left_arm"))
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        elbow = next(i for i, name in c.arm_actuators.items() if name == "left_elbow_joint")
        c.mj_data.ctrl[elbow] -= 0.03
        held = c.mj_data.ctrl.copy()
        c._update_robot_state()
        robot = c.robot
        measured = robot.state.q.copy()
        hand_frame = c.manipulator_config["left_arm"]["link_name"]
        measured_hand = robot.get_T_world_frame(hand_frame).copy()
        try:
            for i, joint in c.arm_actuators.items():
                robot.set_joint(joint, held[i])
            robot.update_kinematics()
            held_hand = robot.get_T_world_frame(hand_frame).copy()
        finally:
            robot.state.q = measured
            robot.update_kinematics()
        self.assertGreater(np.linalg.norm(measured_hand - held_hand), 0.01)
        c.step_control()
        np.testing.assert_array_equal(c.mj_data.ctrl, held)
        regrip_target = left_task.T_world_frame.copy()
        np.testing.assert_allclose(regrip_target, measured_hand, atol=1e-12)
        controllers["left"][:3, 3] += pelvis[:3, :3] @ [0, 0, -0.01]
        c.step_control()
        self.assertTrue(c._ik_valid, c.status)
        np.testing.assert_allclose(left_task.T_world_frame[:3, 3] - regrip_target[:3, 3], [0.01, 0, 0], atol=1e-9)

        before = c.mj_data.ctrl.copy()
        c.xr_client.advance_timestamps = False
        with patch("time.monotonic", return_value=c._last_xr_update_at + c.XR_TRACKING_TIMEOUT_S + 0.01):
            c.step_control()
        self.assertFalse(c._ik_valid)
        self.assertFalse(c._stream_fresh)
        np.testing.assert_array_equal(before, c.mj_data.ctrl)
        c.xr_client.advance_timestamps = True
        with patch.object(c.backend, "propose", side_effect=AssertionError("recovery must hold")):
            c.step_control()
        np.testing.assert_array_equal(before, c.mj_data.ctrl)
        c.step_control()
        self.assertTrue(c._ik_valid, c.status)


class WaistAdapterTests(unittest.TestCase):
    def test_public_sdk_getters_are_bracketed_and_partial_arrays_rejected(self):
        sdk = Mock()
        sdk.is_body_data_available.return_value = True
        sdk.get_body_timestamp_ns.return_value = 10
        sdk.get_body_joints_timestamp.return_value = [22] + [0] * 23
        sdk.get_body_joints_timestamp.return_value[15] = 45
        sdk.get_time_stamp_ns.return_value = 11
        body = np.tile([0.0, 0, 0, 0, 0, 0, 1], (24, 1))
        body[1:3, 0] = [-0.1, 0.1]
        sdk.get_body_joints_pose.return_value = body
        sdk.get_left_controller_pose.return_value = [0, 0, 0, 0, 0, 0, 1]
        sdk.get_right_controller_pose.return_value = [0, 0, 0, 0, 0, 0, 1]
        with patch("xrobotoolkit_teleop.common.xr_client.xrt", sdk):
            client = XrClient.__new__(XrClient)
            sample_read = client.get_waist_sample()
            self.assertEqual(sample_read.body_timestamp_ns, 10)
            self.assertEqual(sample_read.body_head_timestamp_ns, 45)
            np.testing.assert_allclose(sample_read.hip_positions, [[-0.1, 0, 0], [0.1, 0, 0]])
            self.assertEqual(client.get_pelvis_timestamp_ns(), 22)
            sdk.get_body_timestamp_ns.side_effect = RuntimeError("unavailable")
            self.assertIsNone(client.get_waist_sample().body_timestamp_ns)
            sdk.get_time_stamp_ns.side_effect = [11, 12, 12, 12]
            self.assertEqual(client.get_waist_sample().xr_timestamp_ns, 12)
            sdk.get_body_joints_pose.return_value = np.zeros((1, 7))
            sdk.get_time_stamp_ns.side_effect = None
            with self.assertRaisesRegex(ValueError, "body pose array"):
                client.get_waist_sample()
            sdk.get_time_stamp_ns.side_effect = [1, 2, 3, 4, 5, 6]
            with self.assertRaisesRegex(ValueError, "changed during"):
                client.get_waist_sample()

    def test_packet_timestamp_duplicates_are_valid_until_timeout(self):
        tracking = WaistTracking()
        with patch("time.monotonic", return_value=100.0) as clock:
            tracking.update(sample(1))
            self.assertIsNotNone(tracking.update(sample(2)))
            clock.return_value += 0.2
            self.assertIsNotNone(tracking.update(sample(2)))
            clock.return_value += 0.1
            self.assertIsNone(tracking.update(sample(2)))
        self.assertIn("stopped advancing", tracking.reason)

    def test_frozen_body_holds_with_live_packets_and_controllers(self):
        tracking = WaistTracking()
        with patch("time.monotonic", return_value=100.0) as clock:
            tracking.update(sample(1))
            relative = tracking.update(sample(2))
            tracking.target("left", relative["left"], np.eye(4), 1.0)
            clock.return_value += 0.1
            moving = [0.12, 0, 0, 0, 0, 0, 1]
            stale = replace(sample(3, controllers=dict.fromkeys(("left", "right"), moving)), body_head_timestamp_ns=2)
            repeated = tracking.update(stale)
            np.testing.assert_allclose(tracking.target("left", repeated["left"], None, 1.0), np.eye(4))
            clock.return_value += 0.16
            stale = replace(stale, xr_timestamp_ns=4, observed_at=clock.return_value)
            self.assertIsNone(tracking.update(stale))
            self.assertEqual(tracking.reason, "body frame timestamp stopped advancing")
            self.assertFalse(tracking.has_reference("left"))
            self.assertIsNone(tracking.update(replace(stale, xr_timestamp_ns=5)))
            self.assertIsNotNone(tracking.update(sample(6)))

    def test_first_grip_requires_body_frame_progress(self):
        tracking = WaistTracking()
        self.assertIsNone(tracking.update(replace(sample(1), body_head_timestamp_ns=None)))
        self.assertIn("body frame timestamp unavailable", tracking.reason)
        self.assertIsNone(tracking.update(sample(2)))
        self.assertIsNone(tracking.update(replace(sample(3), body_head_timestamp_ns=2)))
        self.assertIn("body frame timestamp progression", tracking.reason)
        self.assertIsNotNone(tracking.update(sample(4)))

    def test_waist_only_motion_and_independent_clutches(self):
        tracking = WaistTracking()
        tracking.update(sample(1))
        relative = tracking.update(sample(2))
        hand = np.eye(4)
        tracking.target("left", relative["left"], hand, 1)
        relative = tracking.update(sample(3, [0.1, 0, 0, 0, 0, 0, 1]))
        target = tracking.target("left", relative["left"], hand, 1)
        np.testing.assert_allclose(target[:3, 3], -R_HEADSET_TO_WORLD @ [0.1, 0, 0])
        np.testing.assert_allclose(tracking.target("right", relative["right"], hand, 1), hand)
        self.assertTrue(tracking.release("right"))
        self.assertTrue(tracking.has_reference("left"))


class WaistOmnihandTests(unittest.TestCase):
    def setUp(self):
        self.r = make_waist_retargeter(self, "x2_omnihand")
        self.measured, self.home = fixture(self.r)
        self.home.update(dict.fromkeys(self.r.hand_names, 0.0))
        self.previous = self.home.copy()
        self.limits = self.r.limits | self.r.profile.hands.limits

    def test_waist_loss_holds_fingers_even_with_live_triggers(self):
        def step(stamp, data=True, intent="TRACK"):
            frame = XRFrame(
                stamp,
                {},
                {"left": 0.0, "right": 0.0},
                triggers={"left": 1.0, "right": 1.0},
                waist=sample(stamp) if data else None,
            )
            return self.r.step(self.measured, self.previous, self.home, self.limits, intent=intent, frame=frame)

        for stamp in (1, 2, 3):
            step(stamp)
        proposal = step(4)
        self.assertIsNotNone(proposal, self.r.hold_reason)
        self.assertTrue(any(proposal[n] != self.previous[n] for n in self.r.hand_names))
        self.assertIsNone(step(5, False))
        self.assertIsNotNone(step(6, False, "HOME"))


class WaistBridgeTests(unittest.TestCase):
    def test_xr_home_samples_only_button_and_timestamp_in_either_mode(self):
        for mode in ("fixed-heading", "waist"):
            with self.subTest(mode=mode):
                xr = Mock()
                xr.get_button_state_by_name.return_value = True
                xr.get_timestamp_ns.return_value = 100
                bridge = RoboJuDoBridge(context=MemoryContext(), xr_client=xr, tracking_reference=mode)
                self.addCleanup(bridge.close)
                frame = bridge._xr_frame()
                self.assertTrue(frame.return_to_default)
                self.assertEqual(frame.timestamp_ns, 100)
                xr.get_pose_by_name.assert_not_called()
                xr.get_waist_sample.assert_not_called()
                xr.get_key_value_by_name.assert_not_called()

    def test_bridge_frame_uses_existing_sdk_adapter_without_headset(self):
        xr = Mock()
        xr.get_waist_sample.return_value = sample(100)
        xr.get_timestamp_ns.return_value = 101
        xr.get_key_value_by_name.return_value = 0.0
        xr.get_button_state_by_name.return_value = False
        bridge = RoboJuDoBridge(context=MemoryContext(), xr_client=xr, tracking_reference="waist")
        self.addCleanup(bridge.close)
        frame = bridge._xr_frame()
        self.assertIsNotNone(frame.waist)
        self.assertEqual(frame.timestamp_ns, 100)
        self.assertEqual(frame.poses, {})
        xr.get_pose_by_name.assert_not_called()
        xr.get_timestamp_ns.assert_not_called()
        xr.get_waist_sample.side_effect = ValueError("body unavailable")
        frame = bridge._xr_frame()
        self.assertIsNone(frame.waist)
        self.assertEqual(frame.timestamp_ns, 101)
        self.assertIn("body unavailable", frame.waist_error)

    def test_loss_stops_publication_and_recovery_rebases(self):
        bridge = RoboJuDoBridge(context=MemoryContext(), tracking_reference="waist")
        self.addCleanup(bridge.close)
        state = feedback_state(bridge)
        with patch("time.monotonic", return_value=100.0) as clock:

            def tick(i, grip=0.0, valid=True):
                clock.return_value = 100.0 + i * 0.02
                state["state_id"] += 1
                bridge.feedback.messages.append(copy.deepcopy(state))
                frame = XRFrame(i + 1, {}, {"left": grip, "right": 0.0}, waist=sample(i + 1) if valid else None)
                with patch.object(bridge, "_xr_frame", return_value=frame):
                    return bridge.poll()

            for i in range(4):
                tick(i)
            self.assertTrue(tick(4, 1.0), bridge.hold_reason)
            count = len(bridge.commands.frames)
            self.assertFalse(tick(5, 1.0, False))
            self.assertEqual(len(bridge.commands.frames), count)
            self.assertTrue(tick(6, 1.0), bridge.hold_reason)
            self.assertEqual(bridge.commands.frames[-1]["schema_version"], 3)


class WaistDiagnosticTests(unittest.TestCase):
    def test_recording_tracks_packet_progress_with_zero_body_timestamps(self):
        script = Path(__file__).resolve().parents[1] / "scripts/misc/diagnose_x2_waist.py"
        main = runpy.run_path(str(script))["main"]
        xr = Mock()
        packets = count(1)

        def read():
            stamp = next(packets)
            return sample(stamp, [float("nan"), 0, 0, 0, 0, 0, 1] if stamp == 1 else None)

        xr.get_waist_sample.side_effect = read
        xr.get_pelvis_timestamp_ns.side_effect = chain(
            [0, 0, 0, RuntimeError("joint timestamps unavailable")], repeat(0)
        )
        output = io.StringIO()
        with (
            patch.dict(main.__globals__, XrClient=lambda: xr),
            patch("sys.argv", [str(script), "--duration", ".24"]),
            patch("time.monotonic", side_effect=[i * 0.01 for i in range(100)]),
            patch("time.sleep"),
            contextlib.redirect_stdout(output),
        ):
            main()
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(records)
        self.assertNotIn("NaN", output.getvalue())
        self.assertIsNone(records[0]["sample"]["pelvis_pose"][0])
        self.assertIn("observed_update_hz", records[0])
        self.assertIn("xr_timestamp_ns", records[0]["sample"])
        self.assertEqual([r["pelvis_timestamp_ns"] for r in records[:4]], [0, 0, 0, None])
        self.assertEqual(records[3]["pelvis_timestamp_error"], "joint timestamps unavailable")
        self.assertEqual(records[2]["pelvis_observed_update_hz"], 0)
        self.assertTrue(all(r["sample"]["body_timestamp_ns"] == 0 for r in records))
        self.assertIsNone(records[0]["relative"])
        self.assertTrue(all(r["hold_reason"] is None and r["relative"] is not None for r in records[1:]))
        xr.close.assert_called_once()
