"""External-state retargeting and bridge tests without XR or robot hardware."""

import copy
import json
import os
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import zmq
from meshcat import transformations as tf
from xrobotoolkit_teleop.common.x2_arm_retargeter import X2ArmRetargeter, X2XRFrame
from xrobotoolkit_teleop.common.x2_defaults import DEFAULTS, X2TeleopSettings
from xrobotoolkit_teleop.integrations.robojudo_x2 import RoboJuDoX2Bridge
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD


def fixture(retargeter):
    home = dict(
        zip(
            retargeter.arm_names,
            np.array(
                [0.35, 0.1, 0.0, -0.87, 0.0, 0.0, 0.0, 0.35, -0.1, 0.0, -0.87, 0.0, 0.0, 0.0], dtype=np.float32
            ).tolist(),
            strict=True,
        )
    )
    measured = dict.fromkeys(retargeter.joint_names, 0.0)
    measured.update(home)
    for side in ("left", "right"):
        measured[f"{side}_knee_joint"] = 0.3
        measured[f"{side}_hip_pitch_joint"] = -0.1
        measured[f"{side}_ankle_pitch_joint"] = -0.2
    return measured, home


class RetargeterTests(unittest.TestCase):
    def setUp(self):
        self.r = X2ArmRetargeter(ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"))
        self.addCleanup(self.r.close)
        self.measured, self.home = fixture(self.r)
        self.previous = self.home.copy()
        self.poses = {
            name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for name in ("headset", "left_controller", "right_controller")
        }
        self.timestamp = 0

    def step(self, grip=0.0, intent="TRACK", advance=True):
        self.timestamp += int(advance)
        frame = X2XRFrame(self.timestamp, self.poses, {"left": grip, "right": 0.0})
        return self.r.step(self.measured, self.previous, self.home, self.r.limits, intent=intent, frame=frame)

    def engage(self):
        self.assertIsNone(self.step())
        self.assertIsNone(self.step())
        proposal = self.step(1.0)
        self.assertIsNotNone(proposal)
        np.testing.assert_allclose(list(proposal.values()), list(self.previous.values()), atol=1e-7, rtol=0)
        return proposal

    def test_defaults_and_external_model_observe_all_joints(self):
        self.assertEqual(DEFAULTS.arm_velocity_limit, 8.0)
        self.assertEqual(DEFAULTS.return_velocity_limit, 1.0)
        self.assertEqual(DEFAULTS.control_frequency, 50.0)
        self.assertEqual(len(self.r.joint_names), 31)
        self.assertEqual(len(self.r.arm_names), 14)
        self.measured["waist_yaw_joint"] = 0.2
        self.step(intent="HOME")
        self.assertAlmostEqual(self.r.placo_robot.get_joint("waist_yaw_joint"), 0.2)
        with self.assertRaises(ValueError):
            X2TeleopSettings(collision_margin=0.1, collision_trigger=0.02)

    def test_grip_engagement_moves_only_active_arm_and_respects_urdf_speeds(self):
        self.engage()
        self.poses["left_controller"][2] -= 0.03
        proposal = self.step(1.0)
        self.assertIsNotNone(proposal)
        self.assertGreater(max(abs(proposal[n] - self.previous[n]) for n in proposal), 0.001)
        for name in proposal:
            self.assertLessEqual(
                abs(proposal[name] - self.previous[name]), min(8.0, self.r.velocities[name]) / 50 + 1e-7
            )
            if name.startswith("right_"):
                self.assertEqual(proposal[name], self.previous[name])

    def test_targets_follow_torso_without_controller_motion(self):
        self.engage()
        baseline = self.step(1.0)
        self.measured["waist_yaw_joint"] = 0.2
        self.measured["waist_pitch_joint"] = 0.05
        proposal = self.step(1.0)
        self.assertIsNotNone(proposal)
        np.testing.assert_allclose(list(proposal.values()), list(baseline.values()), atol=1e-4, rtol=0)

    def test_repeated_pose_cycles_track_targets_and_preserve_elbow_branch(self):
        def hand_frames(joints):
            for name, value in self.measured.items():
                self.r.placo_robot.set_joint(name, value)
            for name, value in joints.items():
                self.r.placo_robot.set_joint(name, value)
            self.r.placo_robot.update_kinematics()
            torso = np.linalg.inv(self.r.placo_robot.get_T_world_frame("torso_link"))
            return {
                side: torso @ self.r.placo_robot.get_T_world_frame(f"{side}_hand_center") for side in ("left", "right")
            }

        reference = hand_frames(self.home)
        t_pose = self.home.copy()
        for side, sign in (("left", 1), ("right", -1)):
            t_pose.update(
                {
                    f"{side}_shoulder_pitch_joint": 0.0,
                    f"{side}_shoulder_roll_joint": sign * 1.55,
                    f"{side}_elbow_joint": 0.0,
                }
            )

        def advance(desired, grip):
            frames = hand_frames(desired)
            for side in ("left", "right"):
                delta = np.eye(4)
                basis = R_HEADSET_TO_WORLD
                delta[:3, :3] = basis.T @ frames[side][:3, :3] @ reference[side][:3, :3].T @ basis
                self.poses[f"{side}_controller"] = np.r_[
                    basis.T @ (frames[side][:3, 3] - reference[side][:3, 3]),
                    np.roll(tf.quaternion_from_matrix(delta), -1),
                ]
            self.timestamp += 1
            frame = X2XRFrame(self.timestamp, self.poses, {"left": grip, "right": grip})
            proposal = self.r.step(self.measured, self.previous, self.home, self.r.limits, intent="TRACK", frame=frame)
            self.assertIsNotNone(proposal, self.r.hold_reason)
            for name in proposal:
                self.assertLessEqual(
                    abs(proposal[name] - self.previous[name]), min(8, self.r.velocities[name]) / 50 + 1e-7
                )
            self.r.acknowledge(proposal)
            self.previous = proposal
            self.measured.update(proposal)
            return frames

        # Arm both clutches with fresh, released input before starting the replay.
        for _ in range(3):
            self.step(0.0)
        advance(self.home, 1.0)
        for _ in range(3):
            for start, end, label in ((self.home, t_pose, "T"), (t_pose, self.home, "home")):
                for index in range(120):
                    alpha = (1 - np.cos(np.pi * min(index / 79, 1))) / 2
                    desired = {name: start[name] + alpha * (end[name] - start[name]) for name in self.home}
                    target = advance(desired, 1.0)
                actual = hand_frames(self.previous)
                for side in actual:
                    # LeRobot regularizes joint angles and filters the solution;
                    # its upstream objective trades some task-space accuracy for posture.
                    tolerance = 0.015 if os.environ.get("BRIDGE_TEST_BACKEND") == "lerobot" else 0.002
                    self.assertLess(np.linalg.norm(actual[side][:3, 3] - target[side][:3, 3]), tolerance)
                    relative = actual[side][:3, :3] @ target[side][:3, :3].T
                    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
                    self.assertLess(angle, 0.10)
                if label == "home":
                    # Redundant-joint posture differs with upstream manipulability;
                    # preserve the elbow branch while checking hand errors above.
                    for side in ("left", "right"):
                        self.assertLess(self.previous[f"{side}_elbow_joint"], -0.5)

    def test_tracking_loss_rebases_without_replaying_motion_or_requiring_release(self):
        with patch("time.monotonic", return_value=100.0) as clock:
            self.engage()
            clock.return_value = 100.251
            self.assertIsNone(self.step(1.0, advance=False))
            self.assertIn("tracking timed out", self.r.hold_reason)
            self.poses["left_controller"][2] -= 0.5
            clock.return_value = 100.26
            proposal = self.step(1.0)
            self.assertIsNotNone(proposal)
            np.testing.assert_allclose(list(proposal.values()), list(self.previous.values()), atol=1e-7, rtol=0)
            self.assertIsNone(self.r.hold_reason)

    def test_inactive_arm_servo_lag_does_not_make_command_bounds_infeasible(self):
        self.measured["right_elbow_joint"] += 0.2
        self.engage()
        self.poses["left_controller"][2] -= 0.03
        proposal = self.step(1.0)
        self.assertIsNotNone(proposal)
        for name in self.home:
            if name.startswith("right_"):
                self.assertEqual(proposal[name], self.previous[name])

    def test_collision_backtracking_accepts_only_independently_checked_shorter_step(self):
        self.engage()
        self.poses["left_controller"][2] -= 0.03
        validate = self.r._validate_motion
        checked = []

        def reject_full_step(measured, previous, candidate):
            checked.append(candidate.copy())
            if len(checked) == 1:
                raise RuntimeError("nonlinear clearance rejected full step")
            validate(measured, previous, candidate)

        with patch.object(self.r, "_validate_motion", side_effect=reject_full_step):
            proposal = self.step(1.0)
        self.assertIsNotNone(proposal)
        self.assertGreaterEqual(len(checked), 2)
        full = self.r._positions(checked[0])
        for name in self.home:
            self.assertLessEqual(
                abs(proposal[name] - self.previous[name]), abs(full[name] - self.previous[name]) + 1e-7
            )

    def test_hold_reasons_for_grip_release_robot_gate_and_rejected_motion(self):
        self.engage()
        self.assertIsNone(self.step(0.0))
        self.assertIn("no grip", self.r.hold_reason)
        self.assertIsNone(self.step(intent="HOLD"))
        self.assertIn("RoboJuDo requested HOLD", self.r.hold_reason)
        with patch.object(self.r, "_validate_motion", side_effect=RuntimeError("test collision clearance")):
            self.assertIsNone(self.step(intent="HOME"))
        self.assertEqual(self.r.hold_reason, "test collision clearance")

    def test_home_needs_no_xr_and_respects_one_rad_per_second(self):
        self.previous["left_elbow_joint"] += 0.1
        self.measured.update(self.previous)
        proposal = self.r.step(self.measured, self.previous, self.home, self.r.limits, intent="HOME")
        self.assertIsNotNone(proposal)
        self.assertEqual(self.r.status, "homing")
        self.assertLess(proposal["left_elbow_joint"], self.previous["left_elbow_joint"])
        for name in proposal:
            self.assertLessEqual(abs(proposal[name] - self.previous[name]), 1 / 50 + 1e-7)

    def test_limit_intersection_and_float32_targets(self):
        limits = self.r.limits.copy()
        limits["left_elbow_joint"] = (-0.9, -0.85)
        self.previous["left_elbow_joint"] = -0.86
        proposal = self.r.step(self.measured, self.previous, self.home, limits, intent="HOME")
        self.assertIsNotNone(proposal)
        self.assertTrue(-0.9 <= proposal["left_elbow_joint"] <= -0.85)
        self.assertTrue(all(float(np.float32(value)) == value for value in proposal.values()))

    def test_float32_quantization_rounds_joint_and_speed_bounds_inward(self):
        self.r.active = {"left": True, "right": True}
        robot = self.r.placo_robot
        for name, value in self.measured.items():
            robot.set_joint(name, value)
        baseline = robot.state.q.copy()
        joint = "right_shoulder_roll_joint"
        offset = robot.get_joint_offset(joint)
        # Deployment joint limits, URDF limits and representative speed windows.
        for lower, upper in ((-2.993, 0.061), self.r.limits[joint], (-0.123, 0.037)):
            for endpoint in (lower, upper):
                with self.subTest(lower=lower, upper=upper, endpoint=endpoint):
                    candidate = baseline.copy()
                    candidate[offset] = endpoint
                    result = self.r._quantize(candidate, {joint: (lower, upper)}, self.previous)
                    value = float(result[offset])
                    self.assertGreaterEqual(value, lower)
                    self.assertLessEqual(value, upper)
                    self.assertEqual(value, float(np.float32(value)))

    def test_return_obeys_reduced_overall_arm_cap(self):
        self.previous["left_elbow_joint"] += 0.1
        self.measured.update(self.previous)
        proposal = self.r.step(
            self.measured,
            self.previous,
            self.home,
            self.r.limits,
            intent="HOME",
            arm_velocity_limit=0.25,
            return_velocity_limit=1.0,
        )
        self.assertIsNotNone(proposal)
        for name in proposal:
            self.assertLessEqual(abs(proposal[name] - self.previous[name]), 0.25 / 50 + 1e-7)

    def test_inactive_targets_cannot_override_intersected_position_limits(self):
        self.engage()
        joint = "right_shoulder_roll_joint"
        cases = [
            (self.r.limits, self.r.limits[joint][0] - 0.001),
            (dict(self.r.limits, **{joint: (-0.2, -0.095)}), -0.09),
        ]
        for limits, previous in cases:
            with self.subTest(previous=previous):
                self.previous[joint] = float(np.float32(previous))
                self.r.rebase()
                # Isolate position checks from collisions due to large servo error.
                with patch.object(self.r, "_validate_motion"):
                    frame = X2XRFrame(self.timestamp + 1, self.poses, {"left": 1.0, "right": 0.0})
                    result = self.r.step(self.measured, self.previous, self.home, limits, intent="TRACK", frame=frame)
                self.assertIsNone(result)
                self.assertIn("outside compatible limits", self.r.hold_reason)

    def test_held_targets_cannot_bypass_final_bounds_check(self):
        self.r.active = {"left": True, "right": False}
        joint = "right_shoulder_roll_joint"
        candidate = self.r.placo_robot.state.q.copy()
        previous = dict(self.previous, **{joint: float(np.float32(-2.993))})
        with self.assertRaisesRegex(RuntimeError, "held arm target"):
            self.r._quantize(candidate, {joint: (-2.993, 0.061)}, previous)

    def test_large_finite_quaternion_is_normalized_without_overflow(self):
        frame = X2XRFrame(1, {"headset": [0, 0, 0, 1e200, 0, 0, 1e200]}, {})
        transform = self.r._pose(frame, "headset")
        np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-12)
        self.assertLess(transform[1, 2], -0.99)

    def test_real_colliding_path_is_rejected(self):
        self.step(intent="HOME")
        robot = self.r.placo_robot
        measured = robot.state.q.copy()
        rng = np.random.default_rng(14)
        for _ in range(100):
            for joint in self.r.arm_joints["left"]:
                robot.set_joint(joint, rng.uniform(*self.r.limits[joint]))
            if min(self.r.collision_checker.distances()) < 0:
                unsafe = robot.state.q.copy()
                break
        else:
            self.fail("failed to construct a real colliding configuration")
        with self.assertRaises(RuntimeError):
            self.r._validate_motion(measured, measured.copy(), unsafe)

    def test_invalid_pose_holds_and_resets_clutches(self):
        self.engage()
        self.poses["left_controller"][0] = np.nan
        self.assertIsNone(self.step(1.0))
        self.poses["left_controller"][0] = 0.0
        proposal = self.step(1.0)
        self.assertIsNotNone(proposal)
        np.testing.assert_allclose(list(proposal.values()), list(self.previous.values()), atol=1e-7, rtol=0)

    def test_explicit_reset_requires_release_but_rebase_preserves_engagement(self):
        self.engage()
        self.r.rebase()
        self.assertFalse(self.r._need_release)
        self.r.reset()
        self.assertTrue(self.r._need_release)


class MemorySocket:
    def __init__(self):
        self.messages = []
        self.frames = []

    def setsockopt(self, *args):
        pass

    def getsockopt(self, *args):
        return False

    def bind(self, *args):
        pass

    def connect(self, *args):
        pass

    def close(self, **kwargs):
        pass

    def recv(self, **kwargs):
        if not self.messages:
            raise zmq.Again()
        message = self.messages.pop(0)
        return message if isinstance(message, bytes) else json.dumps(message).encode()

    def send_json(self, message, **kwargs):
        self.frames.append(message)


class MemoryContext:
    def socket(self, *args):
        return MemorySocket()


class BridgeTests(unittest.TestCase):
    def setUp(self):
        with patch("zmq.Context.instance", return_value=MemoryContext()):
            self.bridge = RoboJuDoX2Bridge()
        self.addCleanup(self.bridge.close)
        self.measured, self.home = fixture(self.bridge.retargeter)
        self.state = {
            "schema_version": 1,
            "type": "upper_body_state",
            "session_id": "robot",
            "generation": 1,
            "state_id": 1,
            "command_id": 0,
            "intent": "HOME",
            "valid": True,
            "positions": self.measured,
            "joint_age_s": dict.fromkeys(self.measured, 0.0),
            "last_commanded_positions": self.home,
            "default_positions": self.home,
            "position_limits": {name: list(bounds) for name, bounds in self.bridge.retargeter.limits.items()},
            "control_dt": 0.02,
            "arm_velocity_limit": 8.0,
            "return_velocity_limit": 1.0,
            "feedback_timeout_s": 0.1,
        }

    def test_home_roundtrip_without_sdk_initialization(self):
        self.state["visualize_targets"] = True
        self.bridge.feedback.messages.append(self.state)
        self.assertTrue(self.bridge.poll())
        proposal = self.bridge.commands.frames[-1]
        self.assertEqual(proposal["intent"], "HOME")
        self.assertEqual(proposal["status"], "homing")
        self.assertEqual(proposal["state_id"], self.state["state_id"])
        self.assertEqual(proposal["command_id"], self.state["command_id"])
        self.assertEqual(set(proposal["positions"]), set(self.home))
        self.assertIsNone(self.bridge.xr_client)
        self.assertEqual(set(proposal["target_poses"]), {"left", "right"})

    def test_rejected_ik_can_publish_display_targets_without_a_command(self):
        self.state["visualize_targets"] = True
        self.bridge.retargeter.target_poses = {"left": np.eye(4).tolist()}
        self.bridge.feedback.messages.append(self.state)
        with patch.object(self.bridge.retargeter, "step", return_value=None):
            self.assertFalse(self.bridge.poll())
        frame = self.bridge.commands.frames[-1]
        self.assertEqual(frame["type"], "upper_body_visualization")
        self.assertIsNone(frame["positions"])
        self.assertIn("left", frame["target_poses"])
        self.assertIsNone(self.bridge._pending_baseline)

    def test_invalid_stale_and_duplicate_feedback_cannot_produce_commands(self):
        for mutate in ("age", "partial", "nan", "duplicate_json"):
            bad = copy.deepcopy(self.state)
            if mutate == "age":
                bad["joint_age_s"]["left_elbow_joint"] = 1.0
            elif mutate == "partial":
                bad["positions"].pop("waist_yaw_joint")
            elif mutate == "nan":
                bad["positions"]["waist_yaw_joint"] = float("nan")
            else:
                bad = b'{"schema_version": 1, "schema_version": 1}'
            self.bridge.feedback.messages.append(bad)
            self.assertFalse(self.bridge.poll(), mutate)
        self.bridge.feedback.messages.append(self.state)
        self.assertTrue(self.bridge.poll())
        self.bridge.feedback.messages.append(self.state)
        self.assertFalse(self.bridge.poll())

    def test_feedback_outage_resets_clutches(self):
        self.bridge.feedback.messages.append(self.state)
        self.assertTrue(self.bridge.poll())
        with patch("time.monotonic", return_value=time.monotonic() + 1):
            self.assertFalse(self.bridge.poll())
        self.assertTrue(self.bridge.retargeter._need_release)
        self.assertEqual(self.bridge.hold_reason, "robot feedback missing or timed out")

    def test_invalid_feedback_retains_rejection_reason(self):
        bad = copy.deepcopy(self.state)
        bad["positions"].pop("waist_yaw_joint")
        self.bridge.feedback.messages.append(bad)
        self.assertFalse(self.bridge.poll())
        self.assertIn("invalid robot feedback", self.bridge.hold_reason)
        self.assertIn("positions", self.bridge.hold_reason)

    def test_stale_packet_before_valid_feedback_does_not_rebase_tracking(self):
        bad = copy.deepcopy(self.state)
        bad["joint_age_s"]["left_elbow_joint"] = 1.0
        self.bridge.feedback.messages.extend([bad, self.state])
        # Keep the current generation so only the receive-batch path is tested.
        self.bridge._key = (self.state["session_id"], self.state["generation"])
        with patch.object(self.bridge.retargeter, "rebase") as rebase:
            with patch.object(self.bridge.retargeter, "step", return_value=None):
                self.assertFalse(self.bridge.poll())
        rebase.assert_not_called()

    def test_constructor_failure_closes_already_created_resources(self):
        retargeter = Mock()
        commands = Mock()
        context = Mock()
        context.socket.side_effect = [commands, RuntimeError("socket creation failed")]
        with (
            patch("xrobotoolkit_teleop.integrations.robojudo.ExternalStateRetargeter", return_value=retargeter),
            patch("zmq.Context.instance", return_value=context),
            self.assertRaisesRegex(RuntimeError, "socket creation failed"),
        ):
            RoboJuDoX2Bridge()
        commands.close.assert_called_once_with(linger=0)
        retargeter.close.assert_called_once_with()

    def test_close_is_idempotent_for_an_owned_xr_client(self):
        xr = Mock()
        with patch("zmq.Context.instance", return_value=MemoryContext()):
            bridge = RoboJuDoX2Bridge(xr_client=xr)
        bridge.close()
        bridge.close()
        xr.close.assert_called_once_with()

    def test_sdk_diagnostics_report_unchanged_timestamp_and_grip_values(self):
        from types import SimpleNamespace

        xr = SimpleNamespace(
            get_timestamp_ns=lambda: 123,
            get_pose_by_name=lambda name: [0, 0, 0, 0, 0, 0, 1],
            get_key_value_by_name=lambda name: 0.95 if name == "left_grip" else 0.0,
            get_button_state_by_name=lambda name: False,
            close=lambda: None,
        )
        self.bridge.xr_client = xr
        with patch("time.monotonic", return_value=100.0) as clock:
            self.bridge._xr_report_at = 100.0
            self.bridge._xr_frame()
            clock.return_value = 102.0
            self.bridge._xr_frame()
            with self.assertLogs("xrobotoolkit_teleop.integrations.robojudo", level="INFO") as logs:
                self.bridge.report_xr_diagnostics()
        self.assertIn("timestamp=123", logs.output[0])
        self.assertIn("age_s=2.0", logs.output[0])
        self.assertIn("observed_updates_hz=0.5", logs.output[0])
        self.assertIn("'left': 0.95", logs.output[0])

    def test_waits_for_acknowledgement_instead_of_resending_old_baseline(self):
        self.bridge.feedback.messages.append(self.state)
        self.assertTrue(self.bridge.poll())
        pending = copy.deepcopy(self.state)
        pending["state_id"] += 1
        self.bridge.feedback.messages.append(pending)
        self.assertFalse(self.bridge.poll())
        acknowledged = copy.deepcopy(pending)
        acknowledged["state_id"] += 1
        acknowledged["command_id"] += 1
        self.bridge.feedback.messages.append(acknowledged)
        self.assertTrue(self.bridge.poll())
        self.assertEqual(len(self.bridge.commands.frames), 2)

    def test_matching_acknowledgement_can_arrive_after_local_wait_deadline(self):
        with patch("time.monotonic", return_value=100.0) as clock:
            self.bridge.feedback.messages.append(copy.deepcopy(self.state))
            self.assertTrue(self.bridge.poll())
            clock.return_value = 100.11
            self.state["state_id"] += 1
            self.state["command_id"] += 1
            self.bridge.feedback.messages.append(self.state)
            self.assertTrue(self.bridge.poll())
            self.assertAlmostEqual(self.bridge._timing_ms["ack_wait"], 110.0)

    def test_roundtrip_diagnostics_include_pipeline_compute_and_ack_delays(self):
        self.state["pipeline_timing_ms"] = {"policy": 55.0, "env_step": 4.0}
        with patch("time.monotonic", return_value=100.0) as clock:
            self.bridge.feedback.messages.append(copy.deepcopy(self.state))
            self.assertTrue(self.bridge.poll())
            self.assertEqual(self.bridge._timing_ms["policy"], 55.0)
            self.assertEqual(self.bridge._timing_ms["env_step"], 4.0)
            self.assertIn("retarget", self.bridge._timing_ms)
            self.assertIn("measurement_age_lower_bound", self.bridge._timing_ms)
            clock.return_value = 100.06
            self.state["state_id"] += 1
            self.state["command_id"] += 1
            self.bridge.feedback.messages.append(self.state)
            self.assertTrue(self.bridge.poll())
            self.assertAlmostEqual(self.bridge._timing_ms["ack_wait"], 60.0)

    def test_optional_timing_metadata_cannot_break_control_or_retain_old_mode_values(self):
        self.state["pipeline_timing_ms"] = {"policy": 55.0, "env_step": 4.0}
        self.bridge.feedback.messages.append(copy.deepcopy(self.state))
        self.assertTrue(self.bridge.poll())
        self.state["state_id"] += 1
        self.state["command_id"] += 1
        self.state["pipeline_timing_ms"] = {"policy": 10**400, "env_step": float("nan")}
        self.bridge.feedback.messages.append(copy.deepcopy(self.state))
        self.assertTrue(self.bridge.poll())
        self.assertNotIn("policy", self.bridge._timing_ms)
        self.assertNotIn("env_step", self.bridge._timing_ms)
        self.state["state_id"] += 1
        self.state["generation"] += 1
        self.state["intent"] = "HOLD"
        self.bridge.feedback.messages.append(self.state)
        self.assertFalse(self.bridge.poll())
        self.assertNotIn("ack_wait", self.bridge._timing_ms)

    def test_lost_home_proposal_retries_after_deadline(self):
        with patch("time.monotonic", return_value=100.0) as clock:
            self.bridge.feedback.messages.append(self.state)
            self.assertTrue(self.bridge.poll())
            clock.return_value = 100.11
            pending = copy.deepcopy(self.state)
            pending["state_id"] += 1
            self.bridge.feedback.messages.append(pending)
            self.assertTrue(self.bridge.poll())
        self.assertEqual(len(self.bridge.commands.frames), 2)

    def test_tracking_retry_and_command_timeout_generation_preserve_engagement(self):
        self.state["intent"] = "TRACK"
        poses = {name: [0, 0, 0, 0, 0, 0, 1] for name in ("headset", "left_controller", "right_controller")}
        with patch("time.monotonic", return_value=100.0) as clock:
            for index in range(5):
                self.state["state_id"] = index + 1
                frame = X2XRFrame(index + 1, poses, {"left": 0.0 if index < 2 else 1.0, "right": 0.0})
                if index == 3:
                    clock.return_value = 100.11
                    poses["left_controller"][2] = -0.5
                if index == 4:
                    self.state["generation"] += 1
                    self.state["reset_reason"] = "tracking command timeout"
                self.bridge.feedback.messages.append(copy.deepcopy(self.state))
                with patch.object(self.bridge, "_xr_frame", return_value=frame):
                    sent = self.bridge.poll()
                if index >= 2:
                    self.assertTrue(sent)
                    np.testing.assert_allclose(
                        list(self.bridge.commands.frames[-1]["positions"].values()),
                        list(self.home.values()),
                        atol=1e-7,
                        rtol=0,
                    )
            self.assertFalse(self.bridge.retargeter._need_release)
            self.state["generation"] += 1
            self.state["state_id"] += 1
            self.state["reset_reason"] = "control intent changed"
            self.bridge.feedback.messages.append(copy.deepcopy(self.state))
            with patch.object(self.bridge, "_xr_frame", return_value=frame):
                self.assertFalse(self.bridge.poll())
            self.assertTrue(self.bridge.retargeter._need_release)

    def test_computation_overrun_resets_clutches_and_drops_proposal(self):
        with patch("time.monotonic", return_value=100.0) as clock:

            def overrun(*args, **kwargs):
                clock.return_value = 100.2
                self.bridge.retargeter._need_release = False
                return self.home.copy()

            self.bridge.feedback.messages.append(self.state)
            with patch.object(self.bridge.retargeter, "step", side_effect=overrun):
                self.assertFalse(self.bridge.poll())
        self.assertFalse(self.bridge.retargeter._need_release)
        self.assertEqual(self.bridge.commands.frames, [])
        self.assertEqual(self.bridge.hold_reason, "robot measurements expired during retargeting")


if __name__ == "__main__":
    unittest.main()
