"""Unified desired-target bridge, policy handshake and interruption behavior."""

import copy
import os
import unittest
from unittest.mock import patch

from test_robojudo_x2 import MemoryContext, fixture
from xrobotoolkit_teleop.integrations.retargeter import XRFrame
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge


def feedback_state(bridge, collision=False):
    r = bridge.retargeter
    measured, home = fixture(r)
    home.update(dict.fromkeys(r.hand_names, 0.0))
    state = dict(
        schema_version=3,
        type="upper_body_state",
        session_id="robot",
        generation=1,
        state_id=1,
        command_id=0,
        intent="TRACK",
        valid=True,
        positions=measured,
        joint_age_s=dict.fromkeys(measured, 0.0),
        last_commanded_positions=home,
        default_positions=home.copy(),
        position_limits={
            n: list(bounds)
            for n, bounds in (r.limits | ({} if r.profile.hands is None else r.profile.hands.limits)).items()
        },
        executor={
            **bridge._model_identity,
            "collision_checking": collision,
            "collision_margin": 0.02,
            "collision_trigger": 0.08,
        },
        control_dt=0.02,
        arm_velocity_limit=8.0,
        return_velocity_limit=1.0,
        feedback_timeout_s=0.1,
    )

    if r.hand_names:
        state["executor"].update(hand_command_mode="direct", hand_velocity_limit=2.0)
    return state


class ArmBridgeTests(unittest.TestCase):
    robot = "x2"
    finger_mode = "direct"

    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=100.0))
        self.bridge = RoboJuDoBridge(
            robot=self.robot, context=MemoryContext(), ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo")
        )
        self.addCleanup(self.bridge.close)
        self.state = feedback_state(self.bridge)
        if self.bridge.retargeter.hand_names:
            self.state["executor"]["hand_command_mode"] = self.finger_mode
        self.poses = {n: [0, 0, 0, 0, 0, 0, 1] for n in ("headset", "left_controller", "right_controller")}

    def send(self):
        self.bridge.feedback.messages.append(copy.deepcopy(self.state))
        return self.bridge.receive_feedback()

    def test_publishing_and_solver_history_do_not_wait_for_completion(self):
        r = self.bridge.retargeter
        with (
            patch.object(r, "_validate_motion", side_effect=AssertionError("robot owns executable validation")),
            patch.object(r.backend, "accept", wraps=r.backend.accept) as accept,
        ):
            for i in range(15):
                self.clock.return_value = 100 + i * 0.02
                self.state["state_id"] += 1
                self.bridge.feedback.messages.append(copy.deepcopy(self.state))
                frame = XRFrame(
                    i + 1, self.poses, {"left": float(i >= 2), "right": 0.0}, triggers={"left": 0.8, "right": 0.0}
                )
                with patch.object(self.bridge, "_xr_frame", return_value=frame):
                    self.bridge.poll()
            self.assertGreater(accept.call_count, 0)
        message = self.bridge.commands.frames[-1]
        self.assertEqual(message["schema_version"], 3)
        self.assertEqual(message["type"], "upper_body_desired")
        self.assertEqual(message["command_id"], 0)
        self.assertEqual(message["executor"], self.state["executor"])
        if r.hand_names:
            self.assertAlmostEqual(message["positions"]["L_index_mcp_joint"], 0.8)
            self.assertIn("L_index_mcp_joint", message["active_joint_names"])
        right = set(r.arm_joints["right"])
        self.assertFalse(right.intersection(message["active_joint_names"]))

    def test_failed_local_history_commit_holds_after_publication(self):
        self.assertTrue(self.send())
        desired = dict(self.state["last_commanded_positions"])
        r = self.bridge.retargeter
        with (
            patch.object(r, "step", return_value=desired),
            patch.object(r, "acknowledge", return_value=False),
            patch.object(self.bridge, "_xr_frame", return_value=XRFrame(1, self.poses, {"left": 0, "right": 0})),
        ):
            self.assertFalse(self.bridge.poll())
        self.assertEqual(len(self.bridge.commands.frames), 1)
        self.assertEqual(self.bridge.status, "hold")
        self.assertIsNone(self.bridge._latest)

    def test_first_completion_feedback_is_observed_once_without_publishing(self):
        self.assertTrue(self.send())
        self.bridge._command_log.sent("robot", 1, 123, 100.0)
        self.state.update(state_id=2, applied_bridge_id=self.bridge.bridge_id, applied_frame_id=123, command_id=1)
        self.clock.return_value = 100.03
        self.assertTrue(self.send())
        self.clock.return_value = 100.06
        self.assertTrue(self.send())
        self.assertEqual(self.bridge._command_log.completed_count, 1)
        self.assertAlmostEqual(self.bridge._command_log.total_ms, 30.0)
        self.assertEqual(self.bridge.commands.frames, [])
        self.bridge._command_log.sent("robot", 1, 124, 100.06)
        self.state.update(generation=2, state_id=3, applied_frame_id=124)
        self.assertTrue(self.send())
        self.assertEqual(len(self.bridge._command_log.pending), 0)
        self.assertEqual(self.bridge._command_log.completed_count, 1)

    def test_policy_is_adopted_and_cannot_change_within_a_session(self):
        self.state["executor"]["collision_checking"] = True
        self.assertTrue(self.send())
        self.assertTrue(self.bridge.retargeter.settings.collision_checking)
        self.assertIsNotNone(self.bridge.retargeter.collision_checker)
        self.state["executor"]["collision_checking"] = False
        self.state["generation"] += 1
        self.state["state_id"] += 1
        self.assertFalse(self.send())
        self.assertTrue(self.bridge.settings.collision_checking)
        self.state["session_id"] = "restarted-robot"
        self.assertTrue(self.send())
        self.assertFalse(self.bridge.settings.collision_checking)

    def test_policy_initialization_failure_holds_without_commands(self):
        self.state["executor"]["collision_checking"] = True
        with patch("xrobotoolkit_teleop.integrations.retargeter.CollisionChecker", side_effect=RuntimeError("failed")):
            self.assertFalse(self.send())
        self.assertIsNone(self.bridge.executor_contract)
        self.assertEqual(self.bridge.commands.frames, [])

    def test_invalid_identity_numeric_and_schema_are_rejected(self):
        for key, value in (("schema_version", 1), ("valid", False), ("control_dt", True)):
            state = copy.deepcopy(self.state)
            state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bridge.validate_feedback(state)
        for key, value in (
            ("model_fingerprint", "wrong"),
            ("robot_profile", "wrong"),
            ("collision_checking", 1),
            ("collision_margin", float("nan")),
        ):
            state = copy.deepcopy(self.state)
            state["executor"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bridge.validate_feedback(state)

    def test_cached_feedback_does_not_renew_measurement_lease(self):
        self.assertTrue(self.send())
        self.clock.return_value = 100.11
        self.assertFalse(self.send())
        self.assertEqual(self.bridge.commands.frames, [])

    def test_initialization_time_is_charged_to_measurement_lease(self):
        original = self.bridge.retargeter.configure_collision_policy

        def slow(settings):
            original(settings)
            self.clock.return_value += 0.11

        with patch.object(self.bridge.retargeter, "configure_collision_policy", side_effect=slow):
            self.assertFalse(self.send())
        self.assertIsNone(self.bridge._latest)
        self.state["state_id"] += 1
        self.assertTrue(self.send())

    def test_intent_and_restart_reset_clutch_without_sampling_xr(self):
        self.assertTrue(self.send())
        self.state["intent"] = "HOLD"
        self.state["generation"] += 1
        self.state["state_id"] += 1
        with patch.object(self.bridge.retargeter, "reset", wraps=self.bridge.retargeter.reset) as reset:
            self.assertTrue(self.send())
            reset.assert_called_once()
        with patch.object(self.bridge, "_xr_frame", side_effect=AssertionError("HOLD does not need XR")):
            self.assertFalse(self.bridge.poll())


class OmniHandBridgeTests(ArmBridgeTests):
    robot = "x2_omnihand"

    def test_filtered_mode_is_adopted_echoed_and_immutable(self):
        self.state["executor"]["hand_command_mode"] = "filtered"
        self.assertTrue(self.send())
        self.assertEqual(self.bridge.executor_contract["hand_command_mode"], "filtered")
        self.state["executor"]["hand_command_mode"] = "direct"
        self.state["state_id"] += 1
        self.assertFalse(self.send())

    def test_invalid_finger_settings_are_rejected(self):
        for key, value in (
            ("hand_command_mode", "unknown"),
            ("hand_velocity_limit", 0),
            ("hand_velocity_limit", float("nan")),
            ("hand_velocity_limit", True),
        ):
            state = copy.deepcopy(self.state)
            state["executor"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.bridge.validate_feedback(state)


class FilteredOmniHandBridgeTests(OmniHandBridgeTests):
    finger_mode = "filtered"
