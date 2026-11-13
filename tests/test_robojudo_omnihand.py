"""Combined arm/finger retargeting against the articulated collision model."""

import copy
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from test_robojudo_x2 import MemoryContext, fixture
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge


class OmniHandStreamingBridgeTests(unittest.TestCase):
    def test_delayed_or_missing_completions_do_not_pause_xr_or_publishing(self):
        bridge = RoboJuDoBridge(
            robot="x2_omnihand", ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"), context=MemoryContext()
        )
        self.addCleanup(bridge.close)
        measured, home = fixture(bridge.retargeter)
        home.update(dict.fromkeys(bridge.retargeter.hand_names, 0.0))
        state = dict(
            schema_version=2,
            type="upper_body_state",
            session_id="robot",
            generation=1,
            state_id=0,
            command_id=0,
            intent="TRACK",
            valid=True,
            positions=measured,
            joint_age_s=dict.fromkeys(measured, 0.0),
            last_commanded_positions=home,
            default_positions=home,
            position_limits={
                n: list(bounds)
                for n, bounds in (bridge.retargeter.limits | bridge.retargeter.profile.hands.limits).items()
            },
            control_dt=0.02,
            arm_velocity_limit=8.0,
            return_velocity_limit=1.0,
            hand_velocity_limit=2.0,
            feedback_timeout_s=0.3,
        )
        poses = {name: [0, 0, 0, 0, 0, 0, 1] for name in ("headset", "left_controller", "right_controller")}
        with patch("time.monotonic", return_value=100.0) as clock:
            for index in range(15):
                clock.return_value = 100.0 + index * 0.02
                frame = XRFrame(index + 1, poses, {"left": 0.0, "right": 0.0}, triggers={"left": 1.0, "right": 0.0})
                # Feedback remains fresh but never acknowledges any command.
                if index % 3 == 0:
                    state["state_id"] += 1
                    bridge.feedback.messages.append(copy.deepcopy(state))
                with patch.object(bridge, "_xr_frame", return_value=frame) as sample:
                    sent = bridge.poll()
                sample.assert_called_once()
                if index >= 1:
                    self.assertTrue(sent)
                    command = bridge.commands.frames[-1]
                    self.assertEqual(command["command_id"], 0)
                    self.assertGreater(command["positions"]["L_index_mcp_joint"], 0)
        self.assertEqual(len(bridge.commands.frames), 14)
        self.assertEqual(len(bridge._outstanding), 14)


class OmniHandRetargeterTests(unittest.TestCase):
    def test_application_samples_include_independent_arm_and_hand_progress(self):
        start = self.r.robot.state.q.copy()
        target = start.copy()
        names = [self.r.arm_names[0], *(names[0] for names in self.r.profile.hands.groups.values())]
        offsets = [self.r.robot.get_joint_offset(n) for n in names]
        target[offsets] += 0.01
        self.r._project_hands(target)
        samples = list(self.r._application_samples(start, target))
        for mask in range(8):
            expected = [target[offset] if mask & (1 << i) else start[offset] for i, offset in enumerate(offsets)]
            self.assertTrue(any(np.allclose(sample[offsets], expected) for sample in samples), mask)

    def test_collision_in_partial_hand_application_rejects_combined_motion(self):
        start = self.r.robot.state.q.copy()
        target = start.copy()
        offsets = [self.r.robot.get_joint_offset(names[0]) for names in self.r.profile.hands.groups.values()]
        target[offsets] += 0.01
        self.r._project_hands(target)

        def clearances(*args):
            progress = self.r.robot.state.q[offsets] - start[offsets]
            # Both endpoints and the synchronized path are clear. Only a
            # left-hand write arriving before the right-hand write collides.
            return np.array([-0.01 if progress[0] > 0.005 and progress[1] < 0.005 else 0.1])

        with (
            patch.object(self.r.collision_checker, "clearances", side_effect=clearances),
            patch.object(self.r.collision_checker, "can_bound", False),
            self.assertRaisesRegex(RuntimeError, "self-collision clearance"),
        ):
            self.r._validate_motion(start, start.copy(), target)

    def test_couplings_cannot_overwrite_arms_or_duplicate_passive_joints(self):
        original = self.r.profile.hands
        first, *remaining = original.couplings
        for invalid in (
            replace(original, couplings=(("left_elbow_joint", first[1], first[2]), *remaining)),
            replace(original, couplings=(first, first, *remaining[1:])),
            replace(original, couplings=((first[0], first[0], first[2]), *remaining)),
        ):
            with self.subTest(couplings=invalid.couplings[0]), self.assertRaises(ValueError):
                replace(self.r.profile, hands=invalid).validate()

    def setUp(self):
        self.r = ExternalStateRetargeter(robot="x2_omnihand", ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"))
        self.addCleanup(self.r.close)
        self.measured, self.home = fixture(self.r)
        self.home.update(dict.fromkeys(self.r.hand_names, 0.0))
        self.previous = self.home.copy()
        self.limits = self.r.limits | self.r.profile.hands.limits
        self.timestamp = 0
        self.poses = {
            n: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for n in ("headset", "left_controller", "right_controller")
        }

    def step(self, grip=0.0, trigger=0.0, right_grip=0.0, right_trigger=0.0, intent="TRACK"):
        self.timestamp += 1
        frame = XRFrame(
            self.timestamp,
            self.poses,
            {"left": grip, "right": right_grip},
            triggers={"left": trigger, "right": right_trigger},
        )
        return self.r.step(
            self.measured,
            self.previous,
            self.home,
            self.limits,
            intent=intent,
            frame=frame,
            control_dt=0.02,
            hand_velocity_limit=2.0,
        )

    def engage(self):
        self.assertIsNone(self.step())
        self.assertIsNone(self.step())

    def test_articulated_profile_and_independent_trigger(self):
        self.assertEqual(len(self.r.joint_names), 69)
        self.assertEqual(len(self.r.command_names), 38)
        self.assertEqual(len(self.r.hand_couplings), 14)
        self.engage()
        result = self.step(trigger=1.0)
        self.assertEqual({n: result[n] for n in self.r.arm_names}, {n: self.previous[n] for n in self.r.arm_names})
        self.assertIsNone(self.r._pending)
        self.assertIsNotNone(result)
        self.assertGreater(result["L_index_mcp_joint"], 0.0)
        self.assertEqual(result["R_index_mcp_joint"], 0.0)
        self.assertLessEqual(max(abs(result[n]) for n in self.r.hand_names), 0.040001)
        self.assertTrue(self.r.acknowledge(result))

    def test_right_hand_mirroring_and_home_holds(self):
        self.engage()
        result = self.step(right_trigger=0.4)
        self.assertIsNotNone(result)
        self.assertGreater(result["R_thumb_roll_joint"], 0.0)
        self.assertLess(result["R_thumb_abad_joint"], 0.0)
        self.r.acknowledge(result)
        self.previous = result
        result = self.step(intent="HOME")
        self.assertIsNotNone(result)
        self.assertEqual({n: result[n] for n in self.r.hand_names}, {n: self.previous[n] for n in self.r.hand_names})

    def test_invalid_trigger_holds_only_that_hand(self):
        self.engage()
        result = self.step(trigger=float("nan"), right_trigger=0.4)
        self.assertEqual(result["L_index_mcp_joint"], 0.0)
        self.assertGreater(result["R_index_mcp_joint"], 0.0)
        self.assertTrue(self.r.acknowledge(result))

    def test_finger_only_motion_respects_hold_and_stale_tracking(self):
        self.engage()
        self.assertIsNone(self.step(trigger=1.0, intent="HOLD"))
        with patch.object(self.r, "_fresh", return_value=False):
            self.assertIsNone(self.step(trigger=1.0))
        self.assertIsNone(self.r._pending_positions)

    def test_fingers_can_move_when_arm_solver_holds(self):
        self.engage()
        self.step(grip=1.0)
        with patch.object(self.r.backend, "propose", side_effect=RuntimeError("arm solver unavailable")):
            result = self.step(grip=1.0, trigger=0.4)
        self.assertIsNotNone(result)
        self.assertGreater(result["L_index_mcp_joint"], 0.0)
        self.assertIsNone(self.r._pending)
        self.assertTrue(self.r.acknowledge(result))

    def test_finger_only_collision_failure_holds(self):
        self.engage()
        with patch.object(self.r, "_validate_motion", side_effect=RuntimeError("finger obstruction")):
            self.assertIsNone(self.step(trigger=1.0))
        self.assertIn("finger obstruction", self.r.hold_reason)
        self.assertIsNone(self.r._pending_positions)

    def test_finger_only_motion_rechecks_current_measured_path(self):
        self.engage()
        result = self.step(trigger=0.4)
        self.r.reject()
        with patch.object(self.r, "_validate_motion", side_effect=RuntimeError("new obstruction")) as checked:
            self.assertIsNone(self.step(trigger=0.4))
        checked.assert_called_once()
        self.assertIsNotNone(result)

    def test_xr_home_button_holds_fingers(self):
        self.engage()
        frame = XRFrame(
            self.timestamp + 1,
            self.poses,
            {"left": 0.0, "right": 0.0},
            return_to_default=True,
            triggers={"left": 1.0, "right": 1.0},
        )
        result = self.r.step(self.measured, self.previous, self.home, self.limits, intent="TRACK", frame=frame)
        self.assertIsNotNone(result)
        self.assertEqual({n: result[n] for n in self.r.hand_names}, {n: self.previous[n] for n in self.r.hand_names})

    def test_combined_collision_failure_can_hold_fingers_and_keep_checked_arms(self):
        self.engage()
        validate = self.r._validate_motion

        def reject_fingers(measured, previous, candidate):
            index = self.r.robot.get_joint_offset("L_index_mcp_joint")
            if candidate[index] > previous[index]:
                raise RuntimeError("finger obstruction")
            return validate(measured, previous, candidate)

        with patch.object(self.r, "_validate_motion", side_effect=reject_fingers):
            result = self.step(grip=1.0, trigger=1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["L_index_mcp_joint"], 0.0)

    def test_acknowledgement_must_include_exact_fingers(self):
        self.engage()
        result = self.step(grip=1.0, trigger=0.5)
        changed = result.copy()
        changed["L_index_mcp_joint"] += 0.01
        self.assertFalse(self.r.acknowledge(changed))
        self.assertIsNone(self.r._pending_positions)

    def test_held_fingers_are_checked_with_arm_path_once(self):
        self.previous["L_index_mcp_joint"] = float(np.float32(0.1))
        validate = self.r._validate_motion
        with patch.object(self.r, "_validate_motion", wraps=validate) as checked:
            result = self.step(intent="HOME")
        self.assertIsNotNone(result)
        self.assertEqual(checked.call_count, 1)
        measured, previous, candidate = checked.call_args.args
        offset = self.r.robot.get_joint_offset("L_index_mcp_joint")
        self.assertEqual(measured[offset], 0.0)
        self.assertEqual(previous[offset], self.previous["L_index_mcp_joint"])
        self.assertEqual(candidate[offset], self.previous["L_index_mcp_joint"])

    def test_finger_targets_round_inside_strict_limits(self):
        self.limits["L_index_mcp_joint"] = (0.0, 0.2)
        self.r.profile.hands.closed.update(dict.fromkeys(self.r.profile.hands.groups["left"], 0.0))
        self.r.profile.hands.closed["L_index_mcp_joint"] = 0.2
        self.engage()
        for _ in range(8):
            result = self.step(grip=1.0, trigger=1.0)
            self.assertIsNotNone(result)
            self.assertLessEqual(result["L_index_mcp_joint"], 0.2)
            self.assertLessEqual(result["L_index_mcp_joint"] - self.previous["L_index_mcp_joint"], 0.04)
            self.r.acknowledge(result)
            self.previous = result
        self.assertGreater(result["L_index_mcp_joint"], 0.19999)

    def test_invalid_held_fingers_are_rejected_before_model_mutation(self):
        self.engage()
        self.assertIsNotNone(self.step(grip=1.0, trigger=0.3))
        measured_model = self.r.robot.state.q.copy()
        self.previous["R_thumb_roll_joint"] = float(np.float32(-0.1))
        with self.assertRaisesRegex(ValueError, "finger baseline"):
            self.step(intent="HOME")
        self.assertIsNone(self.r._pending_positions)
        np.testing.assert_array_equal(self.r.robot.state.q, measured_model)
