"""Desired IK publishing and numeric executor checks, without robot hardware."""

import copy
import os
import unittest
from unittest.mock import patch

import numpy as np
from test_robojudo_x2 import MemoryContext, fixture
from xrobotoolkit_teleop.common.executor_model import ExecutorModel, model_fingerprint
from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
from xrobotoolkit_teleop.integrations.retargeter import XRFrame
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge


class ExecutorBridgeTests(unittest.TestCase):
    def test_publishing_and_solver_history_are_independent_of_completion(self):
        bridge = RoboJuDoBridge(
            robot="x2_omnihand",
            ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"),
            context=MemoryContext(),
            authoritative_executor=True,
            settings=TeleopSettings(collision_checking=False),
        )
        self.addCleanup(bridge.close)
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
            default_positions=home,
            position_limits={n: list(bounds) for n, bounds in (r.limits | r.profile.hands.limits).items()},
            filter_contract="robojudo_executor_v1",
            executor=bridge.executor_contract,
            ema_alpha=0.0,
            arm_joint_names=r.arm_names,
            control_dt=0.02,
            arm_velocity_limit=8.0,
            return_velocity_limit=1.0,
            hand_velocity_limit=2.0,
            hand_command_mode="direct",
            feedback_timeout_s=0.3,
        )
        poses = {
            name: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for name in ("headset", "left_controller", "right_controller")
        }
        with (
            patch("time.monotonic", return_value=100.0) as clock,
            patch.object(
                r, "_validate_motion", side_effect=AssertionError("bridge must not validate executable paths")
            ),
            patch.object(r.backend, "accept", wraps=r.backend.accept) as accept,
        ):
            for i in range(15):
                clock.return_value = 100.0 + i * 0.02
                state["state_id"] += 1
                bridge.feedback.messages.append(copy.deepcopy(state))
                if i > 4:
                    poses["left_controller"][2] = -0.02
                frame = XRFrame(
                    i + 1, poses, {"left": float(i >= 2), "right": 0.0}, triggers={"left": 0.8, "right": 0.0}
                )
                with patch.object(bridge, "_xr_frame", return_value=frame):
                    bridge.poll()
            self.assertGreater(accept.call_count, 0)
        self.assertEqual(len(bridge.commands.frames), 14)
        self.assertEqual(len(bridge._outstanding), 0)
        message = bridge.commands.frames[-1]
        self.assertEqual(message["type"], "upper_body_desired")
        self.assertEqual(message["schema_version"], 3)
        self.assertAlmostEqual(message["positions"]["L_index_mcp_joint"], 0.8)
        # Robot can issue different filtered targets and advance more than one command ID.
        state.update(applied_bridge_id=bridge.bridge_id, applied_frame_id=message["frame_id"], command_id=7)
        bridge._observe_completion(state, 100.3)
        self.assertEqual(bridge._completion_count, 1)
        bad = copy.deepcopy(state)
        bad["executor"]["model_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "executor model"):
            bridge.validate_feedback(bad)

    def test_shared_collision_validator_matches_bridge_and_detects_partial_application(self):
        bridge = RoboJuDoBridge(robot="x2_omnihand", context=MemoryContext())
        self.addCleanup(bridge.close)
        r = bridge.retargeter
        executor = ExecutorModel(r.profile, r.settings)
        self.assertEqual(executor.fingerprint, model_fingerprint(r.profile))
        measured, baseline = fixture(r)
        baseline.update(dict.fromkeys(r.hand_names, 0.0))
        measured.update(baseline)
        candidate = dict(baseline)
        candidate["L_index_mcp_joint"] = 0.01
        # Both validators use exactly the same model configuration and sampled paths.
        executor.validate_transition(measured, baseline, candidate)
        for name, value in measured.items():
            r.robot.set_joint(name, value)
        bridge_start = r.robot.state.q.copy()
        bridge_previous, bridge_target = bridge_start.copy(), bridge_start.copy()
        for name, value in candidate.items():
            offset = r.robot.get_joint_offset(name)
            bridge_previous[offset] = baseline[name]
            bridge_target[offset] = value
        r._project_hands(bridge_previous)
        r._project_hands(bridge_target)
        r._validate_motion(bridge_start, bridge_previous, bridge_target)
        start = executor.robot.state.q.copy()
        target = start.copy()
        offsets = [executor.robot.get_joint_offset(names[0]) for names in r.profile.hands.groups.values()]
        target[offsets] += 0.01
        executor._project_hands(target)

        def clearances(*args):
            progress = executor.robot.state.q[offsets] - start[offsets]
            return np.array([-0.01 if progress[0] > 0.005 and progress[1] < 0.005 else 0.1])

        with (
            patch.object(executor.collision_checker, "clearances", side_effect=clearances),
            patch.object(executor.collision_checker, "can_bound", False),
            self.assertRaisesRegex(RuntimeError, "self-collision clearance"),
        ):
            executor._validate_motion(start, start.copy(), target)

    def test_independent_application_sampling_budget_is_not_relaxed(self):
        bridge = RoboJuDoBridge(robot="x2_omnihand", context=MemoryContext())
        self.addCleanup(bridge.close)
        executor = ExecutorModel(bridge.retargeter.profile, bridge.settings)
        start = executor.robot.state.q.copy()
        target = start.copy()
        for names in executor.profile.hands.groups.values():
            target[executor.robot.get_joint_offset(names[0])] += 0.04
        executor._project_hands(target)
        executor.MAX_COLLISION_SAMPLES = 4
        with self.assertRaisesRegex(RuntimeError, "sampling budget"):
            executor._application_samples(start, target)
