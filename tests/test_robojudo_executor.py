"""Desired IK publishing and numeric executor checks, without robot hardware."""

import unittest
from unittest.mock import patch

import numpy as np
from test_robojudo_x2 import MemoryContext, fixture
from xrobotoolkit_teleop.common.executor_model import ExecutorModel, model_fingerprint
from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge


class ExecutorBridgeTests(unittest.TestCase):
    def test_shared_collision_validator_matches_bridge_and_detects_partial_application(self):
        bridge = RoboJuDoBridge(robot="x2_omnihand", context=MemoryContext())
        self.addCleanup(bridge.close)
        r = bridge.retargeter
        r.configure_collision_policy(TeleopSettings(collision_checking=True))
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
        executor = ExecutorModel(bridge.retargeter.profile, TeleopSettings(collision_checking=True))
        start = executor.robot.state.q.copy()
        target = start.copy()
        for names in executor.profile.hands.groups.values():
            target[executor.robot.get_joint_offset(names[0])] += 0.04
        executor._project_hands(target)
        executor.MAX_COLLISION_SAMPLES = 4
        with self.assertRaisesRegex(RuntimeError, "sampling budget"):
            executor._application_samples(start, target)
