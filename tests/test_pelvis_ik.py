"""Check preferred and assisted waist yaw using the X2 model."""

import importlib.util
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import numpy as np
from xrobotoolkit_teleop.integrations.pelvis_ik import WAIST_YAW, HandGoals, PelvisIKBridge, WaistAssistedBackend


class PelvisBridgeTests(unittest.TestCase):
    def test_home_requires_acknowledged_and_measured_arm_convergence(self):
        bridge = PelvisIKBridge.__new__(PelvisIKBridge)
        bridge.bridge_id = "bridge"
        bridge._home_frame_id = 5
        bridge.retargeter = SimpleNamespace(arm_joints={"left": ("arm",)})
        state = {
            "applied_bridge_id": "bridge",
            "applied_frame_id": 4,
            "xr_posture": {"available": True, "targets": [0.64, 0.0], "defaults": [0.64, 0.0]},
            "default_positions": {"arm": 0.0},
            "last_commanded_positions": {"arm": 0.0},
            "positions": {"arm": 0.2},
        }
        self.assertFalse(bridge._home_complete(state))
        state["applied_frame_id"] = 5
        self.assertFalse(bridge._home_complete(state))
        state["positions"]["arm"] = 0.01
        self.assertTrue(bridge._home_complete(state))

    def test_applied_arm_frame_is_required_to_adopt_assisted_yaw(self):
        bridge = PelvisIKBridge.__new__(PelvisIKBridge)
        bridge.bridge_id = "bridge"
        bridge._height, bridge._yaw = 0.6, 0.0
        bridge._pending_yaw = {5: 0.2, 7: 0.4}
        bridge._acknowledge_waist({"applied_bridge_id": "other", "applied_frame_id": 5})
        self.assertEqual(bridge._yaw, 0.0)
        bridge._acknowledge_waist({"applied_bridge_id": "bridge", "applied_frame_id": 6})
        self.assertEqual(bridge._yaw, 0.0)
        self.assertEqual(bridge._pending_yaw, {7: 0.4})
        bridge._acknowledge_waist({"applied_bridge_id": "bridge", "applied_frame_id": 7})
        self.assertEqual(bridge._yaw, 0.4)


@unittest.skipUnless(
    importlib.util.find_spec("casadi") and importlib.util.find_spec("pinocchio"),
    "requires the LeRobot XR environment",
)
class WaistAssistTests(unittest.TestCase):
    def test_waist_yaw_moves_only_when_fixed_yaw_misses(self):
        from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
        from xrobotoolkit_teleop.ik.types import IKProblem, LinearConstraint
        from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter
        from xrobotoolkit_teleop.integrations.robot_profiles.x2 import load_omnihand_profile

        settings = TeleopSettings(collision_checking=False)
        stack = ExitStack()
        self.addCleanup(stack.close)
        profile = stack.enter_context(load_omnihand_profile(settings))
        retargeter = ExternalStateRetargeter(settings, ik_backend="lerobot", profile=nullcontext(profile))
        self.addCleanup(retargeter.close)
        backend = WaistAssistedBackend(retargeter)
        yaw = WAIST_YAW
        measured = profile.spec.initial_positions.copy()
        arms = {joint: measured[joint] for joint in profile.spec.joints}
        bounds = np.array(list(arms.values()))
        empty = LinearConstraint(np.zeros((0, len(arms))), np.zeros(0))

        for sides in (("left",), ("right",), ("left", "right")):
            for goal_yaw in (0.0, 0.65):
                with self.subTest(sides=sides, goal_yaw=goal_yaw):
                    retargeter.robot.set_joint(yaw, goal_yaw)
                    retargeter.robot.update_kinematics()
                    targets = {
                        side: retargeter.robot.get_T_world_frame(profile.groups[side].frame).copy() for side in sides
                    }
                    retargeter.robot.set_joint(yaw, measured[yaw])
                    retargeter.robot.update_kinematics()
                    backend.begin(measured, 0.0, (-1.5708, 1.5708))
                    proposal = backend.propose(
                        IKProblem(
                            measured,
                            arms,
                            targets,
                            frozenset(name for side in sides for name in profile.groups[side].joints),
                            bounds,
                            bounds,
                            empty,
                            empty,
                            profile.spec.dt,
                        )
                    )
                    self.assertEqual(proposal.positions, arms)
                    self.assertAlmostEqual(retargeter.robot.get_joint(yaw), measured[yaw])
                    if goal_yaw:
                        self.assertGreater(backend.solved_yaw, 0.25)
                        self.assertLessEqual(backend.solved_yaw, 1.5708)
                    else:
                        self.assertEqual(backend.solved_yaw, 0.0)

        # Keeping the controller still preserves the pelvis-local goal even if
        # the measured waist has already started turning toward its target.
        hand_goals = HandGoals(retargeter)
        grips = {"left": 1.0, "right": 0.0}
        controllers = {"left": np.eye(4), "right": np.eye(4)}
        original = hand_goals.update(grips, controllers, measured)
        turned = hand_goals.update(grips, controllers, {**measured, yaw: 0.4})
        np.testing.assert_array_equal(turned["left"], original["left"])

        # A repeated body frame can carry a new grip but cached controller poses.
        hand_goals.reset()
        stale_grips = hand_goals.eligible_grips(grips, can_capture=False)
        self.assertEqual(hand_goals.update(stale_grips, controllers, measured), {})
        fresh_controllers = {side: pose.copy() for side, pose in controllers.items()}
        fresh_controllers["left"][0, 3] = 0.15
        fresh_grips = hand_goals.eligible_grips(grips, can_capture=True)
        anchored = hand_goals.update(fresh_grips, fresh_controllers, measured)
        steady = hand_goals.update(fresh_grips, fresh_controllers, measured)
        np.testing.assert_array_equal(steady["left"], anchored["left"])
