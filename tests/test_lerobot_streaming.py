"""Single-arm warm starts and delayed application through the real retargeter."""

import os
import unittest
from unittest.mock import patch

import numpy as np
from test_robojudo_x2 import fixture
from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame


@unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires native LeRobot dependencies")
class LeRobotStreamingTests(unittest.TestCase):
    def test_single_arm_warmth_survives_inactive_grip_and_resets_on_transitions(self):
        for side, other in (("left", "right"), ("right", "left")):
            with self.subTest(side=side):
                r = ExternalStateRetargeter(
                    TeleopSettings(collision_checking=False), robot="x2_omnihand", ik_backend="lerobot"
                )
                self.addCleanup(r.close)
                measured, previous = fixture(r)
                previous.update(dict.fromkeys(r.hand_names, 0.0))
                home = previous.copy()
                poses = {
                    name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                    for name in ("headset", "left_controller", "right_controller")
                }
                limits = r.limits | r.profile.hands.limits

                def step(
                    stamp, active, r=r, measured=measured, previous=previous, home=home, limits=limits, poses=poses
                ):
                    result = r.step(
                        measured,
                        previous,
                        home,
                        limits,
                        intent="TRACK",
                        frame=XRFrame(stamp, poses, active, triggers={"left": 0.0, "right": 1.0}),
                        control_dt=0.02,
                        streaming=True,
                    )
                    if result is not None:
                        for group in r.arm_joints:
                            if not active[group]:
                                self.assertEqual(
                                    {n: result[n] for n in r.arm_joints[group]},
                                    {n: previous[n] for n in r.arm_joints[group]},
                                )
                        self.assertTrue(r.acknowledge(result))
                        previous.update(result)
                        measured.update(result)
                    return result

                released = {"left": 0.0, "right": 0.0}
                step(1, released)
                step(2, released)
                step(3, {side: 1.0, other: 1.0})
                self.assertIsNotNone(step(4, {side: 1.0, other: 1.0}))
                self.assertIsNotNone(r.backend.solver._warm)
                # Releasing the other arm still invalidates the dual/primal seed.
                with patch.object(r.backend, "reset", wraps=r.backend.reset) as reset:
                    self.assertIsNotNone(step(5, {side: 1.0, other: 0.0}))
                    reset.assert_called_once_with(frozenset(r.arm_joints[other]))
                self.assertEqual(r.solver_diagnostics["warm_calls"], 0)
                # Keeping it released must not reset the continuing arm again.
                with patch.object(r.backend, "reset", wraps=r.backend.reset) as reset:
                    self.assertIsNotNone(step(6, {side: 1.0, other: 0.0}))
                    reset.assert_not_called()
                self.assertEqual(r.solver_diagnostics["warm_calls"], 1)
                self.assertEqual(r.solver_diagnostics["calls"], 1)
                # Engagement anchors the newly active arm and clears diagnostics.
                self.assertIsNotNone(step(7, {side: 1.0, other: 1.0}))
                self.assertEqual(r.solver_diagnostics, {})
                self.assertIsNotNone(step(8, {side: 1.0, other: 1.0}))
                self.assertEqual(r.solver_diagnostics["warm_calls"], 0)
