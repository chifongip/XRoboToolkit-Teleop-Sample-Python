"""One explicit XR collision toggle preserves non-collision safeguards."""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tyro
from test_robojudo_x2 import fixture
from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.integrations.robojudo import main


class CollisionToggleTests(unittest.TestCase):
    def test_setting_defaults_and_validation(self):
        self.assertTrue(TeleopSettings().collision_checking)
        self.assertFalse(TeleopSettings(collision_checking=False).collision_checking)
        for value in (0, 1, None, "False"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                TeleopSettings(collision_checking=value)
        with self.assertRaisesRegex(ValueError, "arm_velocity_limit"):
            TeleopSettings(collision_checking=False, arm_velocity_limit=0)

    def test_cli_enables_collision_checking_only_when_flag_is_present(self):
        for args, expected in (
            ([], False),
            (["--collision-checking"], True),
        ):
            bridge = Mock()
            bridge.poll.side_effect = KeyboardInterrupt
            with (
                self.subTest(args=args),
                patch("xrobotoolkit_teleop.integrations.robojudo.RoboJuDoBridge", return_value=bridge) as factory,
                self.assertLogs("xrobotoolkit_teleop.integrations.robojudo", level="INFO") as logs,
            ):
                tyro.cli(main, args=args)
            self.assertIs(factory.call_args.args[2].collision_checking, expected)
            self.assertTrue(
                any(f"collision checking: {'enabled' if expected else 'disabled'}" in line for line in logs.output)
            )
            bridge.close.assert_called_once()
        for args in (["--no-collision-checking"], ["--collision-checking", "True"], ["--collision-checking", "False"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                tyro.cli(main, args=args, console_outputs=False)

    def test_bridge_schedule_ignores_completion_and_skips_missed_ticks(self):
        self._check_bridge_schedule(0.003, [0, 0.02, 0.04, 0.06])
        self._check_bridge_schedule(0.055, [0, 0.06, 0.12, 0.18])

    def _check_bridge_schedule(self, work_time, expected):
        now = 0.0
        samples = []
        bridge = Mock(status="tracking", hold_reason=None)

        def sleep(duration):
            nonlocal now
            now += duration

        def poll():
            nonlocal now
            if len(samples) == 4:
                raise KeyboardInterrupt
            samples.append(now)
            now += work_time
            return True

        bridge.poll.side_effect = poll
        with (
            patch("xrobotoolkit_teleop.integrations.robojudo.RoboJuDoBridge", return_value=bridge),
            patch("time.monotonic", side_effect=lambda: now),
            patch("time.sleep", side_effect=sleep),
        ):
            main()
        np.testing.assert_allclose(samples, expected, rtol=0, atol=1e-9)
        bridge.feedback.poll.assert_not_called()
        bridge.close.assert_called_once()

    def test_disabled_arm_tracking_and_homing(self):
        self._check_disabled_motion("x2")

    def test_disabled_finger_and_combined_motion(self):
        self._check_disabled_motion("x2_omnihand")

    def test_nonfinite_configurations_are_rejected_with_either_setting(self):
        for enabled in (False, True):
            r = ExternalStateRetargeter.__new__(ExternalStateRetargeter)
            r.settings = TeleopSettings(collision_checking=enabled)
            for index in range(3):
                for value in (np.nan, np.inf, -np.inf):
                    configurations = [np.zeros(8) for _ in range(3)]
                    configurations[index][-1] = value
                    with (
                        self.subTest(enabled=enabled, configuration=index, value=value),
                        self.assertRaisesRegex(RuntimeError, "non-finite robot command configuration"),
                    ):
                        r._validate_motion(*configurations)

    def _check_disabled_motion(self, robot):
        with patch(
            "xrobotoolkit_teleop.integrations.retargeter.CollisionChecker",
            side_effect=AssertionError("collision checker must not be constructed"),
        ):
            r = ExternalStateRetargeter(
                TeleopSettings(collision_checking=False),
                robot=robot,
                ik_backend=os.environ.get("BRIDGE_TEST_BACKEND", "placo"),
            )
            self.addCleanup(r.close)
            self.assertIsNone(r.collision_checker)
            measured, home = fixture(r)
            if r.hand_names:
                home.update(dict.fromkeys(r.hand_names, 0.0))
            previous = home.copy()
            limits = r.limits | (r.profile.hands.limits if r.hand_names else {})
            poses = {
                name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                for name in ("headset", "left_controller", "right_controller")
            }

            def step(stamp, grip=0.0, trigger=0.0, intent="TRACK"):
                return r.step(
                    measured,
                    previous,
                    home,
                    limits,
                    intent=intent,
                    frame=XRFrame(
                        stamp, poses, {"left": grip, "right": 0.0}, triggers={"left": trigger, "right": trigger}
                    ),
                    control_dt=0.02,
                    arm_velocity_limit=0.5,
                    hand_velocity_limit=1.0,
                )

            def check_bounds(result):
                self.assertIsNotNone(result)
                for name, value in result.items():
                    self.assertTrue(np.isfinite(value))
                    self.assertEqual(value, float(np.float32(value)))
                    self.assertTrue(limits[name][0] <= value <= limits[name][1])
                    cap = (
                        min(0.5, r.velocities[name])
                        if name in r.arm_names
                        else min(1.0, r.profile.hands.velocities[name])
                    )
                    self.assertLessEqual(abs(value - previous[name]), cap * 0.02 + 1e-6)

            with (
                patch.object(r.robot, "distances", side_effect=AssertionError("collision query")),
                patch.object(r, "_application_samples", side_effect=AssertionError("collision sampling")),
                patch.object(r.backend, "propose", wraps=r.backend.propose) as propose,
            ):
                previous["left_elbow_joint"] += 0.1
                check_bounds(step(1, intent="HOME"))
                self.assertTrue(r.acknowledge(r._pending_positions.copy()))
                previous.update(home)
                self.assertIsNone(step(2))
                self.assertIsNone(step(3))
                if r.hand_names:
                    fingers = step(4, trigger=1.0)
                    check_bounds(fingers)
                    self.assertGreater(fingers["L_index_mcp_joint"], 0.0)
                    self.assertEqual({n: fingers[n] for n in r.arm_names}, {n: previous[n] for n in r.arm_names})
                    self.assertTrue(r.acknowledge(fingers))
                check_bounds(step(5, grip=1.0, trigger=1.0))
                poses["left_controller"][2] -= 0.03
                combined = step(6, grip=1.0, trigger=1.0)
                check_bounds(combined)
                self.assertTrue(propose.called)
                problem = propose.call_args.args[0]
                self.assertEqual(problem.hard.A.shape, (0, len(r.arm_names)))
                self.assertEqual(problem.soft.A.shape, (0, len(r.arm_names)))
                self.assertIsNotNone(r._pending_positions)
                self.assertFalse(r.acknowledge({}))
                self.assertIsNone(r._pending_positions)
                with patch("time.monotonic", return_value=100.0):
                    step(7, grip=1.0, trigger=1.0)
                with patch("time.monotonic", return_value=100.3):
                    self.assertIsNone(step(7, grip=1.0, trigger=1.0))
                self.assertIsNone(step(8, intent="HOLD"))
                bad = r.robot.state.q.copy()
                bad[-1] = np.nan
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    r._validate_motion(bad, bad, bad)
                previous["left_elbow_joint"] = 100.0
                self.assertIsNone(step(9, intent="HOME"))


if __name__ == "__main__":
    unittest.main()
