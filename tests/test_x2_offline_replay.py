"""End-to-end mocked XR cycles through real IK, command checks, and MuJoCo."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    "x2_offline_replay", Path(__file__).resolve().parents[1] / "scripts/misc/test_x2_ik_offline.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class OfflineReplayTests(unittest.TestCase):
    def check_backend(self, backend):
        for hands in (False, True):
            with (
                self.subTest(backend=backend, omnihand=hands),
                patch(
                    "xrobotoolkit_sdk.init", side_effect=AssertionError("offline replay initialized the XR SDK")
                ) as sdk,
                patch.object(replay.mj_viewer, "launch_passive") as viewer,
                patch("xrobotoolkit_teleop.ik.visualization.IKViewer") as ik_viewer,
                patch.object(replay.time, "sleep"),
            ):
                # Exercise both visualization paths using fake windows; CI stays headless.
                result = replay.run_replay(backend, with_omnihand=hands, visualize=hands, visualize_ik=hands)
                self.assertTrue(result["passed"], result["violations"])
                self.assertEqual(result["failed_cycles"], 0)
                self.assertEqual([e["pose"] for e in result["endpoints"]], ["T", "home", "T", "home"])
                self.assertEqual(result["control_steps"], 601)
                sdk.assert_not_called()
                if hands:
                    viewer.assert_called_once()
                    self.assertEqual(
                        viewer.return_value.__enter__.return_value.sync.call_count, result["control_steps"]
                    )
                    self.assertEqual(ik_viewer.return_value.update.call_count, result["control_steps"])
                    ik_viewer.return_value.close.assert_called_once()
                    viewer.return_value.__exit__.assert_called_once()
                else:
                    viewer.assert_not_called()
                    ik_viewer.assert_not_called()

    def test_placo_cycles_home_to_t_and_back(self):
        self.check_backend("placo")

    @unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires optional Pinocchio/CasADi build")
    def test_lerobot_cycles_home_to_t_and_back(self):
        self.check_backend("lerobot")

    def test_held_commands_do_not_pass_as_a_working_backend(self):
        from xrobotoolkit_teleop.ik.placo_backend import PlacoBackend
        from xrobotoolkit_teleop.ik.types import IKProposal

        def hold(backend, problem):
            return IKProposal(problem.previous.copy())

        with patch.object(PlacoBackend, "propose", hold):
            result = replay.run_replay(cycles=1, move_seconds=0.02, hold_seconds=0.02)
        self.assertFalse(result["passed"])
        self.assertIn("T endpoint position error exceeds 20 mm", result["violations"])

    def test_closing_viewer_stops_and_releases_offline_client(self):
        xr = replay.OfflineXR()
        with (
            patch.object(replay, "OfflineXR", return_value=xr),
            patch.object(replay.mj_viewer, "launch_passive") as viewer,
        ):
            viewer.return_value.__enter__.return_value.is_running.return_value = False
            with self.assertRaises(KeyboardInterrupt):
                replay.run_replay(visualize=True)
        self.assertTrue(xr.closed)
        viewer.return_value.__exit__.assert_called_once()

    def test_interrupted_replay_does_not_exit_successfully(self):
        with (
            patch.object(replay, "run_replay", side_effect=KeyboardInterrupt),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                replay.main()
        self.assertEqual(stopped.exception.code, 130)

    def test_nonfinite_velocity_fails_and_releases_offline_client(self):
        xr = replay.OfflineXR()
        step = replay.AgibotX2TeleopController.step_control

        def corrupt_velocity(controller):
            step(controller)
            controller.mj_data.qvel[0] = np.nan

        output = io.StringIO()
        with (
            patch.object(replay, "OfflineXR", return_value=xr),
            patch.object(replay.AgibotX2TeleopController, "step_control", corrupt_velocity),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as failed:
                replay.main()
        self.assertEqual(failed.exception.code, 1)
        self.assertTrue(xr.closed)
        report = json.loads(output.getvalue())
        self.assertFalse(report["passed"])
        self.assertIn("nonfinite", report["error"])


if __name__ == "__main__":
    unittest.main()
