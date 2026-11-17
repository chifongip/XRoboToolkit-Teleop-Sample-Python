"""Command-rate and completion RTT summaries use the bridge clock only."""

import unittest
from unittest.mock import patch

from xrobotoolkit_teleop.integrations.robojudo import _CommandLog


class CommandLogTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch("time.monotonic", return_value=10.0))
        self.log = _CommandLog("bridge")

    def state(self, frame_id, **kwargs):
        return dict(session_id="robot", generation=1, applied_bridge_id="bridge", applied_frame_id=frame_id, **kwargs)

    def test_rate_rtt_and_window_reset(self):
        for frame in range(50):
            self.log.sent("robot", 1, frame, 10.0)
        self.log.completed(self.state(0), 10.02)
        self.log.completed(self.state(1), 10.04)
        self.clock.return_value = 10.5
        with self.assertNoLogs("xrobotoolkit_teleop.integrations.robojudo", level="INFO"):
            self.log.report("tracking")
        self.clock.return_value = 11.0
        with self.assertLogs("xrobotoolkit_teleop.integrations.robojudo", level="INFO") as logs:
            self.log.report("tracking")
        self.assertIn("XR: send=50.0 Hz | completion RTT=30.0 ms avg / 40.0 ms max", logs.output[0])
        self.clock.return_value = 12.0
        with self.assertLogs("xrobotoolkit_teleop.integrations.robojudo", level="INFO") as logs:
            self.log.report("hold")
        self.assertIn("send=0.0 Hz | completion RTT=n/a", logs.output[0])
        self.assertEqual(len(self.log.pending), 48)

    def test_duplicate_superseded_unknown_and_foreign_acknowledgements(self):
        for frame in range(4):
            self.log.sent("robot", 1, frame, 10.0)
        self.log.completed(self.state(2), 10.03)
        for frame in (2, 0, 1, 99):
            self.log.completed(self.state(frame), 10.06)
        self.assertEqual(self.log.completed_count, 1)
        self.assertEqual(list(self.log.pending), [("robot", 1, 3)])
        for key, value in (("session_id", "other"), ("generation", 2), ("applied_bridge_id", "other")):
            state = self.state(3)
            state[key] = value
            self.log.completed(state, 10.07)
        self.assertEqual(self.log.completed_count, 1)
        self.log.completed(self.state(3), 10.08)
        self.assertEqual(self.log.completed_count, 2)

    def test_pending_storage_is_bounded_and_reset_preserves_only_real_samples(self):
        for frame in range(1000):
            self.log.sent("robot", 1, frame, 10.0)
        self.assertEqual(len(self.log.pending), 512)
        self.log.completed(self.state(0), 10.01)
        self.assertEqual(self.log.completed_count, 0)
        self.log.pending.clear()
        self.log.completed(self.state(999), 10.02)
        self.assertEqual(self.log.completed_count, 0)
