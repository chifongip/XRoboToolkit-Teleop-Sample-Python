import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import Mock, patch

from xrobotoolkit_teleop.integrations import xr_receive_rate
from xrobotoolkit_teleop.integrations.xr_receive_rate import ReceiveRate


class ReceiveRateTests(unittest.TestCase):
    def test_stationary_twenty_hz_stream_has_no_timeouts(self):
        probe = ReceiveRate()
        for index in range(101):
            probe.observe(1 + (index // 10) * 50_000_000, index * 0.005)
        self.assertEqual(probe.updates, 10)
        self.assertEqual(probe.timeout_episodes, 0)
        self.assertIn("20.0 Hz", probe.report(0.5))
        self.assertAlmostEqual(probe.max_poll_gap_s, 0.005)
        self.assertAlmostEqual(probe.max_source_gap_s, 0.05)

    def test_timestamp_stall_counts_one_episode_until_recovery(self):
        probe = ReceiveRate()
        probe.observe(100, 0.0)
        for index in range(1, 81):
            probe.observe(100, index * 0.005)
        self.assertEqual(probe.timeout_episodes, 1)
        self.assertTrue(probe.timed_out)
        probe.observe(101, 0.405)
        self.assertFalse(probe.timed_out)
        self.assertAlmostEqual(max(probe.update_gaps), 0.405)

    def test_polling_stall_is_visible_even_if_timestamp_advanced(self):
        probe = ReceiveRate()
        probe.observe(100, 0.0)
        probe.observe(200, 0.4)
        self.assertAlmostEqual(probe.max_poll_gap_s, 0.4)
        self.assertEqual(probe.timeout_episodes, 0)
        self.assertIn("max poll gap=400.0 ms", probe.report(0.4))

    def test_invalid_startup_and_backwards_timestamps_are_separate(self):
        probe = ReceiveRate()
        probe.observe(0, 0.0)
        probe.observe(None, 1.0)
        self.assertEqual(probe.timeout_episodes, 0)
        probe.observe(100, 2.0)
        probe.observe(50, 2.01)
        self.assertEqual(probe.invalid_samples, 2)
        self.assertEqual(probe.timestamp_resets, 1)
        self.assertEqual(probe.updates, 0)

    def test_long_capture_bounds_percentile_history_and_keeps_lifetime_maximum(self):
        probe = ReceiveRate()
        probe.observe(1, 0.0)
        probe.observe(2, 1.0)
        for index in range(1, 5001):
            probe.observe(index + 2, 1.0 + index * 0.005)
        self.assertEqual(len(probe.update_gaps), 4096)
        self.assertAlmostEqual(probe.max_update_gap_s, 1.0)
        self.assertIn("/1000.0 ms", probe.report(26.0))

    def test_csv_open_failure_still_closes_sdk_client(self):
        client = Mock()
        sdk_module = ModuleType("xrobotoolkit_teleop.common.xr_client")
        sdk_module.XrClient = lambda: client
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "capture.csv"
            with (
                patch.dict(sys.modules, {sdk_module.__name__: sdk_module}),
                patch.object(sys, "argv", ["xr_receive_rate", "--csv", str(path)]),
                self.assertRaises(FileNotFoundError),
            ):
                xr_receive_rate.main()
        client.close.assert_called_once_with()
