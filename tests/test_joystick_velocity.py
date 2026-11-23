import unittest
from unittest.mock import Mock, patch

from xrobotoolkit_teleop.integrations.joystick_velocity import JoystickVelocity


class JoystickVelocityTests(unittest.TestCase):
    def setUp(self):
        self.socket = Mock()
        self.xr = Mock()
        self.xr.get_button_state_by_name.return_value = False
        self.state = {
            "session_id": "robot",
            "generation": 1,
            "xr_velocity": {"commands_map": [[-0.5, 0, 1], [0.5, 0, -0.5], [1, 0, -1]], "deadzone": 0.1},
        }
        self.velocity = JoystickVelocity(self.socket, timeout_s=0.25)

    def sample(self, stamp, now, left=(0, 0), right=(0, 0), *, deadline=float("inf")):
        self.xr.get_timestamp_ns.return_value = stamp
        self.xr.get_joystick_state.side_effect = [left, right]
        with patch("time.monotonic", return_value=now):
            self.velocity.poll(self.xr, self.state, "Y", deadline)

    def assert_velocity(self, forward, left, yaw):
        message = self.socket.send_json.call_args.args[0]
        self.assertEqual(
            message,
            {"linear": {"x": forward, "y": left, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": yaw}},
        )

    def test_startup_requires_neutral_and_maps_both_directions(self):
        self.sample(1, 0.0, (0, 1))
        self.sample(2, 0.02, (0, 1))
        self.socket.send_json.assert_not_called()
        self.sample(3, 0.04, (0.09, 0.09), (0.09, 1))
        self.assert_velocity(0, 0, 0)
        self.sample(4, 0.06, (-0.5, 0.5), (-0.5, 1))
        self.assert_velocity(0.5, 0.25, 0.5)
        self.sample(5, 0.08, (0.5, -0.5), (0.5, -1))
        self.assert_velocity(-0.25, -0.25, -0.5)
        self.sample(6, 0.10)
        self.assert_velocity(0, 0, 0)

    def test_stale_stamp_stops_once_and_requires_neutral(self):
        self.sample(1, 0)
        self.sample(2, 0.02)
        self.sample(3, 0.04, (0, 1))
        self.socket.reset_mock()
        self.sample(3, 0.20, (0, 1))
        self.socket.send_json.assert_not_called()
        self.sample(3, 0.30, (0, 1))
        self.assert_velocity(0, 0, 0)
        self.sample(4, 0.32, (0, 1))
        self.socket.send_json.assert_called_once()
        self.sample(5, 0.34)
        self.sample(6, 0.36, (0, 1))
        self.assert_velocity(1, 0, 0)

    def test_home_and_new_tracking_generation_disarm(self):
        self.sample(1, 0)
        self.sample(2, 0.02)
        self.sample(3, 0.04, (0, 1))
        self.xr.get_button_state_by_name.return_value = True
        self.sample(4, 0.06, (0, 1))
        self.assert_velocity(0, 0, 0)
        self.socket.reset_mock()
        self.xr.get_button_state_by_name.return_value = False
        self.sample(5, 0.08, (0, 1))
        self.socket.send_json.assert_not_called()
        self.sample(6, 0.10)
        self.sample(7, 0.12, (0, 1))
        self.state.update(generation=2, reset_reason="tracking command timeout")
        self.sample(8, 0.14, (0, -1))
        self.assert_velocity(-0.5, 0, 0)
        self.state.update(generation=3, reset_reason="control intent changed")
        self.sample(9, 0.16, (0, 1))
        self.assert_velocity(0, 0, 0)

    def test_torn_sample_is_skipped_and_bad_axes_are_rejected(self):
        self.sample(1, 0)
        self.sample(2, 0.02)
        self.socket.reset_mock()
        self.xr.get_timestamp_ns.side_effect = [3, 4]
        self.sample(3, 0.04, (0, 1))
        self.socket.send_json.assert_not_called()
        self.xr.get_timestamp_ns.side_effect = None
        with self.assertRaises(ValueError):
            self.sample(4, 0.06, (0, float("nan")))
        self.socket.send_json.assert_not_called()

    def test_feedback_deadline_revokes_drive(self):
        self.sample(1, 0)
        self.sample(2, 0.02)
        self.sample(3, 0.04, (0, 1))
        self.sample(3, 0.06, (0, 1), deadline=0.05)
        self.assert_velocity(0, 0, 0)
        self.socket.reset_mock()
        self.sample(4, 0.08, (0, 1))
        self.socket.send_json.assert_not_called()
