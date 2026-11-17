"""Socket buffering is bounded and set before connection; no IK or SDK needed."""

import unittest
from unittest.mock import Mock, call, patch

import tyro
import zmq
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge, main


class ZmqOptionTests(unittest.TestCase):
    def test_feedback_queue_can_change_without_changing_command_queue(self):
        for value in (1, 8, 16):
            with self.subTest(value=value), patch("xrobotoolkit_teleop.integrations.robojudo.ExternalStateRetargeter"):
                context = Mock()
                commands, feedback = Mock(), Mock()
                context.socket.side_effect = [commands, feedback]
                bridge = RoboJuDoBridge(context=context, feedback_receive_hwm=value)
                try:
                    commands.setsockopt.assert_any_call(zmq.SNDHWM, 1)
                    feedback.setsockopt.assert_any_call(zmq.RCVHWM, value)
                    events = feedback.mock_calls
                    self.assertLess(
                        events.index(call.setsockopt(zmq.RCVHWM, value)),
                        events.index(call.connect("tcp://127.0.0.1:8571")),
                    )
                    self.assertFalse(
                        any(c.args and c.args[0] == zmq.CONFLATE for c in feedback.setsockopt.call_args_list)
                    )
                finally:
                    bridge.close()

    def test_invalid_queue_is_rejected_before_resources_are_created(self):
        with patch("xrobotoolkit_teleop.integrations.robojudo.ExternalStateRetargeter") as factory:
            for value in (0, 17, -1, True, 8.0):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "between 1 and 16"):
                    RoboJuDoBridge(feedback_receive_hwm=value)
            factory.assert_not_called()

    def test_cli_passes_the_feedback_queue_option(self):
        bridge = Mock()
        bridge.poll.side_effect = KeyboardInterrupt
        with patch("xrobotoolkit_teleop.integrations.robojudo.RoboJuDoBridge", return_value=bridge) as factory:
            tyro.cli(main, args=["--feedback-receive-hwm", "16"])
        self.assertEqual(factory.call_args.kwargs["feedback_receive_hwm"], 16)
        bridge.close.assert_called_once()
