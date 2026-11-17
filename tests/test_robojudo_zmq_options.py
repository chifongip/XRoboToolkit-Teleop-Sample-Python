"""The release bridge uses bounded feedback buffering."""

import inspect
import unittest
from unittest.mock import Mock, patch

import zmq
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge, main


class ZmqOptionTests(unittest.TestCase):
    def test_fixed_queue_limits(self):
        with (
            patch("xrobotoolkit_teleop.integrations.robojudo.ExternalStateRetargeter"),
            patch("xrobotoolkit_teleop.common.executor_model.model_fingerprint", return_value="test-model"),
        ):
            context = Mock()
            commands, feedback = Mock(), Mock()
            context.socket.side_effect = [commands, feedback]
            bridge = RoboJuDoBridge(context=context)
            self.addCleanup(bridge.close)
            commands.setsockopt.assert_any_call(zmq.SNDHWM, 1)
            feedback.setsockopt.assert_any_call(zmq.RCVHWM, 8)

    def test_public_options_are_deployment_essentials(self):
        self.assertEqual(
            set(inspect.signature(main).parameters),
            {
                "command_endpoint",
                "feedback_endpoint",
                "scale_factor",
                "arm_velocity_limit",
                "return_velocity_limit",
                "robot",
                "ik_backend",
            },
        )
