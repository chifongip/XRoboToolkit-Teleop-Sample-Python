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

    def test_velocity_publisher_uses_the_advertised_port_and_arm_bind_interface(self):
        with (
            patch("xrobotoolkit_teleop.integrations.robojudo.ExternalStateRetargeter"),
            patch("xrobotoolkit_teleop.common.executor_model.model_fingerprint", return_value="test-model"),
        ):
            context = Mock()
            commands, feedback, velocity, replacement = Mock(), Mock(), Mock(), Mock()
            context.socket.side_effect = [commands, feedback, velocity, replacement]
            bridge = RoboJuDoBridge(command_endpoint="tcp://*:8559", context=context)
            self.addCleanup(bridge.close)
            state = {"intent": "HOME"}
            bridge.receive_feedback = Mock(return_value=state)
            bridge._receive_control_feedback()
            self.assertEqual(context.socket.call_count, 2)

            state["xr_velocity"] = {"endpoint": "tcp://xr-host:9058"}
            bridge._receive_control_feedback()
            bridge._receive_control_feedback()
            self.assertEqual(context.socket.call_count, 3)
            velocity.bind.assert_called_once_with("tcp://*:9058")
            velocity.setsockopt.assert_called_once_with(zmq.SNDHWM, 1)
            velocity.send_json.assert_not_called()

            state["xr_velocity"]["endpoint"] = "tcp://xr-host:9059"
            bridge._receive_control_feedback()
            velocity.close.assert_called_once_with(linger=0)
            replacement.bind.assert_called_once_with("tcp://*:9059")
            del state["xr_velocity"]
            bridge._receive_control_feedback()
            replacement.close.assert_called_once_with(linger=0)

    def test_public_options_are_deployment_essentials(self):
        self.assertEqual(
            set(inspect.signature(main).parameters),
            {
                "command_endpoint",
                "feedback_endpoint",
                "posture_endpoint",
                "scale_factor",
                "arm_velocity_limit",
                "return_velocity_limit",
                "robot",
                "ik_backend",
                "tracking_reference",
            },
        )
