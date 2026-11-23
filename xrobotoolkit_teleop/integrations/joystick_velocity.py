"""Map fresh XR thumbstick samples to RoboJuDo's Twist input."""

import math
import time
from numbers import Real

import zmq

ZERO = (0.0, 0.0, 0.0)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("expected a finite joystick or mapping value")
    return float(value)


def _axis_pair(values):
    x, y = map(_finite, values)
    if max(abs(x), abs(y)) > 1:
        raise ValueError("XR stick axes must be within [-1, 1]")
    return x, y


def _velocity(axes, settings):
    deadzone = _finite(settings["deadzone"])
    rows = settings["commands_map"]
    if not 0 <= deadzone < 1 or len(rows) != 3:
        raise ValueError("invalid XR joystick mapping")
    axes = tuple(axis if abs(axis) >= deadzone else 0.0 for axis in axes)
    result = []
    for axis, row in zip(axes, rows, strict=True):
        negative, neutral, positive = map(_finite, row)
        endpoint = negative if axis < 0 else positive
        result.append(neutral + abs(axis) * (endpoint - neutral))
    neutral = not any(axes)
    return (ZERO if neutral else tuple(result)), neutral


class JoystickVelocity:
    def __init__(self, socket, timeout_s):
        self.socket = socket
        self.timeout_s = timeout_s
        self._session = None
        self._stamp = None
        self._observed_at = None
        self._armed = False
        self._moving = False

    def _send(self, velocity):
        forward, left, yaw = velocity
        self.socket.send_json(
            {"linear": {"x": forward, "y": left, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": yaw}},
            flags=zmq.NOBLOCK,
            allow_nan=False,
        )

    def stop(self):
        self._armed = False
        if self._moving:
            self._moving = False
            try:
                self._send(ZERO)
            except zmq.ZMQError:
                # The subscriber's receive timeout also covers a failed stop packet.
                pass

    def close(self):
        self.stop()
        self.socket.close(linger=0)

    def poll(self, xr, state, home_button, feedback_deadline):
        now = time.monotonic()
        if now >= feedback_deadline:
            self.stop()
            return
        session = (state["session_id"], state["generation"])
        if session != self._session:
            # Arm-only lease expiry does not change locomotion authorization.
            if (
                self._session is None
                or session[0] != self._session[0]
                or state.get("reset_reason") != "tracking command timeout"
            ):
                self.stop()
            self._session = session
        if self._observed_at is not None and now - self._observed_at >= self.timeout_s:
            self.stop()

        stamp = xr.get_timestamp_ns()
        if type(stamp) is not int or stamp <= 0:
            raise ValueError("invalid XR joystick timestamp")
        if xr.get_button_state_by_name(home_button):
            self.stop()
            self._stamp = stamp
            return
        if self._stamp is None or stamp < self._stamp:
            self.stop()
            self._stamp = stamp
            return
        if stamp == self._stamp:
            return

        left_x, left_y = _axis_pair(xr.get_joystick_state("left"))
        right_x, _ = _axis_pair(xr.get_joystick_state("right"))
        if xr.get_timestamp_ns() != stamp:
            return
        self._stamp = stamp
        velocity, neutral = _velocity((left_y, left_x, right_x), state["xr_velocity"])
        sampled_at = time.monotonic()
        if sampled_at >= feedback_deadline or sampled_at - now >= self.timeout_s:
            self.stop()
            return
        self._observed_at = sampled_at
        if not self._armed:
            self._armed = neutral
        if self._armed:
            self._send(velocity)
            self._moving = velocity != ZERO
