"""Robot-independent external-state teleoperation alongside RoboJuDo over ZMQ."""

import json
import logging
import time
import uuid
from contextlib import ExitStack
from numbers import Real

import numpy as np
import tyro
import zmq

from xrobotoolkit_teleop.common.teleop_settings import DEFAULTS, TeleopSettings
from xrobotoolkit_teleop.ik.registry import BackendName
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.integrations.robot_profiles import PROFILES

logger = logging.getLogger(__name__)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


class RoboJuDoBridge:
    def __init__(
        self,
        command_endpoint="tcp://127.0.0.1:8559",
        feedback_endpoint="tcp://127.0.0.1:8571",
        settings=DEFAULTS,
        xr_client=None,
        *,
        robot="x2",
        ik_backend="placo",
        profile=None,
        context=None,
    ):
        self.settings = settings
        self.xr_client = xr_client
        self._resources = ExitStack()
        if xr_client is not None:
            self._resources.callback(xr_client.close)
        try:
            self.retargeter = ExternalStateRetargeter(settings, robot=robot, ik_backend=ik_backend, profile=profile)
            self._resources.callback(self.retargeter.close)
            context = zmq.Context.instance() if context is None else context
            self.commands = context.socket(zmq.PUB)
            self._resources.callback(self.commands.close, linger=0)
            self.feedback = context.socket(zmq.SUB)
            self._resources.callback(self.feedback.close, linger=0)
            self.commands.setsockopt(zmq.LINGER, 0)
            self.commands.setsockopt(zmq.SNDHWM, 1)
            self.commands.bind(command_endpoint)
            self.feedback.setsockopt(zmq.LINGER, 0)
            self.feedback.setsockopt(zmq.RCVHWM, 8)
            self.feedback.setsockopt(zmq.MAXMSGSIZE, 65_536)
            self.feedback.setsockopt(zmq.SUBSCRIBE, b"")
            self.feedback.connect(feedback_endpoint)
        except Exception:
            self.close()
            raise
        self.bridge_id = uuid.uuid4().hex
        self._frame_id = 0
        self._key = None
        self._state_id = -1
        self._last_warning = float("-inf")
        self._feedback_timeout = 0.1
        self._clear_pending()
        self._multipart = False
        self.status = "waiting for robot feedback"
        self.hold_reason = "no fresh robot feedback received"
        self._xr_timestamp = None
        self._xr_changed_at = None
        self._xr_updates = 0
        self._xr_grips = {}
        self._xr_triggers = {}
        self._xr_report_at = time.monotonic()
        self._timing_ms = {}

    def close(self):
        self._resources.close()

    def _clear_pending(self):
        self._pending_baseline = None
        self._pending_positions = None
        self._pending_at = None
        self._pending_deadline = None

    def _rebase(self):
        try:
            self.retargeter.rebase()
        finally:
            self._clear_pending()

    def _warn(self, exc):
        if time.monotonic() - self._last_warning >= 1:
            logger.warning("XR bridge holding: %s", exc)
            self._last_warning = time.monotonic()

    def validate_feedback(self, state):
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != (2 if self.retargeter.hand_names else 1)
            or state.get("type") != "upper_body_state"
        ):
            version = 2 if self.retargeter.hand_names else 1
            raise ValueError(f"expected upper_body_state schema version {version}")
        if not isinstance(state.get("session_id"), str) or not 1 <= len(state["session_id"]) <= 64:
            raise ValueError("invalid robot session")
        for name in ("state_id", "command_id", "generation"):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError(f"invalid {name}")
        if state.get("valid") is not True or state.get("intent") not in {"TRACK", "HOME", "HOLD"}:
            raise ValueError("robot feedback is unavailable or invalid")
        for key, names in (
            ("positions", self.retargeter.joint_names),
            ("joint_age_s", self.retargeter.joint_names),
            ("last_commanded_positions", self.retargeter.command_names),
            ("default_positions", self.retargeter.command_names),
        ):
            if not isinstance(state.get(key), dict) or set(state[key]) != set(names):
                raise ValueError(f"robot feedback has incompatible {key}")
            for value in state[key].values():
                _number(value)
        limits = state.get("position_limits")
        if not isinstance(limits, dict) or set(limits) != set(self.retargeter.command_names):
            raise ValueError("incompatible arm limit names")
        for bounds in limits.values():
            if not isinstance(bounds, list) or len(bounds) != 2 or _number(bounds[0]) >= _number(bounds[1]):
                raise ValueError("invalid arm position limits")
        for key in ("control_dt", "arm_velocity_limit", "return_velocity_limit", "feedback_timeout_s"):
            if _number(state.get(key)) <= 0:
                raise ValueError(f"invalid positive parameter: {key}")
        if self.retargeter.hand_names and _number(state.get("hand_velocity_limit")) <= 0:
            raise ValueError("invalid positive hand_velocity_limit")
        if not all(0 <= age <= state["feedback_timeout_s"] for age in state["joint_age_s"].values()):
            raise ValueError("robot joint measurements are stale")

    def _xr_frame(self):
        if self.xr_client is None:
            # HOME is independent of headset/PC Service availability.
            from xrobotoolkit_teleop.common.xr_client import XrClient

            self.xr_client = XrClient()
            self._resources.callback(self.xr_client.close)
        xr = self.xr_client
        profile = self.retargeter.profile
        devices = {profile.headset, *(group.controller for group in profile.groups.values())}
        frame = XRFrame(
            timestamp_ns=xr.get_timestamp_ns(),
            poses={name: xr.get_pose_by_name(name) for name in devices},
            grips={key: xr.get_key_value_by_name(group.grip) for key, group in profile.groups.items()},
            return_to_default=xr.get_button_state_by_name(profile.home_button),
            triggers={}
            if profile.hands is None
            else {side: xr.get_key_value_by_name(f"{side}_trigger") for side in profile.hands.groups},
        )
        if self._xr_timestamp != frame.timestamp_ns:
            self._xr_changed_at = time.monotonic()
            self._xr_updates += 1
        self._xr_timestamp = frame.timestamp_ns
        self._xr_grips = frame.grips.copy()
        self._xr_triggers = frame.triggers.copy()
        return frame

    def report_xr_diagnostics(self):
        """Report observed SDK updates independently of IK and clutch decisions."""
        now = time.monotonic()
        elapsed = now - self._xr_report_at
        if elapsed < 2.0:
            return
        age = None if self._xr_changed_at is None else round(now - self._xr_changed_at, 3)
        logger.info(
            "XR SDK: timestamp=%s, age_s=%s, observed_updates_hz=%.1f, grips=%s, triggers=%s, robot_intent=%s",
            self._xr_timestamp,
            age,
            self._xr_updates / elapsed,
            self._xr_grips,
            self._xr_triggers,
            getattr(self, "_robot_intent", "no feedback"),
        )
        logger.info("Arm round-trip timing (ms): %s", {key: round(value, 1) for key, value in self._timing_ms.items()})
        self._xr_updates = 0
        self._xr_report_at = now

    def poll(self):
        """Bound receive/compute work to the newest validated robot snapshot."""
        latest = None
        invalid_feedback = False
        for _ in range(32):
            try:
                payload = self.feedback.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            more = bool(self.feedback.getsockopt(zmq.RCVMORE))
            multipart = self._multipart or more
            self._multipart = more
            if multipart:
                continue
            try:
                if len(payload) > 65_536:
                    raise ValueError("feedback exceeds size limit")
                state = json.loads(payload, object_pairs_hook=_unique_object)
                self.validate_feedback(state)
                latest = state
            except (TypeError, ValueError, RecursionError, OverflowError) as exc:
                latest = None
                invalid_feedback = True
                self.status = "hold"
                self.hold_reason = f"invalid robot feedback: {exc}"
                self._warn(exc)
        else:
            self._rebase()
            self.status = "feedback backlog"
            self.hold_reason = "feedback receive budget exhausted; waiting for fresh state"
            return False
        if latest is None:
            if invalid_feedback:
                self._rebase()
                return False
            # The bridge only computes on newly received state. Reset clutches
            # after feedback loss, without requiring packets or synchronized clocks.
            if time.monotonic() - getattr(self, "_received_at", float("-inf")) > self._feedback_timeout:
                self._rebase()
                self.status = "waiting for robot feedback"
                self.hold_reason = "robot feedback missing or timed out"
            return False
        now = time.monotonic()
        key = (latest["session_id"], latest["generation"])
        if key != self._key:
            same_robot = self._key is not None and key[0] == self._key[0]
            if same_robot and latest.get("reset_reason") == "tracking command timeout":
                self._rebase()
            else:
                self.retargeter.reset()
            if self._key is not None:
                logger.info("Robot feedback generation changed: %s", latest.get("reset_reason", "session reset"))
            self._key = key
            self._state_id = -1
            self._clear_pending()
            self._timing_ms.clear()
        if latest["state_id"] <= self._state_id:
            return False
        self._state_id = latest["state_id"]
        self._received_at = now
        self._feedback_timeout = latest["feedback_timeout_s"]
        self._robot_intent = latest["intent"]
        self._timing_ms["feedback_age_at_publish"] = 1000 * max(latest["joint_age_s"].values())
        pipeline_timing = latest.get("pipeline_timing_ms", {})
        for name in ("policy", "env_step"):
            self._timing_ms.pop(name, None)
            if isinstance(pipeline_timing, dict):
                try:
                    value = _number(pipeline_timing.get(name))
                except (TypeError, ValueError, OverflowError):
                    continue
                if value >= 0:
                    self._timing_ms[name] = value
        baseline = (*key, latest["command_id"])
        if self._pending_baseline is not None:
            if baseline == self._pending_baseline and now <= self._pending_deadline:
                return False
            expected = (*self._pending_baseline[:2], self._pending_baseline[2] + 1)
            if (
                baseline == expected
                and latest["last_commanded_positions"] == self._pending_positions
            ):
                try:
                    if not self.retargeter.acknowledge(latest["last_commanded_positions"]):
                        raise RuntimeError("retargeter has no matching pending proposal")
                except (RuntimeError, ValueError, TypeError) as exc:
                    self._rebase()
                    self.status = "hold"
                    self.hold_reason = f"IK acknowledgement failed: {exc}"
                    self._warn(exc)
                    return False
                self._timing_ms["ack_wait"] = 1000 * (now - self._pending_at)
            else:
                self._rebase()
                self._warn(ValueError("arm proposal was not acknowledged; rebasing on fresh feedback"))
        self._clear_pending()
        try:
            retarget_started = time.monotonic()
            frame = self._xr_frame() if latest["intent"] == "TRACK" else None
            proposal = self.retargeter.step(
                latest["positions"],
                latest["last_commanded_positions"],
                latest["default_positions"],
                latest["position_limits"],
                intent=latest["intent"],
                frame=frame,
                control_dt=latest["control_dt"],
                arm_velocity_limit=latest["arm_velocity_limit"],
                return_velocity_limit=latest["return_velocity_limit"],
                hand_velocity_limit=latest.get("hand_velocity_limit"),
            )
            self.status = self.retargeter.status
            self.hold_reason = self.retargeter.hold_reason
            self._timing_ms["retarget"] = 1000 * (time.monotonic() - retarget_started)
            # Remote clocks are never compared. This excludes transit/queueing;
            # the robot performs the authoritative full-age check at application.
            age_lower_bound = max(latest["joint_age_s"].values()) + time.monotonic() - now
            self._timing_ms["measurement_age_lower_bound"] = 1000 * age_lower_bound
            if age_lower_bound > latest["feedback_timeout_s"]:
                self._rebase()
                self.status = "stale robot feedback"
                self.hold_reason = "robot measurements expired during retargeting"
                return False
            visualize = latest.get("visualize_targets") is True
            if proposal is None and not visualize:
                return False
            self._frame_id += 1
            message = {
                "schema_version": 2 if self.retargeter.hand_names else 1,
                "type": "upper_body_proposal" if proposal is not None else "upper_body_visualization",
                "bridge_id": self.bridge_id,
                "session_id": latest["session_id"],
                "generation": latest["generation"],
                "state_id": latest["state_id"],
                "command_id": latest["command_id"],
                "frame_id": self._frame_id,
                "intent": latest["intent"],
                "status": self.status,
                "positions": proposal,
            }
            if visualize:
                message["target_poses"] = self.retargeter.target_poses
            self.commands.send_json(
                message,
                flags=zmq.NOBLOCK,
                allow_nan=False,
            )
            if proposal is None:
                return False
            self._pending_positions = proposal.copy()
            self._pending_baseline = baseline
            self._pending_at = time.monotonic()
            self._pending_deadline = self._pending_at + latest["feedback_timeout_s"]
            return True
        except (RuntimeError, TypeError, ValueError, OverflowError, zmq.ZMQError) as exc:
            self._rebase()
            self.status = "hold"
            self.hold_reason = str(exc) or type(exc).__name__
            self._warn(exc)
            return False


def main(
    command_endpoint: str = "tcp://127.0.0.1:8559",
    feedback_endpoint: str = "tcp://127.0.0.1:8571",
    scale_factor: float = DEFAULTS.scale_factor,
    arm_velocity_limit: float = DEFAULTS.arm_velocity_limit,
    return_velocity_limit: float = DEFAULTS.return_velocity_limit,
    hand_velocity_limit: float = DEFAULTS.hand_velocity_limit,
    collision_margin: float = DEFAULTS.collision_margin,
    collision_trigger: float = DEFAULTS.collision_trigger,
    control_frequency: float = DEFAULTS.control_frequency,
    tracking_timeout_s: float = DEFAULTS.tracking_timeout_s,
    grip_threshold: float = DEFAULTS.grip_threshold,
    xr_diagnostics: bool = False,
    robot: str = "x2",
    ik_backend: BackendName = BackendName.placo,
):
    """Publish checked joint targets from live measured RoboJuDo feedback."""
    logging.basicConfig(level=logging.INFO)
    settings = TeleopSettings(
        scale_factor=scale_factor,
        arm_velocity_limit=arm_velocity_limit,
        return_velocity_limit=return_velocity_limit,
        hand_velocity_limit=hand_velocity_limit,
        collision_margin=collision_margin,
        collision_trigger=collision_trigger,
        control_frequency=control_frequency,
        tracking_timeout_s=tracking_timeout_s,
        grip_threshold=grip_threshold,
    )
    if robot not in PROFILES:
        raise ValueError(f"Unknown robot {robot!r}; choose from {tuple(PROFILES)}")
    bridge = RoboJuDoBridge(command_endpoint, feedback_endpoint, settings, robot=robot, ik_backend=ik_backend)
    logger.info(
        "Robot bridge publishing %s, receiving %s; release all grips before engaging",
        command_endpoint,
        feedback_endpoint,
    )
    try:
        status = None
        while True:
            # Wake on feedback instead of maintaining an unrelated phase that
            # can put each proposal just after the robot's next receive tick.
            bridge.feedback.poll(timeout=max(1, min(100, int(1000 / control_frequency))))
            started = time.monotonic()
            sent = bridge.poll()
            if xr_diagnostics:
                bridge.report_xr_diagnostics()
            current = (bridge.status, bridge.hold_reason)
            if status != current:
                detail = f" ({bridge.hold_reason})" if bridge.hold_reason else ""
                logger.info("Robot bridge: %s%s", bridge.status, detail)
                status = current
            if sent:
                time.sleep(max(0, 1 / control_frequency - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    tyro.cli(main)
