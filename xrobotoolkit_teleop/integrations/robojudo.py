"""Robot-independent external-state teleoperation alongside RoboJuDo over ZMQ."""

import json
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import replace
from numbers import Real
from typing import Literal

import numpy as np
import tyro
import zmq

from xrobotoolkit_teleop.common.teleop_settings import DEFAULTS, TeleopSettings, TrackingReference
from xrobotoolkit_teleop.common.waist_tracking import read_waist_sample
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


class _CommandLog:
    """Bounded command correlation using only the bridge's monotonic clock."""

    def __init__(self, bridge_id):
        self.bridge_id = bridge_id
        self.pending = OrderedDict()
        self.started = time.monotonic()
        self.sent_count = self.completed_count = 0
        self.total_ms = self.max_ms = 0.0

    def sent(self, session, generation, frame_id, sent_at):
        self.sent_count += 1
        self.pending[session, generation, frame_id] = sent_at
        while len(self.pending) > 512:
            self.pending.popitem(last=False)

    def completed(self, state, received_at):
        frame_id = state.get("applied_frame_id")
        if state.get("applied_bridge_id") != self.bridge_id or type(frame_id) is not int:
            return
        sent_at = self.pending.pop((state["session_id"], state["generation"], frame_id), None)
        if sent_at is None:
            return
        while self.pending and next(iter(self.pending))[2] < frame_id:
            self.pending.popitem(last=False)
        latency_ms = 1000 * (received_at - sent_at)
        if latency_ms >= 0:
            self.completed_count += 1
            self.total_ms += latency_ms
            self.max_ms = max(self.max_ms, latency_ms)

    def report(self, status):
        now = time.monotonic()
        elapsed = now - self.started
        if elapsed < 1.0:
            return
        rtt = (
            f"{self.total_ms / self.completed_count:.1f} ms avg / {self.max_ms:.1f} ms max"
            if self.completed_count
            else "n/a"
        )
        logger.info("XR: send=%.1f Hz | completion RTT=%s | %s", self.sent_count / elapsed, rtt, status)
        self.started = now
        self.sent_count = self.completed_count = 0
        self.total_ms = self.max_ms = 0.0


class RoboJuDoBridge:
    """Solve desired XR targets; RoboJuDo owns executable transitions and completion."""

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
        tracking_reference: TrackingReference = "fixed-heading",
        context=None,
    ):
        from xrobotoolkit_teleop.common.executor_model import CONTRACT, model_fingerprint

        self._resources = ExitStack()
        self.xr_client = xr_client
        self.robot_name = robot
        self.settings = replace(settings, collision_checking=False)
        self.executor_contract = None
        self._policy_session = None
        self.bridge_id = uuid.uuid4().hex
        self._frame_id = 0
        self._key = None
        self._state_id = -1
        self._latest = None
        self._multipart = False
        self._feedback_timeout = 0.1
        self._last_warning = float("-inf")
        self.status = "waiting for robot feedback"
        self.hold_reason = "no fresh robot feedback received"
        if xr_client is not None:
            self._resources.callback(xr_client.close)
        try:
            self.retargeter = ExternalStateRetargeter(
                self.settings,
                robot=robot,
                ik_backend=ik_backend,
                profile=profile,
                tracking_reference=tracking_reference,
            )
            self._resources.callback(self.retargeter.close)
            self._model_identity = {
                "contract": CONTRACT,
                "robot_profile": robot,
                "model_fingerprint": model_fingerprint(self.retargeter.profile),
            }
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
        self._command_log = _CommandLog(self.bridge_id)

    def close(self):
        self._resources.close()

    def _warn(self, exc):
        if time.monotonic() - self._last_warning >= 1:
            logger.warning("XR bridge: %s", exc)
            self._last_warning = time.monotonic()

    def _hold(self, reason):
        self._latest = None
        self.retargeter.rebase()
        self.status, self.hold_reason = "hold", reason
        return False

    def validate_feedback(self, state):
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != 3
            or state.get("type") != "upper_body_state"
        ):
            raise ValueError("expected upper_body_state schema version 3")
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
            raise ValueError("incompatible command limit names")
        for bounds in limits.values():
            if not isinstance(bounds, list) or len(bounds) != 2 or _number(bounds[0]) >= _number(bounds[1]):
                raise ValueError("invalid position limits")
        for key in ("control_dt", "arm_velocity_limit", "return_velocity_limit", "feedback_timeout_s"):
            if _number(state.get(key)) <= 0:
                raise ValueError(f"invalid positive parameter: {key}")
        if not all(0 <= age <= state["feedback_timeout_s"] for age in state["joint_age_s"].values()):
            raise ValueError("robot joint measurements are stale")
        contract = state.get("executor")
        policy_keys = {"collision_checking", "collision_margin", "collision_trigger"}
        if self.retargeter.hand_names:
            policy_keys.update(("hand_command_mode", "hand_velocity_limit"))
        if (
            not isinstance(contract, dict)
            or set(contract) != set(self._model_identity) | policy_keys
            or any(contract.get(k) != v for k, v in self._model_identity.items())
        ):
            raise ValueError("incompatible executor model or robot profile")
        if type(contract["collision_checking"]) is not bool:
            raise ValueError("invalid collision enablement")
        if not 0 < _number(contract["collision_margin"]) < _number(contract["collision_trigger"]):
            raise ValueError("invalid collision policy")
        if self.retargeter.hand_names:
            if contract["hand_command_mode"] not in ("filtered", "direct"):
                raise ValueError("invalid finger control mode")
            if _number(contract["hand_velocity_limit"]) <= 0:
                raise ValueError("invalid finger velocity limit")
        if self._policy_session == state["session_id"] and contract != self.executor_contract:
            raise ValueError("robot collision policy or finger settings changed within a session")

    def _adopt_policy(self, state):
        if self._policy_session == state["session_id"]:
            return
        contract = state["executor"]
        settings = replace(
            self.settings,
            **{
                k: contract[k]
                for k in (
                    "collision_checking",
                    "collision_margin",
                    "collision_trigger",
                )
            },
        )
        self.retargeter.configure_collision_policy(settings)
        self.settings = settings
        self.executor_contract = dict(contract)
        self._policy_session = state["session_id"]
        logger.info(
            "Robot XR profile %s; collision checking %s",
            self.robot_name,
            "enabled" if settings.collision_checking else "disabled",
        )

    def _ensure_xr(self):
        if self.xr_client is None:
            # Robot-requested HOME bypasses XR sampling.
            from xrobotoolkit_teleop.common.xr_client import XrClient

            self.xr_client = XrClient()
            self._resources.callback(self.xr_client.close)

    def _xr_frame(self):
        self._ensure_xr()
        xr = self.xr_client
        profile = self.retargeter.profile
        if xr.get_button_state_by_name(profile.home_button):
            return XRFrame(timestamp_ns=xr.get_timestamp_ns(), poses={}, grips={}, return_to_default=True)
        waist_mode = self.retargeter.tracking_reference == "waist"
        waist, waist_error = read_waist_sample(xr) if waist_mode else (None, None)
        devices = set() if waist_mode else {profile.headset, *(group.controller for group in profile.groups.values())}
        return XRFrame(
            timestamp_ns=waist.xr_timestamp_ns if waist is not None else xr.get_timestamp_ns(),
            waist=waist,
            waist_error=waist_error,
            poses={name: xr.get_pose_by_name(name) for name in devices},
            grips={key: xr.get_key_value_by_name(group.grip) for key, group in profile.groups.items()},
            triggers=(
                {}
                if profile.hands is None
                else {side: xr.get_key_value_by_name(f"{side}_trigger") for side in profile.hands.groups}
            ),
        )

    def poll(self):
        try:
            latest = self.receive_feedback()
            return self._publish_desired(latest) if latest else False
        finally:
            self._command_log.report(self.status)

    def wait_until_publish(self, deadline):
        """Drain feedback between ticks without sampling XR or running IK."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self.feedback.poll(timeout=max(1, int(remaining * 1000)), flags=zmq.POLLIN):
                self.receive_feedback()

    def receive_feedback(self):
        latest, received_at = None, None
        invalid = False
        for _ in range(32):
            try:
                payload = self.feedback.recv(flags=zmq.NOBLOCK)
                packet_at = time.monotonic()
            except zmq.Again:
                break
            more = bool(self.feedback.getsockopt(zmq.RCVMORE))
            multipart, self._multipart = self._multipart or more, more
            if multipart:
                continue
            try:
                if len(payload) > 65_536:
                    raise ValueError("feedback exceeds size limit")
                state = json.loads(payload, object_pairs_hook=_unique_object)
                self.validate_feedback(state)
                if (state["session_id"], state["generation"]) != self._key:
                    self._command_log.pending.clear()
                else:
                    self._command_log.completed(state, packet_at)
                latest, received_at = state, packet_at
            except (TypeError, ValueError, RecursionError, OverflowError) as exc:
                latest = None
                invalid = True
                self._warn(exc)
                self.hold_reason = f"invalid robot feedback: {exc}"
        else:
            return self._hold("feedback receive budget exhausted; waiting for fresh state")
        if latest is None and invalid:
            return self._hold(self.hold_reason)
        if latest is not None and (
            (latest["session_id"], latest["generation"]) != self._key or latest["state_id"] > self._state_id
        ):
            try:
                self._adopt_policy(latest)
            except (RuntimeError, ValueError, ImportError, OSError) as exc:
                self._warn(exc)
                return self._hold(f"collision policy initialization failed: {exc}")
            self._latest, self._received_at = latest, received_at
        latest = self._latest
        now = time.monotonic()
        if (
            latest is None
            or now - self._received_at + max(latest["joint_age_s"].values()) > latest["feedback_timeout_s"]
        ):
            return self._hold("robot feedback missing or timed out")
        key = (latest["session_id"], latest["generation"])
        if key != self._key:
            same_robot = self._key is not None and key[0] == self._key[0]
            if same_robot and latest.get("reset_reason") == "tracking command timeout":
                self.retargeter.rebase()
            else:
                self.retargeter.reset()
            self._key = key
        self._state_id = latest["state_id"]
        self._feedback_timeout = latest["feedback_timeout_s"]
        return latest

    def _publish_desired(self, latest):
        try:
            frame = self._xr_frame() if latest["intent"] == "TRACK" else None
            desired = self.retargeter.step(
                latest["positions"],
                latest["last_commanded_positions"],
                latest["default_positions"],
                latest["position_limits"],
                intent=latest["intent"],
                frame=frame,
                control_dt=latest["control_dt"],
                desired_only=True,
                streaming=True,
                authoritative_executor=True,
                arm_velocity_limit=latest["arm_velocity_limit"],
                return_velocity_limit=latest["return_velocity_limit"],
            )
            self.status, self.hold_reason = self.retargeter.status, self.retargeter.hold_reason
            age = max(latest["joint_age_s"].values()) + time.monotonic() - self._received_at
            if age > latest["feedback_timeout_s"]:
                return self._hold("robot measurements expired during retargeting")
            visualize = latest.get("visualize_targets") is True
            if desired is None and not visualize:
                return False
            active = [
                n for side, names in self.retargeter.arm_joints.items() if self.retargeter.active[side] for n in names
            ]
            hands = self.retargeter.profile.hands
            if hands is not None and self.status == "tracking" and frame is not None:
                for side, names in hands.groups.items():
                    trigger = frame.triggers.get(side)
                    if not isinstance(trigger, bool) and isinstance(trigger, Real) and np.isfinite(trigger):
                        active.extend(names)
            self._frame_id += 1
            message = {
                "schema_version": 3,
                "type": "upper_body_desired" if desired is not None else "upper_body_visualization",
                "bridge_id": self.bridge_id,
                "session_id": latest["session_id"],
                "generation": latest["generation"],
                "state_id": latest["state_id"],
                "command_id": latest["command_id"],
                "frame_id": self._frame_id,
                "intent": latest["intent"],
                "status": self.status,
                "positions": desired,
                "active_joint_names": active,
                "executor": self.executor_contract,
                "velocity_limits": {
                    "arm": self.settings.arm_velocity_limit,
                    "return": self.settings.return_velocity_limit,
                },
            }
            if visualize:
                message["target_poses"] = self.retargeter.target_poses
            sent_at = time.monotonic()
            self.commands.send_json(message, flags=zmq.NOBLOCK, allow_nan=False)
            if desired is None:
                return False
            self._command_log.sent(latest["session_id"], latest["generation"], self._frame_id, sent_at)
            if not self.retargeter.acknowledge(desired):
                return self._hold("desired IK history could not be committed")
            return True
        except (RuntimeError, TypeError, ValueError, OverflowError, zmq.ZMQError) as exc:
            self._warn(exc)
            return self._hold(str(exc) or type(exc).__name__)


def main(
    command_endpoint: str = "tcp://127.0.0.1:8559",
    feedback_endpoint: str = "tcp://127.0.0.1:8571",
    scale_factor: float = DEFAULTS.scale_factor,
    arm_velocity_limit: float = DEFAULTS.arm_velocity_limit,
    return_velocity_limit: float = DEFAULTS.return_velocity_limit,
    robot: str = "x2",
    ik_backend: BackendName = BackendName.placo,
    tracking_reference: Literal["fixed-heading", "waist", "pelvis-ik"] = "fixed-heading",
    posture_endpoint: str = "tcp://127.0.0.1:8557",
):
    """Publish bounded joint targets from live measured RoboJuDo feedback."""
    logging.basicConfig(level=logging.INFO)
    settings = TeleopSettings(
        scale_factor=scale_factor,
        arm_velocity_limit=arm_velocity_limit,
        return_velocity_limit=return_velocity_limit,
    )
    if robot not in PROFILES:
        raise ValueError(f"Unknown robot {robot!r}; choose from {tuple(PROFILES)}")
    if tracking_reference == "pelvis-ik":
        from xrobotoolkit_teleop.integrations.pelvis_ik import PelvisIKBridge

        bridge = PelvisIKBridge(
            command_endpoint,
            feedback_endpoint,
            settings,
            robot=robot,
            ik_backend=ik_backend,
            posture_endpoint=posture_endpoint,
        )
    else:
        bridge = RoboJuDoBridge(
            command_endpoint,
            feedback_endpoint,
            settings,
            robot=robot,
            ik_backend=ik_backend,
            tracking_reference=tracking_reference,
        )
    logger.info("Tracking reference: %s", tracking_reference)
    logger.info(
        "Robot bridge publishing %s, receiving %s; release all grips before engaging",
        command_endpoint,
        feedback_endpoint,
    )
    try:
        status = None
        period = 1 / settings.control_frequency
        deadline = time.monotonic()
        while True:
            bridge.wait_until_publish(deadline)
            bridge.poll()
            current = (bridge.status, bridge.hold_reason)
            if status != current:
                detail = f" ({bridge.hold_reason})" if bridge.hold_reason else ""
                logger.info("Robot bridge: %s%s", bridge.status, detail)
                status = current
            deadline += period
            now = time.monotonic()
            if deadline < now:
                deadline += (int((now - deadline) / period) + 1) * period
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    tyro.cli(main)
