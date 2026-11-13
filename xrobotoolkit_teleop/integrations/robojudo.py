"""Robot-independent external-state teleoperation alongside RoboJuDo over ZMQ."""

import json
import logging
import time
import uuid
from collections import OrderedDict, deque
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
        self._outstanding = OrderedDict()
        self._completion_records = OrderedDict()
        self._latest = None
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
        self._timing_samples = {}
        self._feedback_details = {}
        self._proposal_count = self._completion_count = 0

    def _record_timing(self, name, value):
        self._timing_ms[name] = value
        self._timing_samples.setdefault(name, deque(maxlen=512)).append(value)

    def close(self):
        try:
            if hasattr(self, "_outstanding"):
                self._clear_outstanding()
        finally:
            self._resources.close()

    def _clear_outstanding(self):
        records = list(self._outstanding.values())
        self._outstanding.clear()
        self._completion_records.clear()
        for _, transaction, _, _, _ in records:
            self.retargeter.discard_proposal(transaction)

    def _observe_completion(self, state, now):
        frame_id = state.get("applied_frame_id")
        if state.get("applied_bridge_id") != self.bridge_id or type(frame_id) is not int:
            return
        record = self._completion_records.pop(frame_id, None)
        if record is None:
            return
        positions, sent_at, baseline = record
        transaction_record = self._outstanding.pop(frame_id, None)
        expected = (*baseline[:2], baseline[2] + 1)
        current = (state["session_id"], state["generation"], state["command_id"])
        matched = current == expected and positions == state["last_commanded_positions"]
        if matched:
            # Observe hardware completion even if its solver transaction expired.
            self._record_timing("completion_latency", 1000 * (now - sent_at))
            self._completion_count += 1
        try:
            if transaction_record is not None:
                transaction = transaction_record[1]
                if matched:
                    self.retargeter.complete_proposal(transaction, positions)
                else:
                    self.retargeter.discard_proposal(transaction)
        except (RuntimeError, ValueError, TypeError) as exc:
            # Completion bookkeeping never gates the publishing schedule.
            self._warn(exc)
            self.retargeter.backend.reset()
            self._clear_outstanding()
        # Earlier proposals can no longer be selected after this completion.
        for older in list(self._outstanding):
            if older < frame_id:
                self.retargeter.discard_proposal(self._outstanding.pop(older)[1])
        for older in list(self._completion_records):
            if older < frame_id:
                self._completion_records.pop(older)

    def _expire_outstanding(self, now):
        for frame_id, record in list(self._outstanding.items()):
            if now > record[3]:
                self.retargeter.discard_proposal(self._outstanding.pop(frame_id)[1])
        while len(self._outstanding) > 512:
            self.retargeter.discard_proposal(self._outstanding.popitem(last=False)[1][1])
        # Retain only lightweight completion metadata beyond the solver deadline.
        while len(self._completion_records) > 512:
            self._completion_records.popitem(last=False)

    def _rebase(self):
        try:
            self.retargeter.rebase()
        finally:
            self._clear_outstanding()

    def _warn(self, exc):
        if time.monotonic() - self._last_warning >= 1:
            logger.warning("XR bridge: %s", exc)
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
        age = "unknown" if self._xr_changed_at is None else f"{1000 * (now - self._xr_changed_at):.0f} ms"

        def inputs(values):
            result = []
            for side in ("left", "right"):
                try:
                    result.append(f"{_number(values.get(side)):.2f}")
                except (TypeError, ValueError, OverflowError):
                    result.append("?")
            return "/".join(result)

        logger.info(
            "XR: %.1f Hz, age=%s, intent=%s; grips=%s, triggers=%s (L/R)",
            self._xr_updates / elapsed,
            age,
            getattr(self, "_robot_intent", "no feedback"),
            inputs(self._xr_grips),
            inputs(self._xr_triggers),
        )
        oldest = self._feedback_details.get("oldest_joint")
        source = "unknown"
        if isinstance(oldest, str):
            source = "left hand" if oldest.startswith("L_") else "right hand" if oldest.startswith("R_") else "body"
        samples = self._timing_samples.get("feedback_age_at_publish")
        if self._feedback_details.get("valid") is False:
            summary = f"unavailable ({source})"
        elif samples:
            summary = f"age={samples[-1]:.1f} ms ({source}), p95={np.percentile(samples, 95):.1f} ms"
        else:
            summary = "waiting for measurements"
        counts = self._feedback_details.get("command_counts", {})
        counts = counts if isinstance(counts, dict) else {}
        receiver_summary = ", ".join(
            f"{name}={counts[name]}"
            for name in ("received", "superseded", "rejected", "completed")
            if type(counts.get(name)) is int and counts[name] >= 0
        )
        logger.info(
            "Feedback: %s; sent=%d, completed=%d; publish=%.1f Hz%s",
            summary,
            self._proposal_count,
            self._completion_count,
            self._proposal_count / elapsed,
            f"; robot {receiver_summary}" if receiver_summary else "",
        )
        stages = []
        for label, name in (
            ("policy", "policy"),
            ("env_step", "env_step"),
            ("retarget", "retarget"),
            ("xr_sample", "xr_sample"),
            ("ik", "ik"),
            ("completion_latency", "completion_latency"),
        ):
            values = self._timing_samples.get(name)
            if values:
                stages.append(f"{label}={np.percentile(values, 95):.1f}")
        if stages:
            logger.info("Processing p95 (ms): %s", ", ".join(stages))
        if self._timing_samples.get("ik_calls"):
            calls = sum(self._timing_samples["ik_calls"])
            tracked = sum(self._timing_samples.get("ik_tracked_calls", ()))
            warm = sum(self._timing_samples.get("ik_warm_calls", ()))
            iterations = self._timing_samples.get("ik_iterations")
            logger.info(
                "IK: calls=%d, warm=%d/%d%s",
                calls,
                warm,
                tracked,
                f"; iterations per step p95={np.percentile(iterations, 95):.1f}" if iterations else "",
            )
        self._timing_samples.clear()
        self._proposal_count = self._completion_count = 0
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
                if isinstance(state, dict) and state.get("type") == "upper_body_state":
                    details = state.get("feedback_diagnostics", {})
                    self._feedback_details = {
                        **(details if isinstance(details, dict) else {}),
                        "state_id": state.get("state_id"),
                        "valid": state.get("valid"),
                    }
                self.validate_feedback(state)
                latest = state
            except (TypeError, ValueError, RecursionError, OverflowError) as exc:
                latest = None
                invalid_feedback = True
                self.status = "hold"
                self.hold_reason = f"invalid robot feedback: {exc}"
                self._warn(exc)
        else:
            self._latest = None
            self._rebase()
            self.status = "feedback backlog"
            self.hold_reason = "feedback receive budget exhausted; waiting for fresh state"
            return False
        if latest is None and invalid_feedback:
            self._latest = None
            self._rebase()
            return False
        now = time.monotonic()
        if latest is not None and (
            (latest["session_id"], latest["generation"]) != self._key or latest["state_id"] > self._state_id
        ):
            self._latest = latest
            self._received_at = now
        else:
            latest = self._latest
        if latest is None or now - self._received_at > self._feedback_timeout:
            self._rebase()
            self.status = "waiting for robot feedback"
            self.hold_reason = "robot feedback missing or timed out"
            return False
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
            self._clear_outstanding()
            self._timing_ms.clear()
        self._state_id = latest["state_id"]
        self._feedback_timeout = latest["feedback_timeout_s"]
        self._robot_intent = latest["intent"]
        self._record_timing("feedback_age_at_publish", 1000 * max(latest["joint_age_s"].values()))
        sources = self._feedback_details.get("source_age_ms", {})
        sources = sources if isinstance(sources, dict) else {}
        extra_timing = {
            f"{source}_measurement_age": sources.get(source) for source in ("body", "left_hand", "right_hand")
        }
        for name in ("handoff_delay_ms", "snapshot_cache_age_ms"):
            extra_timing[f"hand_{name.removesuffix('_ms')}"] = self._feedback_details.get(name)
        previous = self._feedback_details.get("previous_publication")
        previous = previous if isinstance(previous, dict) else {}
        for name in ("build_ms", "encode_send_ms"):
            extra_timing[f"previous_feedback_{name.removesuffix('_ms')}"] = previous.get(name)
        for name, value in extra_timing.items():
            self._timing_ms.pop(name, None)
            try:
                value = _number(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if value >= 0:
                self._record_timing(name, value)
        pipeline_timing = latest.get("pipeline_timing_ms", {})
        for name in ("policy", "env_step", "body_snapshot", "hand_snapshot", "feedback_assembly"):
            self._timing_ms.pop(name, None)
            if isinstance(pipeline_timing, dict):
                try:
                    value = _number(pipeline_timing.get(name))
                except (TypeError, ValueError, OverflowError):
                    continue
                if value >= 0:
                    self._record_timing(name, value)
        baseline = (*key, latest["command_id"])
        self._expire_outstanding(now)
        self._observe_completion(latest, now)
        try:
            retarget_started = time.monotonic()
            frame = self._xr_frame() if latest["intent"] == "TRACK" else None
            self._record_timing("xr_sample", 1000 * (time.monotonic() - retarget_started))
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
                streaming=True,
            )
            self.status = self.retargeter.status
            self.hold_reason = self.retargeter.hold_reason
            self._record_timing("retarget", 1000 * (time.monotonic() - retarget_started))
            diagnostics = getattr(self.retargeter, "solver_diagnostics", {})
            if isinstance(diagnostics, dict) and diagnostics.get("calls"):
                self._record_timing("ik", diagnostics["ik_ms"])
                self._record_timing("ik_calls", diagnostics["calls"])
                for key in ("tracked_calls", "warm_calls", "iterations"):
                    if key in diagnostics:
                        self._record_timing(f"ik_{key}", diagnostics[key])
            # Remote clocks are never compared. This excludes transit/queueing;
            # the robot performs the authoritative full-age check at application.
            age_lower_bound = max(latest["joint_age_s"].values()) + time.monotonic() - self._received_at
            self._record_timing("measurement_age_lower_bound", 1000 * age_lower_bound)
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
            sent_at = time.monotonic()
            self._completion_records[self._frame_id] = (proposal.copy(), sent_at, baseline)
            self._outstanding[self._frame_id] = (
                proposal.copy(),
                self.retargeter.detach_proposal(),
                sent_at,
                self._received_at + latest["feedback_timeout_s"],
                baseline,
            )
            self._expire_outstanding(time.monotonic())
            self._proposal_count += 1
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
    collision_checking: tyro.conf.FlagCreatePairsOff[bool] = False,
):
    """Publish bounded joint targets from live measured RoboJuDo feedback.

    Args:
        collision_checking: Enable collision avoidance and arm/hand path checks; omitted means disabled.
    """
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
        collision_checking=collision_checking,
    )
    if robot not in PROFILES:
        raise ValueError(f"Unknown robot {robot!r}; choose from {tuple(PROFILES)}")
    bridge = RoboJuDoBridge(command_endpoint, feedback_endpoint, settings, robot=robot, ik_backend=ik_backend)
    logger.info("XR collision checking: %s", "enabled" if settings.collision_checking else "disabled")
    logger.info(
        "Robot bridge publishing %s, receiving %s; release all grips before engaging",
        command_endpoint,
        feedback_endpoint,
    )
    try:
        status = None
        period = 1 / control_frequency
        deadline = time.monotonic()
        while True:
            time.sleep(max(0, deadline - time.monotonic()))
            bridge.poll()
            if xr_diagnostics:
                bridge.report_xr_diagnostics()
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
