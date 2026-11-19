"""Pelvis-relative X2 tracking and waist-assisted IK for the RoboJuDo bridge."""

import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
import zmq

from xrobotoolkit_teleop.common.waist_tracking import WaistTracking
from xrobotoolkit_teleop.ik.lerobot_solver import LeRobotIKSolver
from xrobotoolkit_teleop.ik.types import IKProposal
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge, _number
from xrobotoolkit_teleop.utils.geometry import relative_target, xr_pose_to_matrix

UP = np.array([0.0, 1.0, 0.0])
XR_CONTROL_AXES = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
PELVIS_FRAME = "pelvis"
WAIST_YAW = "waist_yaw_joint"
POSITION_TOLERANCE = 0.02
ROTATION_TOLERANCE = math.radians(10)
HOME_ARM_TOLERANCE_RAD = 0.03
logger = logging.getLogger(__name__)


class HandGoals:
    """Hold clutch goals in the selected robot reference and return pelvis-relative IK goals."""

    def __init__(self, retargeter):
        self.robot = retargeter.robot
        self.groups = retargeter.profile.groups
        self.scale = retargeter.settings.scale_factor
        self.threshold = retargeter.settings.grip_threshold
        self.reset()

    def reset(self):
        self.goals = {}
        self.references = {}

    def eligible_grips(self, grips, *, can_capture):
        return {side: grip if can_capture or side in self.references else 0.0 for side, grip in grips.items()}

    def update(self, grips, controllers, measured):
        for name, value in measured.items():
            self.robot.set_joint(name, value)
        self.robot.update_kinematics()
        pelvis = self.robot.get_T_world_frame(PELVIS_FRAME)
        for side, group in self.groups.items():
            if grips[side] <= self.threshold:
                self.goals.pop(side, None)
                self.references.pop(side, None)
                continue
            controller = controllers[side]
            if side not in self.references:
                if side not in self.goals:
                    self.goals[side] = np.linalg.solve(pelvis, self.robot.get_T_world_frame(group.frame))
                self.references[side] = (controller, self.goals[side])
                continue
            captured_controller, hand = self.references[side]
            self.goals[side] = relative_target(controller, captured_controller, hand, XR_CONTROL_AXES, self.scale)
        return self.goals.copy()


class PelvisGoalTasks:
    """Adapt the retargeter's relative tracking interface to prepared hand goals."""

    def __init__(self, retargeter):
        self.robot = retargeter.robot
        self.torso = retargeter.profile.reference_frame
        self.clear()

    def clear(self):
        self.valid = False
        self.can_capture = True
        self.reason = None

    def invalidate(self, reason):
        self.clear()
        self.reason = reason

    def update(self, goals, error):
        self.valid = goals is not None
        self.reason = error
        return goals

    def has_reference(self, arm):
        return True

    def release(self, arm):
        return False

    def target(self, arm, goal, hand, scale):
        pelvis = self.robot.get_T_world_frame(PELVIS_FRAME)
        torso = self.robot.get_T_world_frame(self.torso)
        # The retargeter publishes torso-relative visualization and converts
        # this transform back to model coordinates for LeRobot's IK task.
        return np.linalg.solve(torso, pelvis @ goal)


@dataclass
class ArmResult:
    positions: dict | None
    status: str
    hold_reason: str | None
    active: dict
    target_poses: dict
    waist_yaw: float | None = None


def start_arm_solver(settings, profile):
    # The worker owns the native solver and its mutable history. Spawn avoids
    # inheriting the XR SDK, the bridge sockets, or a locked native runtime.
    global _arm_solver, _arm_clutch
    _arm_solver = ExternalStateRetargeter(
        settings, ik_backend="lerobot", profile=nullcontext(profile), tracking_reference="waist"
    )
    _arm_solver.waist_tracking = PelvisGoalTasks(_arm_solver)
    _arm_solver.backend = WaistAssistedBackend(_arm_solver)
    _arm_clutch = None


def solve_arms(state, frame, policy, restart, clutch, accepted):
    global _arm_clutch
    solver = _arm_solver
    if accepted is not None:
        solver.acknowledge(accepted)
    if policy is not None and policy != solver.settings:
        solver.configure_collision_policy(policy)
    if restart == "reset":
        solver.reset()
    elif restart == "rebase":
        solver.rebase()
    if frame is not None and not frame.return_to_default:
        # The input loop has already seen both grips released.
        solver._need_release = False
        if _arm_clutch != clutch:
            solver.backend.reset()
        _arm_clutch = clutch
    preferred_yaw = state.get("preferred_waist_yaw")
    solver.backend.begin(state["positions"], preferred_yaw, tuple(state["xr_posture"]["limits"][1]))
    positions = solver.step(
        state["positions"],
        state["last_commanded_positions"],
        state["default_positions"],
        state["position_limits"],
        intent=state["intent"],
        frame=frame,
        control_dt=state["control_dt"],
        desired_only=True,
        streaming=True,
        authoritative_executor=True,
        arm_velocity_limit=state["arm_velocity_limit"],
        return_velocity_limit=state["return_velocity_limit"],
    )
    waist_yaw = solver.backend.solved_yaw if preferred_yaw is not None and positions is not None else None
    return ArmResult(positions, solver.status, solver.hold_reason, solver.active, solver.target_poses, waist_yaw)


class ArmWorker:
    """Run one IK job at a time and publish on the bridge's ZMQ sockets."""

    def __init__(self, retargeter):
        self.profile = retargeter.profile
        self.joint_names = retargeter.joint_names
        self.command_names = retargeter.command_names
        self.hand_names = retargeter.hand_names
        self.arm_joints = retargeter.arm_joints
        self.pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=start_arm_solver,
            initargs=(retargeter.settings, self.profile),
        )
        self.policy = None
        self.restart = None
        self.future = None
        self.result = None
        self.accepted = None

    def __getattr__(self, name):
        if name in {"status", "hold_reason", "active", "target_poses"} and self.result is not None:
            return getattr(self.result, name)
        raise AttributeError(name)

    def configure_collision_policy(self, settings):
        self.policy = settings

    def reset(self):
        self.restart = "reset"

    def rebase(self):
        if self.restart is None:
            self.restart = "rebase"

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def submit(self, state, frame, token, received_at, observed_at):
        self.request = state, frame, token, received_at, observed_at
        self.future = self.pool.submit(solve_arms, state, frame, self.policy, self.restart, token, self.accepted)
        self.restart = self.accepted = None

    def step(self, *_args, **_kwargs):
        """Supply the completed result to the standard bridge publisher."""
        return self.result.positions

    def acknowledge(self, positions):
        if positions != self.result.positions:
            return False
        self.accepted = positions
        return True


class WaistIKSolver(LeRobotIKSolver):
    def _build_optimizer(self, ca):
        super()._build_optimizer(ca)
        yaw = self.model.idx_qs[self.model.getJointId(WAIST_YAW)]
        self.tracking_cost += (self.q[yaw] - self.reference[yaw]) ** 2


class WaistAssistedBackend:
    """Use the existing arm transport and finger path with a separate waist result."""

    def __init__(self, retargeter):
        self.retargeter = retargeter
        self.arm_backend = retargeter.backend
        self.solvers = {}
        self.begin(None, None, None)

    def __getattr__(self, name):
        return getattr(self.arm_backend, name)

    def begin(self, measured, preferred_yaw, yaw_limits):
        self.measured = measured
        self.preferred_yaw = preferred_yaw
        self.yaw_limits = yaw_limits
        self.solved_yaw = None

    def _solver(self, active):
        key = active, self.yaw_limits
        solver = self.solvers.get(key)
        if solver is None:
            spec = self.arm_backend.spec
            # Only engaged hands enter the objective. Bounds still hold every inactive arm joint.
            solver = WaistIKSolver(
                replace(
                    spec,
                    joints=(*spec.joints, WAIST_YAW),
                    frames={side: spec.frames[side] for side in active},
                    groups={side: spec.groups[side] for side in active},
                    limits={**spec.limits, WAIST_YAW: self.yaw_limits},
                )
            )
            self.solvers[key] = solver
        return solver

    def propose(self, problem):
        if self.preferred_yaw is None or problem.joint_target is not None or not problem.active:
            return self.arm_backend.propose(problem)
        self.solved_yaw = None
        groups = self.arm_backend.spec.groups
        active = tuple(side for side, names in groups.items() if problem.active.intersection(names))
        solver = self._solver(active)
        retargeter = self.retargeter
        robot = retargeter.robot
        previous_configuration = robot.state.q.copy()
        offsets = [robot.get_joint_offset(name) for name in solver.joint_names]
        previous = {**problem.previous, WAIST_YAW: self.preferred_yaw}
        # Keep the measured baseline separate from the candidate configurations sent to FK.
        measured = np.array(previous_configuration)
        for name, value in self.measured.items():
            measured[robot.get_joint_offset(name)] = value

        # This solve plans arm/waist motion with the sampled fingers fixed.
        # The robot executor separately checks actual finger proposals together with the arms.
        baseline = measured.copy()
        baseline[offsets[:-1]] = previous_configuration[offsets[:-1]]

        def try_candidate(yaw_min, yaw_max):
            snapshot = solver.capture_state()
            try:
                candidate = solver.solve(
                    replace(
                        problem,
                        measured=self.measured,
                        previous=previous,
                        lower=np.r_[problem.lower, yaw_min],
                        upper=np.r_[problem.upper, yaw_max],
                        hard=hard,
                        soft=soft,
                    )
                )
                if retargeter.settings.collision_checking:
                    configuration = baseline.copy()
                    configuration[offsets] = [candidate[name] for name in solver.joint_names]
                    # The waist policy and arm executor can begin applying commands in either order.
                    retargeter._validate_motion(measured, baseline, configuration)
                    for first in (offsets[-1:], offsets[:-1]):
                        intermediate = baseline.copy()
                        intermediate[first] = configuration[first]
                        if np.array_equal(intermediate, baseline) or np.array_equal(intermediate, configuration):
                            continue
                        retargeter._validate_motion(measured, baseline, intermediate)
                        retargeter._validate_motion(intermediate, intermediate, configuration)
                poses = solver.forward_kinematics({**self.measured, **candidate})
                error = 0.0
                for side, pose in poses.items():
                    target = problem.targets[side]
                    distance = np.linalg.norm(pose[:3, 3] - target[:3, 3])
                    cosine = (np.trace(pose[:3, :3].T @ target[:3, :3]) - 1) / 2
                    angle = math.acos(float(np.clip(cosine, -1, 1)))
                    error = max(error, distance / POSITION_TOLERANCE, angle / ROTATION_TOLERANCE)
                return candidate, error
            except RuntimeError:
                return None, math.inf
            finally:
                solver.restore_state(snapshot)
                robot.state.q = previous_configuration.copy()
                robot.update_kinematics()

        try:
            robot.state.q = baseline
            robot.update_kinematics()
            hard, soft = retargeter._collision_constraints(baseline, offsets)
            candidate, error = try_candidate(self.preferred_yaw, self.preferred_yaw)
            if error > 1:
                assisted, assisted_error = try_candidate(*self.yaw_limits)
                if assisted is not None and assisted_error < error:
                    candidate = assisted
        finally:
            robot.state.q = previous_configuration
            robot.update_kinematics()
        if candidate is None:
            raise RuntimeError("Pelvis IK could not produce a valid arm and waist candidate")
        self.solved_yaw = candidate.pop(WAIST_YAW)
        # The selected candidate is stateless. The executor feeds back the next arm baseline.
        return IKProposal(candidate)

    def accept(self, proposal, positions):
        if proposal.pending is not None:
            self.arm_backend.accept(proposal, positions)


def horizontal_unit(direction):
    direction = np.asarray(direction, dtype=float).copy()
    direction[1] = 0.0
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length < 1e-3:
        raise ValueError("XR body heading is undefined")
    return direction / length


def torso_heading(pelvis, shoulders):
    shoulders = np.asarray(shoulders, dtype=float)
    if shoulders.shape != (2, 3) or not np.isfinite(shoulders).all():
        raise ValueError("XR shoulder tracking unavailable")
    forward = horizontal_unit(-pelvis[:3, 2])
    axis = horizontal_unit(shoulders[0] - shoulders[1])
    return np.stack((np.cross(forward, UP), UP, -forward)) @ axis


class PelvisIKBridge(RoboJuDoBridge):
    def __init__(
        self,
        *args,
        robot="x2_omnihand",
        ik_backend="lerobot",
        posture_endpoint="tcp://127.0.0.1:8557",
        **kwargs,
    ):
        if robot != "x2_omnihand" or ik_backend != "lerobot":
            raise ValueError("pelvis-ik requires --robot x2_omnihand --ik-backend lerobot")
        super().__init__(*args, robot=robot, ik_backend=ik_backend, tracking_reference="waist", **kwargs)
        try:
            context = kwargs.get("context") or zmq.Context.instance()
            self.posture_socket = context.socket(zmq.PUB)
            self._resources.callback(self.posture_socket.close, linger=0)
            self.posture_socket.setsockopt(zmq.SNDHWM, 1)
            self.posture_socket.bind(posture_endpoint)
            self.tracking = WaistTracking(self.settings.tracking_timeout_s)
            self.hands = HandGoals(self.retargeter)
            self.retargeter = ArmWorker(self.retargeter)
            self._resources.callback(self.retargeter.close)
        except Exception:
            self.close()
            raise
        self._posture_sequence = 0
        self._session = None
        self._authority = None
        self._arm_ready = False
        self._posture_ready = False
        self._reference = None
        self._reference_epoch = 0
        self._active_grips = ()
        self._heading = None
        self._twist = 0.0
        self._height = None
        self._yaw = None
        self._preferred_yaw = None
        self._xr_stamp = None
        self._xr_advanced_at = float("-inf")
        self._home_pressed = False
        self._home_pending = False
        self._home_frame_id = None
        self._pending_yaw = {}
        self._publish_frame = None

    def validate_feedback(self, state):
        super().validate_feedback(state)
        posture = state.get("xr_posture")
        if not isinstance(posture, dict):
            raise ValueError("robot has no pelvis IK posture capability")
        if (
            type(posture.get("schema_version")) is not int
            or posture["schema_version"] != 1
            or type(posture.get("epoch")) is not int
            or posture["epoch"] < 0
            or type(posture.get("available")) is not bool
        ):
            raise ValueError("invalid XR posture capability")
        limits, defaults, targets = (posture.get(key) for key in ("limits", "defaults", "targets"))
        if any(not isinstance(value, list) or len(value) != 2 for value in (limits, defaults, targets)):
            raise ValueError("invalid XR posture limits, defaults or targets")
        for bounds, default, target in zip(limits, defaults, targets, strict=True):
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("invalid XR posture bounds")
            lower, upper = map(_number, bounds)
            if lower >= upper or not lower <= _number(default) <= upper or not lower <= _number(target) <= upper:
                raise ValueError("XR posture defaults or targets outside limits")

    def _clear_reference(self, *, reset=False):
        self._reference_epoch += 1
        self._reference = self._heading = None
        self._preferred_yaw = self._height = self._yaw = None
        self._active_grips = ()
        self.hands.reset()
        self._pending_yaw.clear()
        if reset:
            self._arm_ready = self._posture_ready = False
            self._home_pending = False
            self._home_frame_id = None

    def _hold(self, reason):
        self._clear_reference()
        self.tracking.invalidate(reason)
        return super()._hold(reason)

    def _xr_frame(self):
        return self._publish_frame

    def _posture(self, state, height, yaw):
        posture = state["xr_posture"]
        if not posture["available"] or time.monotonic() > self._feedback_deadline(state, self._received_at):
            return False
        self._posture_sequence += 1
        try:
            self.posture_socket.send_json(
                {
                    "schema_version": 1,
                    "bridge_id": self.bridge_id,
                    "session_id": state["session_id"],
                    "generation": state["generation"],
                    "state_id": state["state_id"],
                    "epoch": posture["epoch"],
                    "sequence": self._posture_sequence,
                    "height": float(height),
                    "waist_yaw": float(yaw),
                },
                flags=zmq.NOBLOCK,
                allow_nan=False,
            )
        except zmq.ZMQError as exc:
            self._warn(exc)
            return False
        return True

    @staticmethod
    def _feedback_deadline(state, received_at):
        return received_at + state["feedback_timeout_s"] - max(state["joint_age_s"].values())

    def _sample(self, state, now):
        self._ensure_xr()
        xr = self.xr_client
        stamp = xr.get_timestamp_ns()
        if type(stamp) is not int or stamp <= 0 or self._xr_stamp is None or stamp < self._xr_stamp:
            self._xr_advanced_at = float("-inf")
        elif stamp > self._xr_stamp:
            self._xr_advanced_at = now
        self._xr_stamp = stamp if type(stamp) is int else None
        fresh = now - self._xr_advanced_at <= self.settings.tracking_timeout_s
        home = fresh and xr.get_button_state_by_name(self.retargeter.profile.home_button)
        if home and not self._home_pressed:
            self._clear_reference(reset=True)
            self._home_pending = True
        self._home_pressed = home
        if self._home_pending and not fresh:
            return None, now
        if self._home_pending and not home and self._home_complete(state):
            self._home_pending = False
            self._home_frame_id = None
        if self._home_pending:
            self._posture(state, *state["xr_posture"]["defaults"])
            return XRFrame(stamp, {}, {}, return_to_default=True), now
        sample = xr.get_waist_sample(pelvis_ik=True)
        controllers = self.tracking.update(sample)
        if controllers is None:
            raise ValueError(self.tracking.reason)
        if any(
            not isinstance(values, dict)
            or set(values) != set(controllers)
            or any(isinstance(v, bool) or not isinstance(v, Real) or not 0 <= v <= 1 for v in values.values())
            for values in (sample.grips, sample.triggers)
        ):
            raise ValueError("invalid XR grips or triggers")
        pelvis = xr_pose_to_matrix(sample.pelvis_pose)
        heading = torso_heading(pelvis, sample.shoulder_positions)
        if self._heading is not None and np.dot(heading, self._heading) < math.cos(1):
            raise ValueError("XR torso heading discontinuity")
        self._heading = heading
        pressed = tuple(side for side, grip in sample.grips.items() if grip > self.settings.grip_threshold)
        if not pressed:
            self._arm_ready = True
            self._posture_ready = state["xr_posture"]["available"]
        if not self._arm_ready:
            self.hold_reason = "release both grips before engaging pelvis IK"
            return None, sample.observed_at
        grips = self.hands.eligible_grips(sample.grips, can_capture=self.tracking.can_capture)
        active = tuple(side for side in pressed if grips[side] > self.settings.grip_threshold)
        if active != self._active_grips:
            self._reference_epoch += 1
            self._pending_yaw.clear()
            self._active_grips = active
        posture_grips = pressed if self._reference is not None else active
        self._sample_posture(state, posture_grips, pelvis[1, 3], heading)
        goals = self.hands.update(grips, controllers, state["positions"])
        frame = XRFrame(sample.xr_timestamp_ns, {}, grips, triggers=sample.triggers, waist=goals)
        return frame, sample.observed_at

    def _home_complete(self, state):
        if (
            self._home_frame_id is None
            or state.get("applied_bridge_id") != self.bridge_id
            or (state.get("applied_frame_id") or 0) < self._home_frame_id
        ):
            return False
        posture = state["xr_posture"]
        return (not posture["available"] or posture["targets"] == posture["defaults"]) and all(
            abs(state[key][name] - state["default_positions"][name]) <= HOME_ARM_TOLERANCE_RAD
            for names in self.retargeter.arm_joints.values()
            for name in names
            for key in ("positions", "last_commanded_positions")
        )

    def _acknowledge_waist(self, state):
        applied = state.get("applied_frame_id")
        if state.get("applied_bridge_id") != self.bridge_id or type(applied) is not int:
            return
        if applied in self._pending_yaw and self._height is not None:
            self._yaw = self._pending_yaw[applied]
        self._pending_yaw = {key: yaw for key, yaw in self._pending_yaw.items() if key > applied}

    def _sample_posture(self, state, active, height, heading):
        self._preferred_yaw = None
        posture = state["xr_posture"]
        if not active or not self._posture_ready or not posture["available"]:
            self._reference = None
            self._height = self._yaw = None
            return
        (min_height, max_height), (min_yaw, max_yaw) = posture["limits"]
        if self._reference is None:
            self._height, self._yaw = posture["targets"]
            self._reference = height, heading, self._height, self._yaw
            self._twist = 0.0
        anchor_height, anchor_heading, base_height, base_yaw = self._reference
        twist = math.atan2(np.dot(UP, np.cross(anchor_heading, heading)), np.dot(anchor_heading, heading))
        self._twist += math.remainder(twist - self._twist, math.tau)
        height = base_height + height - anchor_height
        requested_height = np.clip(self._height, height - 0.01, height + 0.01)
        requested_height = float(np.clip(requested_height, min_height, max_height))
        yaw = math.copysign(max(0.0, abs(self._twist) - 0.03), self._twist)
        self._preferred_yaw = float(np.clip(base_yaw + yaw, min_yaw, max_yaw))
        self._height = requested_height

    def poll(self):
        try:
            return self._poll()
        finally:
            self._command_log.report(self.status)

    def _poll(self):
        state = self.receive_feedback()
        if not state:
            return False
        session = state["session_id"], state["generation"]
        if session != self._session:
            recoverable = (
                self._session is not None
                and self._session[0] == session[0]
                and (state.get("reset_reason") == "tracking command timeout")
            )
            if recoverable:
                # An upper-body lease can expire while the standalone posture stream
                # and body tracking are still live. Only fence the old arm job.
                self._reference_epoch += 1
                self._pending_yaw.clear()
                self._home_frame_id = None
            else:
                self._clear_reference(reset=True)
            self._session = session
        posture = state["xr_posture"]
        authority = posture["epoch"], posture["available"]
        if authority != self._authority:
            self._reference = None
            self._posture_ready = False
            self._reference_epoch += 1
            self._pending_yaw.clear()
            self._authority = authority
        frame, observed_at = None, time.monotonic()
        if state["intent"] == "TRACK":
            try:
                frame, observed_at = self._sample(state, observed_at)
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
                self._clear_reference()
                if self.tracking.valid:
                    self.tracking.invalidate(str(exc))
                self.status, self.hold_reason = "hold", str(exc)
        elif state["intent"] != "HOME":
            self.status, self.hold_reason = "hold", "robot is not accepting XR targets"

        worker = self.retargeter
        allowed = frame is not None or state["intent"] == "HOME"
        token = session, self._reference_epoch, bool(frame and frame.return_to_default)
        self._acknowledge_waist(state)
        published = False
        if worker.future is not None and worker.future.done():
            old_state, old_frame, old_token, received_at, sampled_at = worker.request
            try:
                worker.result = worker.future.result()
                now = time.monotonic()
                if (
                    allowed
                    and old_token == token
                    and old_state["intent"] == state["intent"]
                    and now <= self._feedback_deadline(old_state, received_at)
                    and (old_frame is None or now - sampled_at <= self.settings.tracking_timeout_s)
                ):
                    self._publish_frame = old_frame
                    self._received_at, current_received_at = received_at, self._received_at
                    try:
                        published = super()._publish_desired(old_state)
                    finally:
                        self._received_at = current_received_at
                        self._publish_frame = None
                    yaw = worker.result.waist_yaw
                    logger.debug(
                        "Pelvis IK yaw preferred=%s solved=%s measured=%.3f published=%s",
                        old_state["preferred_waist_yaw"],
                        yaw,
                        old_state["positions"][WAIST_YAW],
                        published,
                    )
                    if published:
                        if old_frame is not None and old_frame.return_to_default and self._home_frame_id is None:
                            self._home_frame_id = self._frame_id
                        if yaw is not None and self._height is not None:
                            if len(self._pending_yaw) == 16:
                                del self._pending_yaw[next(iter(self._pending_yaw))]
                            self._pending_yaw[self._frame_id] = yaw
            except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
                self._warn(exc)
            finally:
                worker.result = worker.future = None
        if allowed and worker.future is None:
            solve_state = {**state, "preferred_waist_yaw": self._preferred_yaw if self._reference else None}
            worker.submit(solve_state, frame, token, self._received_at, observed_at)
        if frame is not None and not frame.return_to_default and self._height is not None:
            self._posture(state, self._height, self._yaw)
        return published
