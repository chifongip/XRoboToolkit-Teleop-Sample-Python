"""Profile-driven external-state retargeting; no SDK or physics loop."""

import logging
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace

import numpy as np
from meshcat import transformations as tf

from xrobotoolkit_teleop.common.teleop_settings import DEFAULTS
from xrobotoolkit_teleop.ik import create_backend
from xrobotoolkit_teleop.ik.safety import CollisionPolicy
from xrobotoolkit_teleop.ik.types import IKProblem, LinearConstraint, PoseTarget
from xrobotoolkit_teleop.integrations.robot_profiles import load_profile
from xrobotoolkit_teleop.utils.collision_checks import CollisionChecker
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

logger = logging.getLogger(__name__)


@dataclass
class XRFrame:
    timestamp_ns: int
    poses: dict
    grips: dict
    return_to_default: bool = False


class ExternalStateRetargeter(CollisionPolicy):
    def __init__(self, settings=DEFAULTS, *, robot="x2", ik_backend="placo", profile=None):
        self.settings = settings
        self._resources = ExitStack()
        self.backend = None
        self._pending = None
        self._pending_positions = None
        self._closed = False
        try:
            description = self._resources.enter_context(load_profile(robot, settings) if profile is None else profile)
            description.validate()
            # The robot-owned posture reference changes at runtime. Keep it
            # private so reusing a profile cannot modify another bridge's defaults.
            self.profile = replace(
                description,
                spec=replace(description.spec, initial_positions=description.spec.initial_positions.copy()),
            )
            self.backend = create_backend(ik_backend, self.profile.spec)
            self._resources.callback(self.backend.close)
            self.robot = self.backend.robot
            self.joint_names = list(self.profile.observed_joints)
            self.arm_names = list(self.profile.spec.joints)  # compatibility with existing callers
            self.arm_joints = {key: list(g.joints) for key, g in self.profile.groups.items()}
            self._joint_groups = {joint: key for key, joints in self.arm_joints.items() for joint in joints}
            if set(self.robot.joint_names()) != set(self.joint_names):
                raise ValueError("Profile observed joints do not match the model")
            for frame in [self.profile.reference_frame, *self.profile.spec.frames.values()]:
                self.robot.get_T_world_frame(frame)
            self.limits = self.profile.spec.limits.copy()
            self.velocities = self.profile.velocities.copy()
            self.tasks = {
                key: PoseTarget(self.robot.get_T_world_frame(g.frame)) for key, g in self.profile.groups.items()
            }
            self.collision_checker = CollisionChecker(self.robot)
            self.collision_constraint = LinearConstraint(np.zeros((1, self.robot.model.nq)), np.ones(1))
            self.collision_margin = settings.collision_margin
            self.collision_trigger = settings.collision_trigger
            self._solver_collision_margin = settings.collision_margin + min(
                0.002, (settings.collision_trigger - settings.collision_margin) / 2
            )
        except Exception:
            self._resources.close()
            raise
        self._last_timestamp = None
        self._last_advance = None
        self._last_error_log = float("-inf")
        self.status = "hold"
        self.hold_reason = "waiting for XR input"
        self.target_poses = {}
        self.reset()

    @property
    def placo_robot(self):
        """Compatibility alias; the returned object is a numeric model adapter."""
        return self.robot

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.reject()
            finally:
                self._resources.close()

    def reject(self):
        pending = self._pending
        self._pending = None
        self._pending_positions = None
        if pending is not None:
            self.backend.reject(pending)

    def acknowledge(self, positions):
        if self._closed:
            raise RuntimeError("Retargeter is closed")
        if self._pending_positions is None:
            return False
        if self._pending_positions != positions:
            self.reject()
            self.backend.reset()
            return False
        if self._pending is not None:
            self.backend.accept(self._pending, positions)
        self._pending = None
        self._pending_positions = None
        return True

    def _output(self, candidate):
        positions = self._positions(candidate)
        self._pending_positions = positions.copy()
        return positions

    def reset(self):
        self._need_release = True
        self.rebase()

    def rebase(self):
        """Discard interrupted motion without revoking operator engagement."""
        self.reject()
        self.backend.reset()
        self._heading = None
        self._references = {}
        self.active = dict.fromkeys(self.arm_joints, False)

    @staticmethod
    def _pose(frame, name):
        pose = np.asarray(frame.poses.get(name), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"invalid XR pose: {name}")
        norm = math.hypot(*pose[3:])
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"invalid XR quaternion: {name}")
        transform = tf.quaternion_matrix(np.roll(pose[3:] / norm, 1))
        transform[:3, 3] = pose[:3]
        return transform

    def _fresh(self, frame):
        stamp, now = frame.timestamp_ns, time.monotonic()
        if type(stamp) is not int or stamp <= 0:
            self.hold_reason = "invalid XR timestamp"
            return False
        if self._last_timestamp is None or stamp < self._last_timestamp:
            self._last_timestamp, self._last_advance = stamp, None
            self.hold_reason = "waiting for advancing XR timestamps"
            return False
        if stamp > self._last_timestamp:
            self._last_advance = now
        self._last_timestamp = stamp
        fresh = self._last_advance is not None and now - self._last_advance <= self.settings.tracking_timeout_s
        if not fresh:
            self.hold_reason = "XR timestamp stopped advancing; tracking timed out"
        return fresh

    def _tracking_targets(self, frame):
        if frame is None:
            self.hold_reason = "XR frame unavailable"
            self.rebase()
            return False
        if not self._fresh(frame):
            self.rebase()
            return False
        grips = [frame.grips.get(key, float("nan")) for key in self.arm_joints]
        if not np.isfinite(grips).all():
            self.hold_reason = "invalid XR grip values"
            self.rebase()
            return False
        if self._need_release:
            self.hold_reason = "release all grips before engaging"
            if all(value <= self.settings.grip_threshold for value in grips):
                self._need_release = False
                self.hold_reason = "grips released; waiting for engagement"
            return False
        self.active = {
            key: value > self.settings.grip_threshold for key, value in zip(self.arm_joints, grips, strict=True)
        }
        if not any(self.active.values()):
            self.hold_reason = "no grip exceeds the activation threshold"
            self._heading = None
            self._references.clear()
            self.backend.reset()
            return False
        if self._heading is None:
            forward = -self._pose(frame, self.profile.headset)[:3, 2]
            forward[1] = 0
            if np.linalg.norm(forward) < 1e-6:
                raise ValueError("headset forward direction is vertical")
            forward /= np.linalg.norm(forward)
            up = np.array([0.0, 1.0, 0.0])
            self._heading = R_HEADSET_TO_WORLD @ np.column_stack((np.cross(forward, up), up, -forward)).T
        torso = self.robot.get_T_world_frame(self.profile.reference_frame)
        for side, group in self.profile.groups.items():
            if not self.active[side]:
                self._references.pop(side, None)
                self.backend.reset(frozenset(group.joints))
                continue
            controller = self._pose(frame, group.controller)
            if side not in self._references:
                self.backend.reset(frozenset(group.joints))
                hand = np.linalg.inv(torso) @ self.robot.get_T_world_frame(self.profile.groups[side].frame)
                self._references[side] = (controller.copy(), hand)
                self._captured_reference = True
            reference, hand = self._references[side]
            local = hand.copy()
            local[:3, 3] += self.settings.scale_factor * self._heading @ (controller[:3, 3] - reference[:3, 3])
            delta = self._heading @ controller[:3, :3] @ reference[:3, :3].T @ self._heading.T
            local[:3, :3] = delta @ hand[:3, :3]
            self.tasks[side].T_world_frame = torso @ local
            self.target_poses[side] = local.tolist()
        return True

    def _validate_motion(self, measured, previous, candidate):
        robot = self.robot
        try:
            robot.state.q = measured
            measured_distances = self.collision_checker.distances()
            for start in (measured, previous):
                robot.state.q = start
                distances = self.collision_checker.distances()
                floor = np.minimum(np.minimum(distances, measured_distances), self.collision_margin)
                for configuration in self._collision_motion_samples(start, candidate, False):
                    robot.state.q = configuration
                    values = self.collision_checker.clearance_bounds(floor) if np.all(floor > 0) else None
                    if values is None or np.any(values < floor):
                        values = self.collision_checker.distances()
                    if np.any(values < floor - 1e-6):
                        raise RuntimeError("Robot command path violates self-collision clearance")
                    floor = np.maximum(floor, np.minimum(values, self.collision_margin))
                violating = measured_distances < self.collision_margin - 1e-6
                if (
                    start is measured
                    and np.any(violating)
                    and not np.any(values[violating] > measured_distances[violating] + 1e-8)
                ):
                    raise RuntimeError("Robot collision recovery made no progress")
        finally:
            robot.state.q = candidate
            robot.update_kinematics()

    def step(
        self,
        measured_positions,
        previous_targets,
        default_positions,
        position_limits,
        *,
        intent,
        frame=None,
        control_dt=None,
        arm_velocity_limit=None,
        return_velocity_limit=None,
    ):
        """Return checked float32 positions, or None to retain targets.

        History stays pending until acknowledge(applied_positions). A subsequent
        step without acknowledgement discards the previous proposal.
        """
        if self._closed:
            raise RuntimeError("Retargeter is closed")
        self.reject()
        self.status = "hold"
        self.hold_reason = None
        self.target_poses = {}
        self._captured_reference = False
        robot = self.robot
        if set(measured_positions) != set(self.joint_names):
            raise ValueError("measured joints must match the profile observed joints")
        if any(
            set(payload) != set(self.arm_names) for payload in (previous_targets, default_positions, position_limits)
        ):
            raise ValueError("targets, defaults and limits must match the profile controlled joints")
        if not np.isfinite(list(measured_positions.values())).all():
            raise ValueError("measured robot state must be finite")
        dt = 1 / self.settings.control_frequency if control_dt is None else control_dt
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        for name, value in measured_positions.items():
            robot.set_joint(name, value)
        robot.update_kinematics()
        self.collision_checker.clear_cache()
        measured = robot.state.q.copy()
        previous = measured.copy()
        for name in self.arm_names:
            if not np.isfinite(previous_targets[name]):
                raise ValueError("previous arm targets must be finite")
            previous[robot.get_joint_offset(name)] = previous_targets[name]
        # Solve around the last acknowledged command. Masked arms must remain
        # there, rather than at measured servo error outside the command bounds.
        # The complete measured-to-command path is still checked independently.
        robot.state.q = previous.copy()
        robot.update_kinematics()
        bounds = {}
        homing = intent == "HOME"
        try:
            if intent == "HOLD":
                self.hold_reason = "RoboJuDo requested HOLD (mode, ownership, or safety gate)"
                self.reset()
                return None
            if intent not in {"TRACK", "HOME"}:
                raise ValueError("unknown upper-body intent")
            if not homing:
                if frame is not None and frame.return_to_default:
                    if not self._fresh(frame):
                        self.rebase()
                        return None
                    homing = True
                elif not self._tracking_targets(frame):
                    return None
            if homing:
                self.reset()
                self.active = {name: True for name in self.arm_joints}
            caps = [self.settings.arm_velocity_limit]
            if homing:
                caps.append(self.settings.return_velocity_limit)
            for remote_cap in (arm_velocity_limit, return_velocity_limit if homing else None):
                if remote_cap is not None:
                    if not np.isfinite(remote_cap) or remote_cap <= 0:
                        raise ValueError("remote velocity cap must be finite and positive")
                    caps.append(remote_cap)
            cap = min(caps)
            for name in self.arm_names:
                low, high = position_limits[name]
                low, high = max(low, self.limits[name][0]), min(high, self.limits[name][1])
                if not np.isfinite([low, high, previous_targets[name], default_positions[name]]).all() or low >= high:
                    raise ValueError("invalid arm limits or targets")
                if not low <= default_positions[name] <= high:
                    raise ValueError(f"default pose is outside compatible arm limits: {name}")
                if not low <= previous_targets[name] <= high:
                    raise RuntimeError(f"previous target is outside compatible limits: {name}")
                robot.set_joint_limits(name, low, high)
                offset = robot.get_joint_offset(name)
                distance = min(cap, self.velocities[name]) * dt
                bounds[name] = (max(low, previous[offset] - distance), min(high, previous[offset] + distance))
                if bounds[name][0] > bounds[name][1]:
                    raise RuntimeError("previous target is outside compatible limits")
                if not self.active[self._joint_groups[name]]:
                    bounds[name] = (previous_targets[name], previous_targets[name])
            # The shared Placo objective reads the robot-owned fixed default.
            if any(self.backend.spec.initial_positions[n] != default_positions[n] for n in self.arm_names):
                self.backend.reset()
                self.backend.spec.initial_positions.update(default_positions)
            if self._captured_reference and not homing:
                # The engagement/recovery frame only anchors the clutch. Do not
                # let posture regularization change a held target on this frame.
                candidate = self._quantize(previous.copy(), bounds, previous_targets)
                self._validate_motion(measured, previous, candidate)
                self.status = "tracking"
                return self._output(candidate)
            if homing:
                for name, value in default_positions.items():
                    robot.set_joint(name, value)
                robot.update_kinematics()
                torso_inverse = np.linalg.inv(robot.get_T_world_frame(self.profile.reference_frame))
                self.target_poses = {
                    side: (torso_inverse @ robot.get_T_world_frame(self.profile.groups[side].frame)).tolist()
                    for side in self.profile.groups
                }
                robot.state.q = previous.copy()
                robot.update_kinematics()
            candidate = previous.copy()
            if homing:
                for name, (lower, upper) in bounds.items():
                    candidate[robot.get_joint_offset(name)] = np.clip(default_positions[name], lower, upper)
                try:
                    candidate = self._checked_candidate(measured, previous, candidate, bounds, previous_targets)
                    self.status = "homing"
                    return self._output(candidate)
                except RuntimeError:
                    robot.state.q = previous.copy()
            distances = self.collision_checker.distances()
            recovering = np.any(distances < self.collision_margin - 1e-6)
            self._update_collision_constraint(self.collision_constraint)
            recovery = self._add_recovery_constraint(distances) if recovering else None
            offsets = np.array([robot.get_joint_offset(n) for n in self.arm_names])
            fixed = previous.copy()
            fixed[offsets] = 0

            def controlled(constraint):
                return LinearConstraint(constraint.A[:, offsets].copy(), constraint.b - constraint.A @ fixed)

            empty = LinearConstraint(np.zeros((0, len(offsets))), np.zeros(0))
            problem = IKProblem(
                {n: robot.get_joint(n) for n in self.joint_names},
                previous_targets,
                {k: task.T_world_frame.copy() for k, task in self.tasks.items()},
                frozenset(n for k, names in self.arm_joints.items() if self.active[k] for n in names),
                np.array([bounds[n][0] for n in self.arm_names]),
                np.array([bounds[n][1] for n in self.arm_names]),
                controlled(recovery or self.collision_constraint),
                controlled(self.collision_constraint) if recovering else empty,
                dt,
                default_positions if homing else None,
            )
            for attempt in range(2):
                proposal = self.backend.propose(problem)
                try:
                    candidate = previous.copy()
                    for name, value in proposal.positions.items():
                        candidate[robot.get_joint_offset(name)] = value
                    candidate = self._checked_candidate(measured, previous, candidate, bounds, previous_targets)
                except RuntimeError:
                    self.backend.reject(proposal)
                    if attempt:
                        raise
                    self.backend.reset()
                    robot.state.q = previous.copy()
                    robot.update_kinematics()
                else:
                    self._pending = proposal
                    break
            self.status = "homing" if homing else "tracking"
            return self._output(candidate)
        except (RuntimeError, ValueError) as exc:
            self.hold_reason = str(exc) or type(exc).__name__
            if isinstance(exc, ValueError):
                self.rebase()
            if time.monotonic() - self._last_error_log >= 1:
                logger.warning("Retargeter holding targets: %s", exc)
                self._last_error_log = time.monotonic()
            return None
        finally:
            robot.state.q = measured
            robot.update_kinematics()

    def _positions(self, configuration):
        return {name: float(configuration[self.robot.get_joint_offset(name)]) for name in self.arm_names}

    def _checked_candidate(self, measured, previous, candidate, bounds, previous_targets):
        """Try bounded shorter steps when the nonlinear check rejects a QP step."""
        error = None
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = self._quantize(previous + fraction * (candidate - previous), bounds, previous_targets)
            try:
                self._validate_motion(measured, previous, trial)
                return trial
            except RuntimeError as exc:
                error = exc
        raise error

    def _quantize(self, candidate, bounds, previous_targets):
        if not np.isfinite(candidate).all():
            raise RuntimeError("non-finite IK solution")
        for name, (lower, upper) in bounds.items():
            offset = self.robot.get_joint_offset(name)
            low32 = np.float32(lower)
            high32 = np.float32(upper)
            # NumPy 2 weak scalar promotion can round a Python float down to
            # float32 during comparison, hiding an outward-rounded endpoint.
            if float(low32) < float(lower):
                low32 = np.nextafter(low32, np.float32(np.inf))
            if float(high32) > float(upper):
                high32 = np.nextafter(high32, np.float32(-np.inf))
            if low32 > high32:
                raise RuntimeError("command bounds contain no float32 target")
            if not self.active[self._joint_groups[name]]:
                held = np.float32(previous_targets[name])
                if not low32 <= held <= high32:
                    raise RuntimeError(f"held arm target is outside compatible limits: {name}")
                candidate[offset] = held
            else:
                candidate[offset] = np.clip(np.float32(candidate[offset]), low32, high32)
        return candidate
