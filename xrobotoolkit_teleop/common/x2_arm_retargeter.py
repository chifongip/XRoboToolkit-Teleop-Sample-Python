"""Headless, externally measured X2 arm retargeting; no SDK or physics loop."""

import logging
import math
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import placo
from meshcat import transformations as tf
from xrobotoolkit_teleop.common.x2_collision import X2CollisionMixin
from xrobotoolkit_teleop.common.x2_defaults import DEFAULTS
from xrobotoolkit_teleop.utils.collision_checks import CollisionChecker
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

logger = logging.getLogger(__name__)


@dataclass
class X2XRFrame:
    timestamp_ns: int
    poses: dict
    grips: dict
    return_to_default: bool = False


def deployment_urdf():
    """Restore measured body DOFs while retaining fitted collision geometry."""
    assets = Path(ASSET_PATH) / "agibot/x2"
    root = ET.parse(assets / "x2_fixed.urdf").getroot()
    arm_names = [joint.get("name") for joint in root.findall("joint") if joint.get("type") == "revolute"]
    source = ET.parse(assets / "x2_ultra.urdf").getroot()
    for joint in source.findall("joint"):
        if joint.get("type") != "fixed" and joint.get("name") not in arm_names:
            old = root.find(f"joint[@name='{joint.get('name')}']")
            if old is None:
                raise ValueError("X2 deployment model does not match the fixed-body model")
            root.remove(old)
            root.append(deepcopy(joint))
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((assets / mesh.get("filename")).resolve()))
    return root, arm_names


class X2ArmRetargeter(X2CollisionMixin):
    def __init__(self, settings=DEFAULTS):
        self.settings = settings
        self.collision_margin = settings.collision_margin
        self.collision_trigger = settings.collision_trigger
        self._solver_collision_margin = settings.collision_margin + min(
            0.002, (settings.collision_trigger - settings.collision_margin) / 2
        )
        root, self.arm_names = deployment_urdf()
        if len(self.arm_names) != 14:
            raise ValueError("the arm-only retargeter requires exactly 14 X2 arm joints")
        self._model_dir = TemporaryDirectory(prefix="x2-teleop-ik-")
        path = Path(self._model_dir.name) / "x2.urdf"
        pairs_path = Path(ASSET_PATH) / "agibot/x2/collisions.json"
        # Placo looks beside the URDF during construction. Supply the explicit
        # pair list before loading, rather than auto-discovering adjacent bodies.
        try:
            ET.ElementTree(root).write(path)
            (path.parent / "collisions.json").write_bytes(pairs_path.read_bytes())
            self.placo_robot = placo.RobotWrapper(str(path))
            self.placo_robot.load_collision_pairs(str(pairs_path))
            self.collision_checker = CollisionChecker(self.placo_robot)
            self.solver = placo.KinematicsSolver(self.placo_robot)
            self.solver.mask_fbase(True)
            self.solver.enable_joint_limits(True)
            # Command bounds are centered on the acknowledged target, not on
            # measured servo error. Both command paths are checked below.
            self.solver.enable_velocity_limits(False)
            self.joint_names = list(self.placo_robot.joint_names())
            if len(self.joint_names) != 31:
                raise ValueError("the deployment model must expose all 31 X2 joints")
            for name in set(self.joint_names) - set(self.arm_names):
                self.solver.mask_dof(name)
                self.placo_robot.set_joint_limits(name, -1e6, 1e6)
            self.arm_joints = {
                f"{side}_arm": [name for name in self.arm_names if name.startswith(f"{side}_")]
                for side in ("left", "right")
            }
            self.limits = {}
            self.velocities = {}
            for name in self.arm_names:
                limit = root.find(f"joint[@name='{name}']/limit")
                self.limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
                self.velocities[name] = float(limit.get("velocity"))
            self.tasks = {
                side: self.solver.add_frame_task(f"{side}_hand_center", np.eye(4)) for side in ("left", "right")
            }
            for side, task in self.tasks.items():
                task.configure(f"{side}_hand", "soft", 10.0)
            self.posture = self.solver.add_joints_task()
            self.posture.configure("arm_posture", "soft", 1e-4)
            self.collision_constraint = self.solver.add_joint_space_half_spaces_constraint(
                np.zeros((1, self.placo_robot.model.nq)), np.ones(1)
            )
            self.collision_constraint.configure("x2_self_collision", "hard")
        except Exception:
            self.close()
            raise
        self._last_timestamp = None
        self._last_advance = None
        self._last_error_log = float("-inf")
        self.status = "hold"
        self.hold_reason = "waiting for XR input"
        self.target_poses = {}
        self.reset()

    def close(self):
        self._model_dir.cleanup()

    def reset(self):
        self._need_release = True
        self.rebase()

    def rebase(self):
        """Discard motion during a hold without revoking an operator's engagement."""
        self._heading = None
        self._references = {}
        self.active = {"left_arm": False, "right_arm": False}

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
        grips = [frame.grips.get(side, float("nan")) for side in ("left", "right")]
        if not np.isfinite(grips).all():
            self.hold_reason = "invalid XR grip values"
            self.rebase()
            return False
        if self._need_release:
            self.hold_reason = "release both grips before engaging"
            if all(value <= self.settings.grip_threshold for value in grips):
                self._need_release = False
                self.hold_reason = "grips released; waiting for engagement"
            return False
        self.active = {
            f"{side}_arm": value > self.settings.grip_threshold
            for side, value in zip(("left", "right"), grips, strict=True)
        }
        if not any(self.active.values()):
            self.hold_reason = "neither grip exceeds the activation threshold"
            self._heading = None
            self._references.clear()
            return False
        if self._heading is None:
            forward = -self._pose(frame, "headset")[:3, 2]
            forward[1] = 0
            if np.linalg.norm(forward) < 1e-6:
                raise ValueError("headset forward direction is vertical")
            forward /= np.linalg.norm(forward)
            up = np.array([0.0, 1.0, 0.0])
            self._heading = R_HEADSET_TO_WORLD @ np.column_stack((np.cross(forward, up), up, -forward)).T
        torso = self.placo_robot.get_T_world_frame("torso_link")
        for side in ("left", "right"):
            if not self.active[f"{side}_arm"]:
                self._references.pop(side, None)
                continue
            controller = self._pose(frame, f"{side}_controller")
            if side not in self._references:
                hand = np.linalg.inv(torso) @ self.placo_robot.get_T_world_frame(f"{side}_hand_center")
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
        robot = self.placo_robot
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
                        raise RuntimeError("X2 command path violates self-collision clearance")
                    floor = np.maximum(floor, np.minimum(values, self.collision_margin))
                violating = measured_distances < self.collision_margin - 1e-6
                if (
                    start is measured
                    and np.any(violating)
                    and not np.any(values[violating] > measured_distances[violating] + 1e-8)
                ):
                    raise RuntimeError("X2 collision recovery made no progress")
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
        """Return a complete checked float32 proposal, or None to retain targets."""
        self.status = "hold"
        self.hold_reason = None
        self.target_poses = {}
        self._captured_reference = False
        robot = self.placo_robot
        if set(measured_positions) != set(self.joint_names):
            raise ValueError("measured X2 joints must match all 31 model joints")
        if any(
            set(payload) != set(self.arm_names) for payload in (previous_targets, default_positions, position_limits)
        ):
            raise ValueError("arm target/default/limit names must match the 14 X2 arm joints")
        if not np.isfinite(list(measured_positions.values())).all():
            raise ValueError("measured X2 state must be finite")
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
        dt = 1 / self.settings.control_frequency if control_dt is None else control_dt
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        self.solver.dt = dt
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
                robot.set_joint_limits(name, low, high)
                offset = robot.get_joint_offset(name)
                distance = min(cap, self.velocities[name]) * dt
                bounds[name] = (max(low, previous[offset] - distance), min(high, previous[offset] + distance))
                if bounds[name][0] > bounds[name][1]:
                    raise RuntimeError("previous target is outside compatible limits")
                side = name.split("_", 1)[0]
                active = self.active[f"{side}_arm"]
                (self.solver.unmask_dof if active else self.solver.mask_dof)(name)
            for side, task in self.tasks.items():
                task.configure(f"{side}_hand", "soft", 0.0 if homing else 10.0)
            # A moving posture reference preserves every redundant-joint error
            # and loses the preferred elbow branch near a straight-arm pose.
            # Match the demo's fixed reference; previous targets only seed IK
            # and define command speed bounds, never the posture objective.
            self.posture.set_joints(default_positions)
            self.posture.configure("arm_posture", "soft", 1.0 if homing else 1e-4)
            if self._captured_reference and not homing:
                # The engagement/recovery frame only anchors the clutch. Do not
                # let posture regularization change a held target on this frame.
                candidate = self._quantize(previous.copy(), bounds, previous_targets)
                self._validate_motion(measured, previous, candidate)
                self.status = "tracking"
                return self._positions(candidate)
            if homing:
                for name, value in default_positions.items():
                    robot.set_joint(name, value)
                robot.update_kinematics()
                torso_inverse = np.linalg.inv(robot.get_T_world_frame("torso_link"))
                self.target_poses = {
                    side: (torso_inverse @ robot.get_T_world_frame(f"{side}_hand_center")).tolist()
                    for side in ("left", "right")
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
                    return self._positions(candidate)
                except RuntimeError:
                    robot.state.q = previous.copy()
            rows, values = [], []
            for name, (lower, upper) in bounds.items():
                row = np.zeros(len(measured))
                row[robot.get_joint_offset(name)] = 1
                rows.extend((row, -row))
                values.extend((upper, -lower))
            step_constraint = self.solver.add_joint_space_half_spaces_constraint(np.array(rows), np.array(values))
            step_constraint.configure("command_step", "hard")
            recovery = None
            try:
                distances = self.collision_checker.distances()
                recovering = np.any(distances < self.collision_margin - 1e-6)
                self._update_collision_constraint(self.collision_constraint)
                self.collision_constraint.configure("x2_self_collision", "soft" if recovering else "hard", 1e6)
                if recovering:
                    recovery = self._add_recovery_constraint(distances)
                self.solver.solve(True)
                candidate = self._checked_candidate(measured, previous, robot.state.q.copy(), bounds, previous_targets)
            finally:
                self.solver.remove_constraint(step_constraint)
                if recovery is not None:
                    self.solver.remove_constraint(recovery)
                self.collision_constraint.configure("x2_self_collision", "hard")
            self.status = "homing" if homing else "tracking"
            return self._positions(candidate)
        except (RuntimeError, ValueError) as exc:
            self.hold_reason = str(exc) or type(exc).__name__
            if isinstance(exc, ValueError):
                self.rebase()
            if time.monotonic() - self._last_error_log >= 1:
                logger.warning("X2 retargeter holding targets: %s", exc)
                self._last_error_log = time.monotonic()
            return None
        finally:
            robot.state.q = measured
            robot.update_kinematics()

    def _positions(self, configuration):
        return {name: float(configuration[self.placo_robot.get_joint_offset(name)]) for name in self.arm_names}

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
            offset = self.placo_robot.get_joint_offset(name)
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
            if not self.active[f"{name.split('_')[0]}_arm"]:
                held = np.float32(previous_targets[name])
                if not low32 <= held <= high32:
                    raise RuntimeError(f"held arm target is outside compatible limits: {name}")
                candidate[offset] = held
            else:
                candidate[offset] = np.clip(np.float32(candidate[offset]), low32, high32)
        return candidate
