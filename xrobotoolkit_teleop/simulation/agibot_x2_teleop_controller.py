"""Fixed-body X2 teleoperation with independently loaded IK backends."""

from contextlib import ExitStack
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer
from numpy.polynomial.polynomial import polyval

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.common.x2_defaults import DEFAULTS
from xrobotoolkit_teleop.ik import BACKENDS, create_backend
from xrobotoolkit_teleop.ik.safety import CollisionPolicy
from xrobotoolkit_teleop.ik.types import IKProblem, LinearConstraint, PoseTarget, RobotSpec
from xrobotoolkit_teleop.simulation.agibot_x2_hand_config import HAND_CLOSED
from xrobotoolkit_teleop.simulation.x2_runtime import X2MujocoRuntime
from xrobotoolkit_teleop.utils.collision_checks import CollisionChecker
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD, apply_delta_pose
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class AgibotX2TeleopController(CollisionPolicy, X2MujocoRuntime):
    """Shared XR targets, command protection, and simulation for every backend."""

    XR_TRACKING_TIMEOUT_S = DEFAULTS.tracking_timeout_s
    HAND_CLOSED = HAND_CLOSED

    def __init__(
        self,
        xml_path,
        robot_urdf_path,
        scale_factor=DEFAULTS.scale_factor,
        arm_velocity_limit=DEFAULTS.arm_velocity_limit,
        hand_velocity_limit=DEFAULTS.hand_velocity_limit,
        collision_margin=DEFAULTS.collision_margin,
        collision_trigger=DEFAULTS.collision_trigger,
        control_frequency=DEFAULTS.control_frequency,
        return_velocity_limit=DEFAULTS.return_velocity_limit,
        ik_backend="placo",
        visualize_ik=False,
    ):
        ik_backend = getattr(ik_backend, "value", ik_backend)
        if ik_backend not in BACKENDS:
            raise ValueError(f"ik_backend must be one of {tuple(BACKENDS)}")
        self.ik_backend = ik_backend
        self.backend = None
        self._closed = False
        self._pending_proposal = None
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")
        if not np.isfinite(control_frequency) or control_frequency <= 0:
            raise ValueError("control_frequency must be finite and positive")
        control_dt = 1.0 / control_frequency
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_frequency must produce a finite positive control period")
        self.control_frequency = control_frequency
        for name, value in (
            ("arm_velocity_limit", arm_velocity_limit),
            ("hand_velocity_limit", hand_velocity_limit),
            ("return_velocity_limit", return_velocity_limit),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.arm_velocity_limit = arm_velocity_limit
        self.return_velocity_limit = return_velocity_limit
        self.hand_velocity_limit = hand_velocity_limit
        if not np.isfinite(collision_margin) or collision_margin <= 0:
            raise ValueError("collision_margin must be finite and positive")
        if not np.isfinite(collision_trigger) or collision_trigger <= collision_margin:
            raise ValueError("collision_trigger must be finite and greater than collision_margin")
        self.collision_margin = collision_margin
        self.collision_trigger = collision_trigger
        config = {
            f"{side}_arm": {
                "link_name": f"{side}_hand_center",
                "pose_source": f"{side}_controller",
                "control_trigger": f"{side}_grip",
                "vis_target": f"{side}_target",
            }
            for side in ("left", "right")
        }
        self._ik_valid = False
        self._stream_fresh = False
        self._heading_captured = False
        try:
            super().__init__(
                xml_path=xml_path,
                robot_urdf_path=robot_urdf_path,
                manipulator_config=config,
                scale_factor=scale_factor,
                visualize_ik=visualize_ik,
                xr_factory=XrClient,
                dt=control_dt,
            )
            # Require an advancing stream, not just cached poses from the SDK.
            self._last_xr_timestamp_ns = self.xr_client.get_timestamp_ns()
            self._last_xr_update_at = None
        except Exception:
            self.close()
            raise

    def _robot_setup(self):
        super()._robot_setup()
        # Fit an integer number of physics steps into every control cycle.
        # Non-divisor rates such as 60 Hz use slightly smaller physics steps.
        ratio = self.dt / self.mj_model.opt.timestep
        if not np.isfinite(ratio) or ratio > np.iinfo(np.int32).max:
            raise ValueError("control_frequency requires too many physics substeps")
        self.physics_substeps = max(1, math.ceil(ratio))
        self.mj_model.opt.timestep = self.dt / self.physics_substeps
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _configure_actuators(self, arm_names):
        """Validate the supported models and separate arm/hand position servos."""
        mj_names = {self.mj_model.joint(i).name for i in range(self.mj_model.njnt)}
        hand_names = {
            f"{prefix}_{joint}_joint"
            for prefix in ("L", "R")
            for joint in set(self.HAND_CLOSED)
            | {
                "thumb_dip",
                "index_dip",
                "middle_dip",
                "ring_pip",
                "ring_dip",
                "pinky_pip",
                "pinky_dip",
            }
        }
        if len(arm_names) != 14 or mj_names not in (arm_names, arm_names | hand_names):
            raise ValueError("X2 requires matching 14 arm joints, optionally with both Omnihand Pro hands")
        if not np.all(self.mj_model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("X2 requires fixed-body models with hinge joints")
        self.arm_actuators = {}
        self.hand_actuators = {side: {} for side in ("left", "right")}
        hand_targets = {
            f"{prefix}_{joint}_joint": (
                side,
                -closed if side == "right" and joint in ("thumb_roll", "thumb_abad") else closed,
            )
            for side, prefix in (("left", "L"), ("right", "R"))
            for joint, closed in self.HAND_CLOSED.items()
        }
        expected_actuators = set(arm_names)
        if mj_names != arm_names:
            expected_actuators.update(hand_targets)
        seen = set()
        for i in range(self.mj_model.nu):
            if self.mj_model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("X2 requires joint actuators")
            if (
                not np.array_equal(self.mj_model.actuator_gear[i], [1, 0, 0, 0, 0, 0])
                or self.mj_model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE
                or self.mj_model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED
                or self.mj_model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE
                or self.mj_model.actuator_gainprm[i, 0] <= 0
                or self.mj_model.actuator_biasprm[i, 0] != 0
                or self.mj_model.actuator_biasprm[i, 1] != -self.mj_model.actuator_gainprm[i, 0]
            ):
                raise ValueError("X2 requires position servos with unit gear")
            joint = self.mj_model.joint(self.mj_model.actuator_trnid[i, 0]).name
            if joint not in expected_actuators or joint in seen:
                raise ValueError(f"Unexpected or duplicate X2 actuator for {joint}")
            seen.add(joint)
            if joint in arm_names:
                self.arm_actuators[i] = joint
            else:
                side, closed = hand_targets[joint]
                self.hand_actuators[side][i] = closed
        if seen != expected_actuators:
            raise ValueError("X2 model is missing required arm or hand actuators")
        self.hand_velocity_limits = {}
        for side, actuators in self.hand_actuators.items():
            if not actuators:
                continue
            urdf = ET.parse(Path(ASSET_PATH) / "omnihand_pro" / f"omnihand_pro_{side}.urdf").getroot()
            for i in actuators:
                joint = self.mj_model.joint(self.mj_model.actuator_trnid[i, 0]).name
                urdf_limit = float(urdf.find(f"joint[@name='{joint}']/limit").get("velocity"))
                if not np.isfinite(urdf_limit) or urdf_limit <= 0:
                    raise ValueError(f"Invalid URDF velocity limit for {joint}")
                self.hand_velocity_limits[i] = min(self.hand_velocity_limit, urdf_limit)

    def _configure_hand_couplings(self):
        """Validate and order passive joints so each is projected exactly once."""
        model, robot = self.mj_model, self.robot
        passive = set(self.hand_joint_names) - {
            model.joint(model.actuator_trnid[actuator, 0]).name for actuator in self.hand_command_offsets
        }
        pending = {}
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            if model.eq_obj1id[i] < 0 or model.eq_obj2id[i] < 0:
                raise ValueError("X2 hand coupling requires two named joints")
            target, source = (model.joint(index).name for index in (model.eq_obj1id[i], model.eq_obj2id[i]))
            if target not in passive or source not in self.hand_joint_names:
                raise ValueError("X2 joint couplings must drive passive hand joints from hand joints")
            coefficients = model.eq_data[i, :5].copy()
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("X2 hand coupling coefficients must be finite")
            target, source = robot.get_joint_offset(target), robot.get_joint_offset(source)
            if target in pending:
                raise ValueError("Duplicate X2 hand coupling target")
            pending[target] = (source, coefficients)
        ordered = []
        while pending:
            ready = [target for target, (source, _) in pending.items() if source not in pending]
            if not ready:
                raise ValueError("Cyclic X2 hand couplings")
            for target in ready:
                source, coefficients = pending.pop(target)
                ordered.append((target, source, coefficients))
        self.hand_couplings = ordered

    def _reset_clutch_reference(self, name):
        self.ref_ee_xyz[name] = None
        self.ref_ee_quat[name] = None
        self.ref_controller_xyz[name] = None
        self.ref_controller_quat[name] = None

    def _backend_setup(self):
        root = ET.parse(self.robot_urdf_path).getroot()
        movable = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
        arm_names = {name for name in movable if name.startswith(("left_", "right_"))}
        self._configure_actuators(arm_names)
        mj_names = {self.mj_model.joint(i).name for i in range(self.mj_model.njnt)}
        if movable != mj_names:
            raise ValueError("Omnihand Pro requires x2_omnihand_ik.urdf with matching articulated hand joints")
        self.arm_joints = {
            f"{side}_arm": sorted(n for n in arm_names if n.startswith(f"{side}_")) for side in ("left", "right")
        }
        if any(len(names) != 7 for names in self.arm_joints.values()):
            raise ValueError("X2 requires seven joints per arm")
        self.hand_joint_names = sorted(movable - arm_names)
        self.position_limits = {}
        velocities = {}
        for actuator, name in self.arm_actuators.items():
            limit = root.find(f"joint[@name='{name}']/limit")
            if limit is None:
                raise ValueError(f"Missing URDF limits for {name}")
            urdf_lower, urdf_upper, velocity = (float(limit.get(key, "nan")) for key in ("lower", "upper", "velocity"))
            if not np.isfinite([urdf_lower, urdf_upper, velocity]).all() or urdf_lower >= urdf_upper or velocity <= 0:
                raise ValueError(f"Invalid URDF limits for {name}")
            lower, upper = self.mj_model.actuator_ctrlrange[actuator]
            self.position_limits[name] = (max(lower, urdf_lower), min(upper, urdf_upper))
            velocities[name] = velocity
            if self.position_limits[name][0] >= self.position_limits[name][1]:
                raise ValueError(f"Incompatible position limits: {name}")
        initial = {name: float(self.mj_data.qpos[self.mj_model.joint(name).qposadr[0]]) for name in movable}
        pair_file = "collisions_omnihand.json" if self.hand_joint_names else "collisions.json"
        spec = RobotSpec(
            self.robot_urdf_path,
            tuple(self.arm_actuators.values()),
            {name: config["link_name"] for name, config in self.manipulator_config.items()},
            {key: tuple(names) for key, names in self.arm_joints.items()},
            self.position_limits,
            initial,
            str(Path(ASSET_PATH) / "agibot/x2" / pair_file),
            self.dt,
        )
        self.backend = create_backend(self.ik_backend, spec)
        self.robot = self.backend.robot
        for joint in arm_names:
            self.robot.set_velocity_limit(joint, min(self.arm_velocity_limit, velocities[joint]))
        self.hand_command_offsets = {
            actuator: self.robot.get_joint_offset(self.mj_model.joint(self.mj_model.actuator_trnid[actuator, 0]).name)
            for actuators in self.hand_actuators.values()
            for actuator in actuators
        }
        self._configure_hand_couplings()
        self._update_robot_state()
        for name, config in self.manipulator_config.items():
            xyz, quaternion = self._get_link_pose(config["link_name"])
            measured = tf.quaternion_matrix(quaternion)
            measured[:3, 3] = xyz
            if not np.allclose(measured, self.robot.get_T_world_frame(config["link_name"]), atol=1e-4, rtol=0):
                raise ValueError("MuJoCo and IK hand-center frames do not match; check the IK URDF")
            self.effector_task[name] = PoseTarget(measured)
        self.collision_checker = CollisionChecker(self.robot)
        self.collision_pair_count = self.collision_checker.pair_count
        self._check_collision_clearance()
        self.collision_constraint = LinearConstraint(np.zeros((1, self.robot.model.nq)), np.ones(1))
        reserve = min(0.002, (self.collision_trigger - self.collision_margin) / 2)
        self._solver_collision_margin = self.collision_margin + reserve
        self._update_collision_constraint(self.collision_constraint)
        self._update_mocap_target()

    def _command_bounds(self, homing=False):
        result = {}
        for actuator, name in self.arm_actuators.items():
            previous = float(self.mj_data.ctrl[actuator])
            lower, upper = self.position_limits[name]
            velocity = self.robot.model.velocity_limit[self.robot.get_joint_v_offset(name)]
            if homing:
                velocity = min(velocity, self.return_velocity_limit)
            step = velocity * self.dt
            if self.active[f"{name.split('_', 1)[0]}_arm"]:
                bounds = max(lower, previous - step), min(upper, previous + step)
            else:
                bounds = previous, previous
            if bounds[0] > bounds[1] or not lower <= previous <= upper:
                raise RuntimeError(f"Previous joint command outside position limits: {name}")
            result[name] = bounds
        return result

    def _make_problem(self, bounds, joint_target=None):
        robot = self.robot
        measured = {n: robot.get_joint(n) for n in self.backend.spec.joints}
        previous = {name: float(self.mj_data.ctrl[i]) for i, name in self.arm_actuators.items()}
        distances = self.collision_checker.clearances(self.collision_trigger)
        self._update_collision_constraint(self.collision_constraint)
        recovering = np.any(distances < self.collision_margin - 1e-6)
        recovery = self._add_recovery_constraint(distances) if recovering else None
        offsets = np.array([robot.get_joint_offset(n) for n in self.backend.spec.joints])
        fixed = robot.state.q.copy()
        fixed[offsets] = 0

        def controlled(constraint):
            return LinearConstraint(constraint.A[:, offsets].copy(), constraint.b - constraint.A @ fixed)

        empty = LinearConstraint(np.zeros((0, len(offsets))), np.zeros(0))
        return IKProblem(
            measured,
            previous,
            {key: task.T_world_frame.copy() for key, task in self.effector_task.items()},
            frozenset(n for key, names in self.arm_joints.items() if self.active[key] for n in names),
            np.array([bounds[n][0] for n in self.backend.spec.joints]),
            np.array([bounds[n][1] for n in self.backend.spec.joints]),
            controlled(recovery or self.collision_constraint),
            controlled(self.collision_constraint) if recovering else empty,
            self.dt,
            joint_target,
        )

    def _propose_checked(self, measured, bounds, joint_target=None):
        problem = self._make_problem(bounds, joint_target)
        previous = measured.copy()
        for actuator, name in self.arm_actuators.items():
            previous[self.robot.get_joint_offset(name)] = self.mj_data.ctrl[actuator]
        for attempt in range(2):
            proposal = self.backend.propose(problem)
            solution = measured.copy()
            for name in self.backend.spec.joints:
                value = proposal.positions[name]
                if not np.isfinite(value):
                    self.backend.reject(proposal)
                    raise RuntimeError("Non-finite IK proposal")
                lower, upper = bounds[name]
                solution[self.robot.get_joint_offset(name)] = np.clip(value, lower, upper)
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                self.robot.state.q = previous + scale * (solution - previous)
                try:
                    self._validate_collision_motion(measured)
                    if self.hand_command_offsets:
                        self._validate_collision_motion(measured, self.mj_data.ctrl)
                except RuntimeError:
                    continue
                self._pending_proposal = proposal
                return
            self.backend.reject(proposal)
            self.robot.state.q = measured.copy()
            self.robot.update_kinematics()
            if attempt == 0:
                self.backend.reset()
        raise RuntimeError("No collision-free IK command found")

    def _update_ik(self):
        if self._pending_proposal is not None:
            self.backend.reject(self._pending_proposal)
            self._pending_proposal = None
        self._update_robot_state()
        self.collision_checker.clear_cache()
        self._ik_valid = False
        self.status = "hold"
        stream_fresh = self._xr_stream_is_fresh()
        self._stream_fresh = stream_fresh
        if self.xr_client.get_button_state_by_name("Y"):
            self._heading_captured = False
            self._update_return_to_default(stream_fresh)
            return
        for name, config in self.manipulator_config.items():
            grip = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[name] = np.isfinite(grip) and grip > DEFAULTS.grip_threshold
        any_active = any(self.active.values())
        if not any_active:
            self._heading_captured = False
        elif stream_fresh and not self._heading_captured:
            self._heading_captured = self._capture_control_heading()
        tracking_valid = stream_fresh
        for name, config in self.manipulator_config.items():
            if not self.active[name]:
                self._reset_clutch_reference(name)
                self.backend.reset(frozenset(self.arm_joints[name]))
                continue
            pose = self._read_valid_xr_pose(config["pose_source"])
            if not stream_fresh or pose is None or not self._heading_captured:
                tracking_valid = False
                continue
            if self.ref_ee_xyz[name] is None:
                self.ref_ee_xyz[name], self.ref_ee_quat[name] = self._get_link_pose(config["link_name"])
            delta_xyz, delta_rot = self._process_xr_pose(pose, name)
            xyz, quat = apply_delta_pose(self.ref_ee_xyz[name], self.ref_ee_quat[name], delta_xyz, delta_rot)
            target = tf.quaternion_matrix(quat)
            target[:3, 3] = xyz
            self.effector_task[name].T_world_frame = target
        if not tracking_valid:
            for name in self.manipulator_config:
                self._reset_clutch_reference(name)
            self.backend.reset()
            return
        if not any_active:
            self.backend.reset()
            return
        measured = self.robot.state.q.copy()
        try:
            self._propose_checked(measured, self._command_bounds())
            self._ik_valid = True
            self.status = "tracking"
        except (RuntimeError, ValueError) as exc:
            self.robot.state.q = measured
            self.robot.update_kinematics()
            self.status = f"hold: {exc}"
            print(f"X2 IK failed; holding previous commands: {exc}")

    def _update_return_to_default(self, stream_fresh):
        self.backend.reset()
        for name in self.manipulator_config:
            self.active[name] = True
            self._reset_clutch_reference(name)
        if not stream_fresh or any(
            self._read_valid_xr_pose(config["pose_source"]) is None for config in self.manipulator_config.values()
        ):
            return
        previous = self.robot.state.q.copy()
        home = self.mj_model.key("home").qpos
        try:
            bounds = self._command_bounds(homing=True)
            for name, (lower, upper) in bounds.items():
                target = home[self.mj_model.joint(name).qposadr[0]]
                self.robot.set_joint(name, np.clip(target, lower, upper))
            try:
                self._validate_collision_motion(previous)
                if self.hand_command_offsets:
                    self._validate_collision_motion(previous, self.mj_data.ctrl)
            except RuntimeError:
                self.robot.state.q = previous.copy()
                self.robot.update_kinematics()
                self._solve_return_avoidance(previous, home, bounds)
            self._ik_valid = True
            self.status = "homing"
        except (RuntimeError, ValueError) as exc:
            self.robot.state.q = previous
            self.status = f"hold: {exc}"
            print(f"X2 return to default paused; holding previous commands: {exc}")
        finally:
            self.robot.update_kinematics()
            for name, config in self.manipulator_config.items():
                self.effector_task[name].T_world_frame = self.robot.get_T_world_frame(config["link_name"])

    def _solve_return_avoidance(self, previous, home, bounds):
        target = {name: float(home[self.mj_model.joint(name).qposadr[0]]) for name in self.backend.spec.joints}
        self._propose_checked(previous, bounds, target)

    def _project_hand_commands(self, configuration, commands):
        for actuator, offset in self.hand_command_offsets.items():
            configuration[offset] = commands[actuator]
        for target, source, coefficients in self.hand_couplings:
            configuration[target] = polyval(configuration[source], coefficients)

    def _validate_collision_motion(self, measured, hand_commands=None, previous_commands=None):
        """Check sampled joint-space motion to the exact outgoing arm commands.

        Released arms retain their servo targets, which can differ from measured
        positions. Check both measured-to-target and old-to-new target segments.
        These are discrete checks, not continuous collision detection.
        """
        candidate = self.robot.state.q.copy()
        target, held = candidate.copy(), measured.copy()
        previous_commands = self.mj_data.ctrl if previous_commands is None else previous_commands
        for actuator, joint in self.arm_actuators.items():
            offset = self.robot.get_joint_offset(joint)
            held[offset] = previous_commands[actuator]
            if not self.active.get(f"{joint.split('_', 1)[0]}_arm", False):
                target[offset] = held[offset]
            else:
                target[offset] = np.clip(target[offset], *self.mj_model.actuator_ctrlrange[actuator])
        if hand_commands is not None:
            self._project_hand_commands(target, hand_commands)
            self._project_hand_commands(held, previous_commands)
        try:
            if not all(np.all(np.isfinite(q)) for q in (measured, held, target)):
                raise RuntimeError("non-finite X2 collision configuration")
            self.robot.state.q = measured
            measured_distances = self.collision_checker.clearances(self.collision_margin)
            recovering = np.any(measured_distances < self.collision_margin - 1e-6)
            self.robot.state.q = held
            held_distances = self.collision_checker.baseline(self.collision_margin)
            for start, distances, measured_path in (
                (measured, measured_distances, True),
                (held, held_distances, False),
            ):
                # Servo targets can be ahead of or behind the measured arm.
                # Their interpolation may enter an already measured violation,
                # but must not exceed either existing deficit.
                floor = np.minimum(distances, self.collision_margin)
                if not measured_path:
                    floor = np.minimum(floor, measured_distances)
                # Unsigned bounds cannot establish that penetration has not
                # deepened. A held servo target can penetrate even when the
                # measured pose is clear, so check this segment's floor too.
                use_bounds = not recovering and self.collision_checker.can_bound and np.all(floor > 0)
                # Initial distances establish the floor; sample the coupled
                # command path rather than a chord between passive endpoints.
                for configuration in self._collision_motion_samples(start, target, hand_commands is not None):
                    self.robot.state.q = configuration
                    values = self.collision_checker.clearance_bounds(floor) if use_bounds else None
                    if values is None or np.any(values < floor):
                        values = self.collision_checker.clearances(self.collision_margin)
                    if np.any(values < floor - 1e-6):
                        index = int(np.argmin(values - floor))
                        a, b = map(int, self.collision_checker.pair_indices[index])
                        objects = self.robot.collision_model.objects
                        raise RuntimeError(
                            f"X2 self-collision clearance violated: {objects[a].name} / "
                            f"{objects[b].name}: {values[index]:.6f} m < {floor[index]:.6f} m"
                        )
                    if measured_path:
                        # Below the margin, require monotonic separation over
                        # the measured-to-command path, not just safe endpoints.
                        floor = np.maximum(floor, np.minimum(values, self.collision_margin))
                if measured_path:
                    violating = measured_distances < self.collision_margin - 1e-6
                    if np.any(violating) and not np.any(
                        np.minimum(values, self.collision_margin)[violating] > measured_distances[violating] + 1e-8
                    ):
                        raise RuntimeError("X2 self-collision recovery made no progress")
        finally:
            self.robot.state.q = candidate
            self.robot.update_kinematics()

    def _xr_stream_is_fresh(self):
        timestamp = self.xr_client.get_timestamp_ns()
        now = time.monotonic()
        if timestamp <= 0 or timestamp < self._last_xr_timestamp_ns:
            # A reset/backward timestamp must advance again before resuming.
            self._last_xr_timestamp_ns = timestamp
            self._last_xr_update_at = None
            return False
        if timestamp > self._last_xr_timestamp_ns:
            self._last_xr_timestamp_ns = timestamp
            self._last_xr_update_at = now
        return self._last_xr_update_at is not None and now - self._last_xr_update_at <= self.XR_TRACKING_TIMEOUT_S

    def _capture_control_heading(self):
        """Align tracking axes once at grip engagement, keeping gravity vertical."""
        pose = self._read_valid_xr_pose("headset")
        if pose is None:
            return False
        rotation = tf.quaternion_matrix(np.roll(pose[3:], 1))[:3, :3]
        # Native XR forward is -Z, with +Y up. Ignore head pitch and roll.
        forward = -rotation[:, 2]
        forward[1] = 0.0
        length = np.linalg.norm(forward)
        if length < 1e-6:
            return False
        forward /= length
        up = np.array([0.0, 1.0, 0.0])
        heading = np.column_stack((np.cross(forward, up), up, -forward))
        self.R_headset_world = R_HEADSET_TO_WORLD @ heading.T
        return True

    def _read_valid_xr_pose(self, source):
        pose = np.asarray(self.xr_client.get_pose_by_name(source), dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            return None
        pose = pose.copy()
        scale = np.max(np.abs(pose[3:]))
        if scale == 0:
            return None
        quaternion = pose[3:] / scale
        norm = np.linalg.norm(quaternion)
        if scale < 1e-6 / norm:
            return None
        pose[3:] = quaternion / norm
        return pose

    def _send_command(self):
        previous_commands = self.mj_data.ctrl.copy()
        if self._ik_valid:
            for i, joint_name in self.arm_actuators.items():
                side = joint_name.split("_", 1)[0]
                if self.active[f"{side}_arm"]:
                    self.mj_data.ctrl[i] = np.clip(
                        self.robot.get_joint(joint_name), *self.mj_model.actuator_ctrlrange[i]
                    )
        if self._stream_fresh:
            for side, actuators in self.hand_actuators.items():
                if not actuators:
                    continue
                trigger = self.xr_client.get_key_value_by_name(f"{side}_trigger")
                if not np.isfinite(trigger):
                    continue
                trigger = np.clip(trigger, 0.0, 1.0)
                indices = list(actuators)
                limits = self.mj_model.actuator_ctrlrange[indices]
                target = np.clip(trigger * np.array(list(actuators.values())), limits[:, 0], limits[:, 1])
                delta = target - self.mj_data.ctrl[indices]
                moving = np.abs(delta) > 0
                if np.any(moving):
                    max_delta = self.dt * np.array([self.hand_velocity_limits[i] for i in indices])
                    # Move the whole hand by a common fraction of the remaining
                    # distance. Independent joint ramps can tuck the thumb into
                    # fingers that have not caught up during trigger reversals.
                    fraction = min(1.0, np.min(max_delta[moving] / np.abs(delta[moving])))
                    self.mj_data.ctrl[indices] += fraction * delta
        if self.hand_command_offsets and not np.array_equal(self.mj_data.ctrl, previous_commands):
            self._validate_combined_commands(previous_commands)
        if self._pending_proposal is not None:
            proposal, self._pending_proposal = self._pending_proposal, None
            if self._ik_valid:
                self.backend.accept(
                    proposal, {name: float(self.mj_data.ctrl[i]) for i, name in self.arm_actuators.items()}
                )
            else:
                self.backend.reject(proposal)
                self.status = "hold: combined collision check rejected commands"
        self._update_ik_viz()

    def _validate_combined_commands(self, previous_commands):
        """Validate combined arm/finger commands, using the MJCF coupling law."""
        robot = self.robot
        saved = robot.state.q.copy()
        measured = self._measured_configuration()

        def check(commands):
            candidate = measured.copy()
            for actuator, joint in self.arm_actuators.items():
                candidate[robot.get_joint_offset(joint)] = commands[actuator]
            robot.state.q = candidate
            self._validate_collision_motion(measured, commands, previous_commands)

        try:
            try:
                check(self.mj_data.ctrl)
            except RuntimeError as exc:
                hand_changed = any(self.mj_data.ctrl[i] != previous_commands[i] for i in self.hand_command_offsets)
                for actuator in self.hand_command_offsets:
                    self.mj_data.ctrl[actuator] = previous_commands[actuator]
                # Permit a separately valid arm escape while rejecting finger
                # motion; a persistent trigger must not prevent arm recovery.
                held = "hand"
                arm_is_valid = False
                if hand_changed:
                    try:
                        check(self.mj_data.ctrl)
                    except RuntimeError:
                        pass  # The arm proposal must also be rolled back.
                    else:
                        arm_is_valid = True
                if not arm_is_valid:
                    self.mj_data.ctrl[:] = previous_commands
                    self._ik_valid = False
                    self.status = "hold: combined collision check rejected commands"
                    saved = measured
                    held = "arm and hand"
                print(f"X2 collision check failed; holding {held} commands: {exc}")
        finally:
            robot.state.q = saved
            robot.update_kinematics()

    def step_control(self):
        """Advance one configured control period, including its physics substeps."""
        self._update_ik()
        self._update_mocap_target()
        self._send_command()
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=self.physics_substeps)

    def close(self):
        """Release every resource, including when another cleanup raises."""
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        with ExitStack() as cleanup:
            for name in ("xr_client", "backend", "_ik_viewer"):
                resource = getattr(self, name, None)
                if resource is not None:
                    cleanup.callback(resource.close)

    def run(self):
        try:
            with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                viewer.cam.distance = 2.5
                viewer.cam.lookat = [0, 0, 0.7]
                while viewer.is_running() and not self._stop_event.is_set():
                    start = time.monotonic()
                    with viewer.lock():
                        self.step_control()
                    viewer.sync(state_only=True)
                    time.sleep(max(0.0, self.dt - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
