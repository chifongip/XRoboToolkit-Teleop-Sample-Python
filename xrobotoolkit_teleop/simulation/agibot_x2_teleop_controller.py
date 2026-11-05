"""Fixed-body X2 arm teleoperation; no robot hardware connections."""

import json
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer
from numpy.polynomial.polynomial import polyval

from xrobotoolkit_teleop.simulation.agibot_x2_hand_config import HAND_CLOSED
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController
from xrobotoolkit_teleop.utils.collision_checks import CollisionChecker
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose
from xrobotoolkit_teleop.utils.mujoco_utils import calc_placo_q_from_mujoco_qpos
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class AgibotX2TeleopController(MujocoTeleopController):
    """Two controller poses drive hand-center frames; released grips hold commands."""

    XR_TRACKING_TIMEOUT_S = 0.25
    COLLISION_CHECK_STEP_RAD = 0.02
    MAX_COLLISION_SAMPLES = 4096
    HAND_CLOSED = HAND_CLOSED

    def __init__(
        self,
        xml_path,
        robot_urdf_path,
        scale_factor=1.0,
        visualize_placo=False,
        arm_velocity_limit=8.0,
        hand_velocity_limit=2.0,
        collision_margin=0.02,
        collision_trigger=0.08,
        control_frequency=50.0,
    ):
        if not np.isfinite(control_frequency) or control_frequency <= 0:
            raise ValueError("control_frequency must be finite and positive")
        control_dt = 1.0 / control_frequency
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_frequency must produce a finite positive control period")
        self.control_frequency = control_frequency
        for name, value in (("arm_velocity_limit", arm_velocity_limit), ("hand_velocity_limit", hand_velocity_limit)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.arm_velocity_limit = arm_velocity_limit
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
        try:
            super().__init__(
                xml_path=xml_path,
                robot_urdf_path=robot_urdf_path,
                manipulator_config=config,
                floating_base=False,
                scale_factor=scale_factor,
                visualize_placo=visualize_placo,
                dt=control_dt,
            )
            # Require an advancing stream, not just cached poses from the SDK.
            self._last_xr_timestamp_ns = self.xr_client.get_timestamp_ns()
            self._last_xr_update_at = None
        except Exception:
            if hasattr(self, "xr_client"):
                self.xr_client.close()
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

    def _placo_setup(self):
        super()._placo_setup()
        model_names = set(self.placo_robot.joint_names())
        arm_names = {name for name in model_names if name.startswith(("left_", "right_"))}
        self._configure_actuators(arm_names)
        mj_names = {self.mj_model.joint(i).name for i in range(self.mj_model.njnt)}
        if model_names != mj_names:
            raise ValueError("Omnihand Pro requires x2_omnihand_ik.urdf with matching articulated hand joints")
        self.hand_joint_names = sorted(model_names - arm_names)
        for joint in self.hand_joint_names:
            self.solver.mask_dof(joint)
            # These DOFs are observations, including passive coupled joints.
            # MuJoCo enforces their limits; arm IK must remain feasible when
            # measured fingers slightly overshoot a limit during dynamics.
            self.placo_robot.set_joint_limits(joint, -1e6, 1e6)
        self.hand_command_offsets = {
            actuator: self.placo_robot.get_joint_offset(
                self.mj_model.joint(self.mj_model.actuator_trnid[actuator, 0]).name
            )
            for actuators in self.hand_actuators.values()
            for actuator in actuators
        }
        self._configure_hand_couplings()
        self.arm_joints = {
            f"{side}_arm": sorted(name for name in arm_names if name.startswith(f"{side}_"))
            for side in ("left", "right")
        }
        if any(len(names) != 7 for names in self.arm_joints.values()):
            raise ValueError("X2 requires seven joints per arm")
        self._update_robot_state()
        for config in self.manipulator_config.values():
            name = config["link_name"]
            position, quaternion = self._get_link_pose(name)
            ik_pose = self.placo_robot.get_T_world_frame(name)
            if not (
                np.allclose(position, ik_pose[:3, 3], atol=1e-4, rtol=0)
                and np.allclose(tf.quaternion_matrix(quaternion)[:3, :3], ik_pose[:3, :3], atol=1e-3, rtol=0)
            ):
                raise ValueError(
                    "MuJoCo and IK hand-center frames do not match; Omnihand Pro requires x2_omnihand_ik.urdf"
                )
        self.sync_end_effector_poses_to_placo_tasks()
        for name, task in self.effector_task.items():
            # Favor hand-center tracking over the generic manipulability objective.
            task.configure(name, "soft", 10.0)
        self.solver.enable_joint_limits(True)
        # The requested speed can reduce, but never raise, each URDF limit.
        for joint in arm_names:
            offset = self.placo_robot.get_joint_v_offset(joint)
            urdf_limit = self.placo_robot.model.velocity_limit[offset]
            self.placo_robot.set_velocity_limit(joint, min(self.arm_velocity_limit, urdf_limit))
        self.solver.enable_velocity_limits(True)
        # Explicitly load the same pairs for custom URDFs, too. Missing collision
        # shapes must fail setup rather than silently disable protection.
        pair_file = "collisions_omnihand.json" if self.hand_joint_names else "collisions.json"
        pair_path = Path(ASSET_PATH) / "agibot/x2" / pair_file
        pairs = json.loads(pair_path.read_text())
        urdf = ET.parse(self.robot_urdf_path).getroot()
        for name in {name for pair in pairs for name in pair}:
            if urdf.find(f"link[@name='{name}']/collision/geometry") is None:
                raise ValueError(f"X2 IK URDF requires collision geometry for {name}")
        self.placo_robot.load_collision_pairs(str(pair_path))
        self.collision_checker = CollisionChecker(self.placo_robot)
        self.collision_pair_count = self.collision_checker.pair_count
        self._check_collision_clearance()
        self.collision_constraint = self.solver.add_avoid_self_collisions_constraint()
        self.collision_constraint.configure("x2_self_collision", "hard")
        # Reserve up to 2 mm for nonlinear distance changes after the QP's
        # linearized step. The command check still enforces the requested margin.
        reserve = min(0.002, (self.collision_trigger - self.collision_margin) / 2)
        self.collision_constraint.self_collisions_margin = self.collision_margin + reserve
        self.collision_constraint.self_collisions_trigger = self.collision_trigger
        self.posture_task = self.solver.add_joints_task()
        self.posture_task.set_joints({name: self.placo_robot.get_joint(name) for name in arm_names})
        self.posture_task.configure("arm_posture", "soft", 1e-4)
        self._update_mocap_target()

    def _configure_hand_couplings(self):
        """Validate and order passive joints so each is projected exactly once."""
        model, robot = self.mj_model, self.placo_robot
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

    def _update_ik(self):
        self._update_robot_state()
        self.collision_checker.clear_cache()
        self._ik_valid = False
        stream_fresh = self._xr_stream_is_fresh()
        self._stream_fresh = stream_fresh
        tracking_valid = stream_fresh
        for name, config in self.manipulator_config.items():
            grip = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[name] = np.isfinite(grip) and grip > 0.9
            for joint in self.arm_joints[name]:
                if self.active[name]:
                    self.solver.unmask_dof(joint)
                else:
                    self.solver.mask_dof(joint)
            if not self.active[name]:
                self.ref_ee_xyz[name] = None
                self.ref_controller_xyz[name] = None
                continue

            pose = self._read_valid_xr_pose(config["pose_source"])
            if not stream_fresh or pose is None:
                # Drop the reference so resumed tracking also starts without a jump.
                self.ref_ee_xyz[name] = None
                self.ref_controller_xyz[name] = None
                tracking_valid = False
                continue
            if self.ref_ee_xyz[name] is None:
                self.ref_ee_xyz[name], self.ref_ee_quat[name] = self._get_link_pose(config["link_name"])
            delta_xyz, delta_rot = self._process_xr_pose(pose, name)
            xyz, quat = apply_delta_pose(self.ref_ee_xyz[name], self.ref_ee_quat[name], delta_xyz, delta_rot)
            target = tf.quaternion_matrix(quat)
            target[:3, 3] = xyz
            self.effector_task[name].T_world_frame = target

        # Process both clutches before holding commands for a tracking failure.
        if not tracking_valid:
            # Both arms hold, so discard motion from both controllers on every
            # invalid cycle. Recovery rebases each hand center at its measured pose.
            for name in self.manipulator_config:
                self.ref_ee_xyz[name] = None
                self.ref_ee_quat[name] = None
                self.ref_controller_xyz[name] = None
                self.ref_controller_quat[name] = None
            return
        if not any(self.active.values()):
            return
        previous = self.placo_robot.state.q.copy()
        recovery_constraint = None
        try:
            distances = self.collision_checker.distances()
            recovering = np.any(distances < self.collision_margin - 1e-6)
            if recovering:
                # Restoring the full margin in one timestep may be infeasible
                # under joint speed limits. Favor separation, then independently
                # reject steps that worsen an existing violation or create one.
                self.collision_constraint.configure("x2_self_collision", "soft", 1e6)
                recovery_constraint = self._add_recovery_constraint(distances)
            self.solver.solve(True)
            if not np.all(np.isfinite(self.placo_robot.state.q)):
                raise RuntimeError("non-finite IK solution")
            solution = self.placo_robot.state.q.copy()
            # Linearized constraints can slightly worsen another overlapping
            # primitive pair. Shorten recovery steps until the nonlinear checks pass.
            scales = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125) if recovering else (1.0,)
            for scale in scales:
                self.placo_robot.state.q = previous + scale * (solution - previous)
                try:
                    self._validate_collision_motion(previous)
                    break
                except RuntimeError:
                    if scale == scales[-1]:
                        raise
            self._ik_valid = True
        except RuntimeError as exc:
            self.placo_robot.state.q = previous
            self.placo_robot.update_kinematics()
            print(f"X2 IK failed; holding previous commands: {exc}")
        finally:
            self.collision_constraint.configure("x2_self_collision", "hard")
            if recovery_constraint is not None:
                self.solver.remove_constraint(recovery_constraint)

    def _add_recovery_constraint(self, distances):
        """Keep each nearby pair from worsening while soft avoidance separates it.

        Finite differences include signed primitive penetration, where separation of
        several pairs must be coordinated. Only gripped arms may move.
        """
        robot = self.placo_robot
        previous = robot.state.q.copy()
        nearby = distances < self.collision_trigger
        jacobian = np.zeros((np.count_nonzero(nearby), len(previous)))
        epsilon = 1e-5
        try:
            for name, joints in self.arm_joints.items():
                if not self.active[name]:
                    continue
                for joint in joints:
                    offset = robot.get_joint_offset(joint)
                    plus, minus = previous.copy(), previous.copy()
                    plus[offset] += epsilon
                    minus[offset] -= epsilon
                    robot.state.q = plus
                    d_plus = self.collision_checker.distances()[nearby]
                    robot.state.q = minus
                    d_minus = self.collision_checker.distances()[nearby]
                    jacobian[:, offset] = (d_plus - d_minus) / (2 * epsilon)
        finally:
            robot.state.q = previous
            robot.update_kinematics()
        # d(q + dq) >= min(d(q), margin), expressed as A q_new <= b.
        current = distances[nearby]
        floor = np.minimum(current, self.collision_margin)
        constraint = self.solver.add_joint_space_half_spaces_constraint(
            -jacobian, current - floor - jacobian @ previous
        )
        constraint.configure("x2_recovery_clearance", "hard")
        return constraint

    def _check_collision_clearance(self):
        if np.any(self.collision_checker.distances() < self.collision_margin - 1e-6):
            raise RuntimeError("X2 self-collision clearance violated")

    def _project_hand_commands(self, configuration, commands):
        for actuator, offset in self.hand_command_offsets.items():
            configuration[offset] = commands[actuator]
        for target, source, coefficients in self.hand_couplings:
            configuration[target] = polyval(configuration[source], coefficients)

    def _collision_motion_samples(self, start, target, coupled_hands):
        """Sample coupled finger motion while preserving measured tracking error.

        A passive joint follows its polynomial plus a linearly decaying initial
        tracking error. Derivative bounds keep every joint's sample spacing
        within COLLISION_CHECK_STEP_RAD, including nonlinear passive motion.
        """
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
            raise RuntimeError("non-finite X2 collision configuration")
        delta = target - start
        rates = np.abs(delta)
        max_abs = np.maximum(np.abs(start), np.abs(target))
        corrections = []
        if coupled_hands:
            for passive, source, coefficients in self.hand_couplings:
                initial_error = start[passive] - polyval(start[source], coefficients)
                final_error = target[passive] - polyval(target[source], coefficients)
                slope = polyval(max_abs[source], np.arange(1, len(coefficients)) * np.abs(coefficients[1:]))
                rates[passive] = slope * rates[source] + abs(final_error - initial_error)
                max_abs[passive] = polyval(max_abs[source], np.abs(coefficients)) + max(
                    abs(initial_error), abs(final_error)
                )
                corrections.append((passive, source, coefficients, initial_error, final_error))
        max_rate = float(np.max(rates[7:]))
        if not np.isfinite(max_rate) or max_rate > self.MAX_COLLISION_SAMPLES * self.COLLISION_CHECK_STEP_RAD:
            raise RuntimeError("X2 collision motion exceeds sampling budget")
        steps = max(1, int(np.ceil(max_rate / self.COLLISION_CHECK_STEP_RAD)))
        fractions = np.linspace(0, 1, steps + 1)[1:]
        samples = start + fractions[:, None] * delta
        for passive, source, coefficients, initial_error, final_error in corrections:
            samples[:, passive] = (
                polyval(samples[:, source], coefficients) + (1 - fractions) * initial_error + fractions * final_error
            )
        if not np.all(np.isfinite(samples)):
            raise RuntimeError("non-finite X2 collision motion")
        return samples

    def _validate_collision_motion(self, measured, hand_commands=None, previous_commands=None):
        """Check sampled joint-space motion to the exact outgoing arm commands.

        Released arms retain their servo targets, which can differ from measured
        positions. Check both measured-to-target and old-to-new target segments.
        These are discrete checks, not continuous collision detection.
        """
        candidate = self.placo_robot.state.q.copy()
        target, held = candidate.copy(), measured.copy()
        previous_commands = self.mj_data.ctrl if previous_commands is None else previous_commands
        for actuator, joint in self.arm_actuators.items():
            offset = self.placo_robot.get_joint_offset(joint)
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
            self.placo_robot.state.q = measured
            measured_distances = self.collision_checker.distances()
            recovering = np.any(measured_distances < self.collision_margin - 1e-6)
            self.placo_robot.state.q = held
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
                    self.placo_robot.state.q = configuration
                    values = self.collision_checker.clearance_bounds(floor) if use_bounds else None
                    if values is None or np.any(values < floor):
                        values = self.collision_checker.distances()
                    if np.any(values < floor - 1e-6):
                        index = int(np.argmin(values - floor))
                        a, b = map(int, self.collision_checker.pair_indices[index])
                        objects = self.placo_robot.collision_model.objects
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
            self.placo_robot.state.q = candidate
            self.placo_robot.update_kinematics()

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

    def _read_valid_xr_pose(self, source):
        pose = np.asarray(self.xr_client.get_pose_by_name(source), dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[3:]) < 1e-6:
            return None
        pose = pose.copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        return pose

    def _send_command(self):
        previous_commands = self.mj_data.ctrl.copy()
        if self._ik_valid:
            for i, joint_name in self.arm_actuators.items():
                side = joint_name.split("_", 1)[0]
                if self.active[f"{side}_arm"]:
                    self.mj_data.ctrl[i] = np.clip(
                        self.placo_robot.get_joint(joint_name), *self.mj_model.actuator_ctrlrange[i]
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
        if self.visualize_placo:
            self._update_placo_viz()

    def _validate_combined_commands(self, previous_commands):
        """Validate combined arm/finger commands, using the MJCF coupling law."""
        robot = self.placo_robot
        saved = robot.state.q.copy()
        measured = calc_placo_q_from_mujoco_qpos(self.mj_model, robot, self.mj_data.qpos)

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
            self._stop_event.set()
            self.xr_client.close()
