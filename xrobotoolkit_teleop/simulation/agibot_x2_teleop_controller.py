"""Fixed-body X2 arm teleoperation; no robot hardware connections."""

import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class AgibotX2TeleopController(MujocoTeleopController):
    """Two controller poses drive hand-center frames; released grips hold commands."""

    XR_TRACKING_TIMEOUT_S = 0.25
    # Left Omnihand Pro targets; only thumb roll/abduction change sign on the right.
    HAND_CLOSED = {
        "thumb_roll": -0.6,
        "thumb_abad": 0.6,
        # Deep MCP flexion pushes the thumb into the curled fingers. Keep the
        # proximal segment outside them and curl primarily at the distal joints.
        "thumb_mcp": -0.05,
        "thumb_pip": -0.6,
        "index_abad": 0.0,
        "index_mcp": 1.0,
        "index_pip": 1.0,
        "middle_abad": 0.0,
        "middle_mcp": 1.0,
        "middle_pip": 1.0,
        "ring_mcp": 1.0,
        "pinky_mcp": 1.0,
    }

    def __init__(
        self,
        xml_path,
        robot_urdf_path,
        scale_factor=1.0,
        visualize_placo=False,
        arm_velocity_limit=8.0,
        hand_velocity_limit=2.0,
    ):
        for name, value in (("arm_velocity_limit", arm_velocity_limit), ("hand_velocity_limit", hand_velocity_limit)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.arm_velocity_limit = arm_velocity_limit
        self.hand_velocity_limit = hand_velocity_limit
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
                dt=0.01,
            )
            # Require an advancing stream, not just cached poses from the SDK.
            self._last_xr_timestamp_ns = self.xr_client.get_timestamp_ns()
            self._last_xr_update_at = None
        except Exception:
            if hasattr(self, "xr_client"):
                self.xr_client.close()
            raise

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
        arm_names = set(self.placo_robot.joint_names())
        self._configure_actuators(arm_names)
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
        self.posture_task = self.solver.add_joints_task()
        self.posture_task.set_joints({name: self.placo_robot.get_joint(name) for name in arm_names})
        self.posture_task.configure("arm_posture", "soft", 1e-4)
        self._update_mocap_target()

    def _update_ik(self):
        self._update_robot_state()
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
        try:
            self.solver.solve(True)
            if not np.all(np.isfinite(self.placo_robot.state.q)):
                raise RuntimeError("non-finite IK solution")
            self._ik_valid = True
        except RuntimeError as exc:
            self.placo_robot.state.q = previous
            self.placo_robot.update_kinematics()
            print(f"X2 IK failed; holding previous commands: {exc}")

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
        if self.visualize_placo:
            self._update_placo_viz()

    def step_control(self):
        """Advance one 10 ms control cycle, including ten physics substeps."""
        self._update_ik()
        self._update_mocap_target()
        self._send_command()
        substeps = round(self.dt / self.mj_model.opt.timestep)
        if substeps < 1 or not np.isclose(substeps * self.mj_model.opt.timestep, self.dt):
            raise ValueError("Physics timestep must divide the 10 ms control period")
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=substeps)

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
