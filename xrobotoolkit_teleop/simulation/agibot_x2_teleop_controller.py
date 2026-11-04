"""Fixed-body X2 arm teleoperation; no robot hardware connections."""

import time

import mujoco
import numpy as np
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose
from xrobotoolkit_teleop.utils.mujoco_utils import calc_mujoco_ctrl_from_qpos, calc_mujoco_qpos_from_placo_q


class AgibotX2TeleopController(MujocoTeleopController):
    """Two controller poses drive the wrists; released grips hold joint commands."""

    XR_TRACKING_TIMEOUT_S = 0.25

    def __init__(self, xml_path, robot_urdf_path, scale_factor=1.0, visualize_placo=False, velocity_limit=8.0):
        if not np.isfinite(velocity_limit) or velocity_limit <= 0:
            raise ValueError("velocity_limit must be finite and positive")
        self.velocity_limit = velocity_limit
        config = {
            f"{side}_arm": {
                "link_name": f"{side}_wrist_roll_link",
                "pose_source": f"{side}_controller",
                "control_trigger": f"{side}_grip",
                "vis_target": f"{side}_target",
            }
            for side in ("left", "right")
        }
        self._ik_valid = False
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

    def _placo_setup(self):
        super()._placo_setup()
        mj_names = {self.mj_model.joint(i).name for i in range(self.mj_model.njnt)}
        if len(mj_names) != 14 or mj_names != set(self.placo_robot.joint_names()):
            raise ValueError("X2 requires matching fixed-body models with exactly 14 arm joints")
        self.arm_joints = {
            f"{side}_arm": sorted(name for name in mj_names if name.startswith(f"{side}_"))
            for side in ("left", "right")
        }
        if any(len(names) != 7 for names in self.arm_joints.values()):
            raise ValueError("X2 requires seven joints per arm")
        self._update_robot_state()
        self.sync_end_effector_poses_to_placo_tasks()
        for name, task in self.effector_task.items():
            # Favor wrist tracking over the generic manipulability objective.
            task.configure(name, "soft", 10.0)
        self.solver.enable_joint_limits(True)
        # The requested speed can reduce, but never raise, each URDF limit.
        for joint in mj_names:
            offset = self.placo_robot.get_joint_v_offset(joint)
            urdf_limit = self.placo_robot.model.velocity_limit[offset]
            self.placo_robot.set_velocity_limit(joint, min(self.velocity_limit, urdf_limit))
        self.solver.enable_velocity_limits(True)
        self.posture_task = self.solver.add_joints_task()
        self.posture_task.set_joints({name: self.placo_robot.get_joint(name) for name in mj_names})
        self.posture_task.configure("arm_posture", "soft", 1e-4)
        self._update_mocap_target()

    def _update_ik(self):
        self._update_robot_state()
        self._ik_valid = False
        stream_fresh = self._xr_stream_is_fresh()
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
            # invalid cycle. Recovery rebases each wrist at its measured pose.
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
        return (
            self._last_xr_update_at is not None
            and now - self._last_xr_update_at <= self.XR_TRACKING_TIMEOUT_S
        )

    def _read_valid_xr_pose(self, source):
        pose = np.asarray(self.xr_client.get_pose_by_name(source), dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[3:]) < 1e-6:
            return None
        pose = pose.copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        return pose

    def _send_command(self):
        if self._ik_valid:
            qpos = calc_mujoco_qpos_from_placo_q(self.mj_model, self.placo_robot, self.placo_robot.state.q)
            command = calc_mujoco_ctrl_from_qpos(self.mj_model, qpos)
            command = np.clip(command, self.mj_model.actuator_ctrlrange[:, 0], self.mj_model.actuator_ctrlrange[:, 1])
            for i in range(self.mj_model.nu):
                joint_name = self.mj_model.joint(self.mj_model.actuator_trnid[i, 0]).name
                side = joint_name.split("_", 1)[0]
                if self.active[f"{side}_arm"]:
                    self.mj_data.ctrl[i] = command[i]
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
