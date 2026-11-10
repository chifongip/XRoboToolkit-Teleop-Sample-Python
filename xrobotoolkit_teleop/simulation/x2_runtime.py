"""Backend-neutral MuJoCo/XR lifecycle used by the X2 controller."""

import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from meshcat import transformations as tf

from xrobotoolkit_teleop.ik.types import VisualizationSnapshot
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD, quat_diff_as_angle_axis


class X2MujocoRuntime:
    def __init__(
        self,
        xml_path,
        robot_urdf_path,
        manipulator_config,
        scale_factor=1.0,
        visualize_ik=False,
        dt=0.02,
        xr_factory=None,
    ):
        self.xml_path = xml_path
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.scale_factor = scale_factor
        self.dt = dt
        self.visualize_ik = bool(visualize_ik)
        self.R_headset_world = R_HEADSET_TO_WORLD.copy()
        self._stop_event = threading.Event()
        self._ik_viewer = None
        self._visual_links = tuple(link.get("name") for link in ET.parse(robot_urdf_path).getroot().findall("link"))
        for field in ("ref_ee_xyz", "ref_ee_quat", "ref_controller_xyz", "ref_controller_quat"):
            setattr(self, field, dict.fromkeys(manipulator_config))
        self.active = dict.fromkeys(manipulator_config, False)
        self.effector_task = {}
        self.status = "hold"
        self._robot_setup()
        self._backend_setup()
        self.xr_client = xr_factory()
        if self.visualize_ik:
            from xrobotoolkit_teleop.ik.visualization import IKViewer

            self._ik_viewer = IKViewer(robot_urdf_path)

    def _robot_setup(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]
        self.target_mocap_idx = {
            name: mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, config["vis_target"])
            for name, config in self.manipulator_config.items()
        }
        self.target_mocap_idx = {
            name: int(self.mj_model.body_mocapid[index]) if index >= 0 else -1
            for name, index in self.target_mocap_idx.items()
        }

    def _update_robot_state(self):
        for name in self.robot.joint_names():
            self.robot.set_joint(name, self.mj_data.qpos[self.mj_model.joint(name).qposadr[0]])
        self.robot.update_kinematics()

    def _measured_configuration(self):
        q = self.robot.state.q.copy()
        for name in self.robot.joint_names():
            q[self.robot.get_joint_offset(name)] = self.mj_data.qpos[self.mj_model.joint(name).qposadr[0]]
        return q

    def _get_link_pose(self, name):
        index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise ValueError(f"End effector body '{name}' not found in the model")
        return self.mj_data.xpos[index].copy(), self.mj_data.xquat[index].copy()

    def _update_mocap_target(self):
        for name, task in self.effector_task.items():
            index = self.target_mocap_idx[name]
            if index >= 0:
                self.mj_data.mocap_pos[index] = task.T_world_frame[:3, 3]
                self.mj_data.mocap_quat[index] = tf.quaternion_from_matrix(task.T_world_frame)

    def _process_xr_pose(self, pose, name):
        xyz = self.R_headset_world @ np.asarray(pose[:3])
        q = np.roll(pose[3:], 1)
        rotation = np.eye(4)
        rotation[:3, :3] = self.R_headset_world
        basis = tf.quaternion_from_matrix(rotation)
        q = tf.quaternion_multiply(tf.quaternion_multiply(basis, q), tf.quaternion_conjugate(basis))
        if self.ref_controller_xyz[name] is None:
            self.ref_controller_xyz[name] = xyz.copy()
            self.ref_controller_quat[name] = q.copy()
        return (
            (xyz - self.ref_controller_xyz[name]) * self.scale_factor,
            quat_diff_as_angle_axis(self.ref_controller_quat[name], q),
        )

    def visualization_snapshot(self):
        """Render the accepted outgoing commands, including projected finger coupling."""
        saved = self.robot.state.q.copy()
        try:
            for actuator, name in self.arm_actuators.items():
                self.robot.set_joint(name, self.mj_data.ctrl[actuator])
            q = self.robot.state.q.copy()
            self._project_hand_commands(q, self.mj_data.ctrl)
            self.robot.state.q = q
            self.robot.update_kinematics()
            return VisualizationSnapshot(
                {link: self.robot.get_T_world_frame(link) for link in self._visual_links},
                {name: task.T_world_frame.copy() for name, task in self.effector_task.items()},
                {name: self.robot.get_joint(name) for name in self.robot.joint_names()},
                self.status,
                {
                    name: self.robot.get_T_world_frame(config["link_name"])
                    for name, config in self.manipulator_config.items()
                },
            )
        finally:
            self.robot.state.q = saved
            self.robot.update_kinematics()

    def _update_ik_viz(self):
        if self._ik_viewer is not None:
            self._ik_viewer.update(self.visualization_snapshot())
