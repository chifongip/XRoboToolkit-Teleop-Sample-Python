"""Model configuration for the fixed-base Unitree G1 simulations."""

import mujoco
import numpy as np

from xrobotoolkit_teleop.common.teleop_settings import DEFAULTS, TrackingReference
from xrobotoolkit_teleop.simulation.fixed_base_teleop_controller import FixedBaseTeleopController
from xrobotoolkit_teleop.simulation.unitree_g1_model import (
    DEFAULT_G1_VARIANT,
    G1_ASSETS,
    G1Variant,
    g1_arm_joints,
    g1_model_stem,
)


class UnitreeG1TeleopController(FixedBaseTeleopController):
    ROBOT_LABEL = "G1"
    EFFECTOR_SUFFIX = "palm_center"
    CAMERA_LOOKAT = (0, 0, 0.85)

    def __init__(
        self,
        xml_path=None,
        robot_urdf_path=None,
        *,
        variant: G1Variant = DEFAULT_G1_VARIANT,
        scale_factor: float = DEFAULTS.scale_factor,
        arm_velocity_limit: float = DEFAULTS.arm_velocity_limit,
        collision_margin: float = DEFAULTS.collision_margin,
        collision_trigger: float = DEFAULTS.collision_trigger,
        control_frequency: float = DEFAULTS.control_frequency,
        return_velocity_limit: float = DEFAULTS.return_velocity_limit,
        ik_backend="placo",
        visualize_ik: bool = False,
        tracking_reference: TrackingReference = "fixed-heading",
    ):
        stem = g1_model_stem(variant)
        self._model_dir = G1_ASSETS / stem
        self.variant = variant
        if (xml_path is None) != (robot_urdf_path is None):
            raise ValueError("Provide both --xml-path and --robot-urdf-path for a custom G1 model")
        if xml_path is None:
            xml_path = G1_ASSETS / f"{stem}_scene.xml"
            robot_urdf_path = self._model_dir / f"{stem}_ik.urdf"
        super().__init__(
            str(xml_path),
            str(robot_urdf_path),
            scale_factor=scale_factor,
            arm_velocity_limit=arm_velocity_limit,
            collision_margin=collision_margin,
            collision_trigger=collision_trigger,
            control_frequency=control_frequency,
            return_velocity_limit=return_velocity_limit,
            ik_backend=ik_backend,
            visualize_ik=visualize_ik,
            tracking_reference=tracking_reference,
        )

    def _backend_setup(self):
        super()._backend_setup()
        self._matched_frame("torso_link", self.mj_data)
        model = self.mj_model
        reference = mujoco.MjData(model)
        home = model.key("home").qpos
        initial = self.robot.state.q.copy()
        try:
            for group, joints in self.arm_joints.items():
                frame = self.manipulator_config[group]["link_name"]
                for name in joints:
                    address = model.joint(name).qposadr[0]
                    lower, upper = self.position_limits[name]
                    angle = float(home[address])
                    if not lower <= angle <= upper:
                        raise ValueError(f"G1 home position is outside shared joint limits for {name}")
                    # Perturb each joint to catch an axis mismatch at a zero-angle home.
                    test_angle = min(angle + 0.1, upper) if upper - angle > angle - lower else max(angle - 0.1, lower)
                    reference.qpos[:] = home
                    reference.qpos[address] = test_angle
                    mujoco.mj_forward(model, reference)
                    self.robot.set_joint(name, test_angle)
                    self.robot.update_kinematics()
                    self._matched_frame(frame, reference)
                    self.robot.set_joint(name, angle)
        finally:
            self.robot.state.q = initial
            self.robot.update_kinematics()

    def _collision_pairs_path(self):
        return str(self._model_dir / "collisions.json")

    def _configure_model(self, root):
        groups = g1_arm_joints(self.variant)
        expected = {joint for group in groups.values() for joint in group}
        movable = {
            joint.get("name"): joint.get("type") for joint in root.findall("joint") if joint.get("type") != "fixed"
        }
        model = self.mj_model
        mj_names = {model.joint(i).name for i in range(model.njnt)}
        if (
            set(movable) != expected
            or set(movable.values()) != {"revolute"}
            or mj_names != expected
            or not np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        ):
            raise ValueError(f"G1 {self.variant} requires matching fixed-base models with {len(expected)} arm hinges")
        links = {link.get("name") for link in root.findall("link")}
        required = {"torso_link", *(config["link_name"] for config in self.manipulator_config.values())}
        for name in sorted(required):
            if name not in links or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1:
                raise ValueError(f"G1 scene and URDF must both provide the {name} frame")
        self._effort_limits = {}
        for name in sorted(expected):
            limit = root.find(f"joint[@name='{name}']/limit")
            effort = float(limit.get("effort", "nan")) if limit is not None else np.nan
            if not np.isfinite(effort) or effort <= 0:
                raise ValueError(f"Invalid G1 URDF effort limit for {name}")
            self._effort_limits[name] = effort
        return groups, ()

    def _configure_actuators(self, arm_names):
        model = self.mj_model
        seen = set()
        for index in range(model.nu):
            name = self._position_servo_joint(index)
            if name not in arm_names or name in seen:
                raise ValueError(f"Unexpected or duplicate G1 actuator for {name}")
            joint = model.joint(name)
            limits = model.actuator_ctrlrange[index]
            force = model.actuator_forcerange[index]
            home = self.mj_data.qpos[joint.qposadr[0]]
            if (
                not model.actuator_ctrllimited[index]
                or not model.actuator_forcelimited[index]
                or not model.jnt_limited[joint.id]
                or not -self._effort_limits[name] <= force[0] < 0 < force[1] <= self._effort_limits[name]
                or limits[0] < joint.range[0] - 1e-6
                or limits[1] > joint.range[1] + 1e-6
                or not limits[0] <= home <= limits[1]
                or not np.isclose(self.mj_data.ctrl[index], home, atol=1e-8, rtol=0)
            ):
                raise ValueError(f"G1 actuator limits or home keyframe are invalid for {name}")
            self.arm_actuators[index] = name
            seen.add(name)
        if seen != arm_names:
            raise ValueError("G1 scene is missing required arm actuators")
