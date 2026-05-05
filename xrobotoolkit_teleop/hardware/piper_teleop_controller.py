import time
from typing import Dict

import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import (
    HardwareTeleopController,
)
from xrobotoolkit_teleop.hardware.interface.piper import PiperInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)

import os

# Default paths and configurations for dual Piper arms
DEFAULT_DUAL_PIPER_URDF_PATH = os.path.join(
    ASSET_PATH, "agilex_piper/dual_piper.urdf"
)

DEFAULT_SCALE_FACTOR = 1.5
CONTROLLER_DEADZONE = 0.1

# Default CAN ports for the two Piper arms
DEFAULT_CAN_PORTS = {
    "left_arm": "can1",
    "right_arm": "can0",
}

# Gripper configuration (parallel gripper via GripperCtrl, no URDF joint)
DEFAULT_GRIPPER_CONFIG = {
    "type": "parallel",
    "gripper_trigger": "right_trigger",  # Shared; overridden per arm below
    "open_pos": [70.0],    # Fully open (mm)
    "close_pos": [0.0],    # Fully closed (mm)
}

DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "right_link6",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_gripper_joint"],
            "open_pos": [70.0],
            "close_pos": [0.0],
        },
    },
    "left_arm": {
        "link_name": "left_link6",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": ["left_gripper_joint"],
            "open_pos": [70.0],
            "close_pos": [0.0],
        },
    },
}


class PiperTeleopController(HardwareTeleopController):
    """
    Teleoperation controller for dual AgileX Piper arms.

    Extends HardwareTeleopController with Piper-specific hardware
    setup, state reading, and command sending via the Piper CAN SDK.
    """

    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_PIPER_URDF_PATH,
        manipulator_config: dict = DEFAULT_DUAL_PIPER_MANIPULATOR_CONFIG,
        can_ports: Dict[str, str] = DEFAULT_CAN_PORTS,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = True,
        log_dir: str = "logs/dual_piper",
        log_freq: float = 50,
        enable_camera: bool = False,
        camera_fps: int = 30,
    ):
        self.can_ports = can_ports

        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
        )

    def _placo_setup(self):
        """Extend parent setup with arm joint slices and joint regularization."""
        super()._placo_setup()

        # Add soft joint regularization to pull joints toward zero.
        # Prevents IK drift toward joint limits when the user holds a pose.
        joints_task = self.solver.add_joints_task()
        joints_task.set_joints({joint: 0.0 for joint in self.placo_robot.joint_names()})
        joints_task.configure("joints_regularization", "soft", 1e-4)

        self.placo_arm_joint_slice: Dict[str, slice] = {}
        for arm_name, config in self.manipulator_config.items():
            ee_link_name = config["link_name"]  # e.g. "right_link6"
            # Derive prefix: "right_link6" -> "right_", "left_link6" -> "left_"
            arm_prefix = ee_link_name.replace("link6", "")
            arm_joint_names = [f"{arm_prefix}joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        """Initialize Piper hardware interfaces for both arms.

        Idempotent: if arms are already connected, skips re-initialization.
        This handles the double call from __init__ -> run() without
        disconnecting and reconnecting the physical arms.
        """
        if hasattr(self, 'arm_controllers') and self.arm_controllers:
            print("Arms already connected. Skipping re-initialisation.")
            return

        self.arm_controllers: Dict[str, PiperInterface] = {}
        for arm_name, can_port in self.can_ports.items():
            print(f"Setting up Piper {arm_name} on CAN port: {can_port}")
            arm = PiperInterface(can_port=can_port, dt=self.dt)
            arm.connect()
            self.arm_controllers[arm_name] = arm

        print("Going to home position...")
        for arm in self.arm_controllers.values():
            arm.go_home()

        time.sleep(1)  # Wait for the arms to reach home
        print("Arms are at home.")

    def _initialize_camera(self):
        """
        Placeholder for camera initialisation.

        Camera support for Piper is not yet implemented. This logs a warning
        and sets the camera interface to None so the rest of the framework
        continues gracefully.
        """
        if self.enable_camera:
            print(
                "Warning: Camera interface not yet implemented for Piper. "
                "Disable cameras with --enable_camera False or ignore this message."
            )
            self.camera_interface = None

    def _update_robot_state(self):
        """Read current joint states from both arms and update Placo."""
        for arm_name, controller in self.arm_controllers.items():
            q_slice = self.placo_arm_joint_slice[arm_name]
            self.placo_robot.state.q[q_slice] = controller.get_joint_positions()[:6]

    def _send_command(self):
        """Send solved joint targets and gripper commands to the arms."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                # Active: send IK-solved target
                q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()
                controller.set_joint_positions(q_des)
            else:
                # Inactive: send current measured position as keepalive
                q_cur = controller.get_joint_positions()[:6]
                controller.set_joint_positions(q_cur)

            # --- Gripper control ---
            if "gripper_config" in self.manipulator_config[arm_name]:
                gripper_config = self.manipulator_config[arm_name]["gripper_config"]
                trigger_value = self.xr_client.get_key_value_by_name(
                    gripper_config["gripper_trigger"]
                )
                open_mm = gripper_config["open_pos"][0]
                close_mm = gripper_config["close_pos"][0]
                gripper_angle_mm = calc_parallel_gripper_position(
                    open_mm, close_mm, trigger_value
                )
                controller.set_gripper(
                    angle_mm=gripper_angle_mm,
                    effort_nm=0.5,
                    enable=True,
                )

    def _get_robot_state_for_logging(self) -> Dict:
        """Return robot-specific data for logging."""
        return {
            "qpos": {
                arm: c.get_joint_positions() for arm, c in self.arm_controllers.items()
            },
            "qvel": {
                arm: c.get_joint_velocities() for arm, c in self.arm_controllers.items()
            },
            "qpos_des": {
                arm: self.placo_robot.state.q[self.placo_arm_joint_slice[arm]].copy()
                for arm in self.arm_controllers
            },
            "gripper": {
                arm: c.get_gripper_state() for arm, c in self.arm_controllers.items()
            },
        }

    def _get_camera_frame_for_logging(self) -> Dict:
        """
        Placeholder for camera frame logging.

        Returns an empty dictionary since camera support is not yet implemented.
        """
        return {}

    def _shutdown_robot(self):
        """Gracefully shut down both arms (home position then disconnect)."""
        for _, controller in self.arm_controllers.items():
            controller.go_home()
        time.sleep(1)
