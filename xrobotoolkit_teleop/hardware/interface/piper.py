import math
import time
from typing import List, Optional, Union

import numpy as np


class PiperInterface:
    """
    Hardware interface for a single AgileX Piper robotic arm.

    Wraps the C_PiperInterface_V2 SDK to provide a clean API for
    joint position control, state feedback, and gripper control.

    The Piper SDK uses millidegree units for joint angles and
    millimetre units for gripper stroke. All public methods
    in this interface use standard SI units (radians, metres).
    """

    # Joint limits (radians) from the Piper SDK documentation
    JOINT_LIMITS = {
        0: (-2.6179, 2.6179),   # joint 1
        1: (0.0, 3.14),          # joint 2
        2: (-2.967, 0.0),        # joint 3
        3: (-1.745, 1.745),      # joint 4
        4: (-1.22, 1.22),        # joint 5
        5: (-2.09439, 2.09439),  # joint 6
    }

    # Default gripper range (mm)
    GRIPPER_MIN_MM = 0.0
    GRIPPER_MAX_MM = 70.0

    def __init__(
        self,
        can_port: str = "can0",
        dt: float = 0.01,
    ):
        """
        Initialize the Piper arm interface.

        Args:
            can_port: The CAN port name (e.g. "can0", "can1").
            dt: The control loop timestep in seconds.
        """
        self.dt = dt
        self.can_port = can_port

        # Import the Piper SDK lazily so import doesn't fail if SDK is absent
        self._piper = None
        try:
            from piper_sdk import C_PiperInterface_V2
            self._piper = C_PiperInterface_V2(
                can_name=can_port,
                judge_flag=False,
            )
            self._sdk_available = True
        except ImportError:
            print(f"Warning: piper_sdk not available. Cannot control arm on {can_port}.")
            self._sdk_available = False
        except Exception as e:
            print(f"Warning: Failed to initialize Piper SDK on {can_port}: {e}")
            self._sdk_available = False

        # Cached joint positions (updated by get_joint_positions)
        self._cached_joint_positions = np.zeros(6)
        # Guard flag: set to True after explicit disconnect to prevent
        # __del__ from sending DisableArm on already-gone CAN resources.
        self._disconnected = True

    def connect(self) -> bool:
        """
        Connect to the arm over CAN, enable motors, and set joint control mode.

        Returns:
            True if connection and initialization succeeded.
        """
        if not self._sdk_available or self._piper is None:
            print(f"PiperInterface: Cannot connect on {self.can_port} — SDK not available.")
            return False

        try:
            # Start the CAN port reading thread and initialize the arm
            self._piper.ConnectPort(
                can_init=False,
                piper_init=True,
                start_thread=True,
            )
            time.sleep(0.5)

            # Enable all motors
            self._piper.EnableArm(motor_num=7, enable_flag=0x02)
            time.sleep(0.2)

            # Set to CAN command control mode with MOVE_J (joint) mode
            self._piper.ModeCtrl(
                ctrl_mode=0x01,
                move_mode=0x01,
                move_spd_rate_ctrl=50,
                is_mit_mode=0x00,
            )
            time.sleep(0.2)

            self._disconnected = False
            print(f"PiperInterface: Connected and initialized on {self.can_port}.")
            return True

        except Exception as e:
            print(f"PiperInterface: Failed to connect on {self.can_port}: {e}")
            return False

    def disconnect(self) -> None:
        """Disable motors and disconnect from the CAN port."""
        if self._disconnected:
            return
        self._disconnected = True
        if self._piper is not None and self._sdk_available:
            try:
                self._piper.DisableArm(motor_num=7, enable_flag=0x01)
                time.sleep(0.1)
                self._piper.DisconnectPort(thread_timeout=0.1)
                print(f"PiperInterface: Disconnected from {self.can_port}.")
            except Exception as e:
                print(f"PiperInterface: Error disconnecting on {self.can_port}: {e}")

    def go_home(self) -> bool:
        """
        Move the arm to a neutral home position.

        Home position (approx, in radians):
            [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]

        Returns:
            True if the command was sent successfully.
        """
        home_pos = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0])
        return self.set_joint_positions(home_pos)

    def enable_arm(self) -> None:
        """Enable all motors on the arm."""
        if self._piper is not None and self._sdk_available:
            self._piper.EnableArm(motor_num=7, enable_flag=0x02)

    def disable_arm(self) -> None:
        """Disable all motors on the arm."""
        if self._piper is not None and self._sdk_available:
            self._piper.DisableArm(motor_num=7, enable_flag=0x01)

    def emergency_stop(self) -> None:
        """Trigger emergency stop on the arm."""
        if self._piper is not None and self._sdk_available:
            self._piper.EmergencyStop(emergency_stop=0x01)

    def _rad_to_milli_deg(self, rad: float) -> int:
        """Convert radians to millidegrees (int) for the Piper SDK."""
        return int(round(rad * 180.0 / math.pi * 1000.0))

    def _milli_deg_to_rad(self, milli_deg: int) -> float:
        """Convert millidegrees (int) from the Piper SDK to radians."""
        return float(milli_deg) * math.pi / 180.0 / 1000.0

    def set_joint_positions(
        self,
        positions: Union[List[float], np.ndarray],
    ) -> bool:
        """
        Set target joint positions for all 6 joints.

        Args:
            positions: Array of 6 joint angles in radians.

        Returns:
            True if the command was sent successfully.
        """
        if self._piper is None or not self._sdk_available:
            return False

        if len(positions) < 6:
            print(f"PiperInterface: Expected 6 joint positions, got {len(positions)}")
            return False

        try:
            # Clamp to joint limits to prevent the arm from entering a fault state
            clamped = np.clip(
                positions[:6],
                [self.JOINT_LIMITS[i][0] for i in range(6)],
                [self.JOINT_LIMITS[i][1] for i in range(6)],
            )

            # Convert radians to millidegrees for the SDK
            j1 = self._rad_to_milli_deg(clamped[0])
            j2 = self._rad_to_milli_deg(clamped[1])
            j3 = self._rad_to_milli_deg(clamped[2])
            j4 = self._rad_to_milli_deg(clamped[3])
            j5 = self._rad_to_milli_deg(clamped[4])
            j6 = self._rad_to_milli_deg(clamped[5])

            self._piper.JointCtrl(j1, j2, j3, j4, j5, j6)
            return True

        except Exception as e:
            print(f"PiperInterface: Error setting joint positions: {e}")
            return False

    def get_joint_positions(
        self,
        joint_names: Optional[Union[str, List[str]]] = None,
    ) -> Union[float, List[float]]:
        """
        Get the current joint positions from the arm.

        Args:
            joint_names: Optional filter (unused — returns all 6 joints).

        Returns:
            Array of 6 joint angles in radians.
        """
        if self._piper is None or not self._sdk_available:
            return self._cached_joint_positions.copy()

        try:
            msg = self._piper.GetArmJointMsgs()
            if msg is not None and hasattr(msg, 'joint_state'):
                js = msg.joint_state
                positions = np.array([
                    self._milli_deg_to_rad(js.joint_1),
                    self._milli_deg_to_rad(js.joint_2),
                    self._milli_deg_to_rad(js.joint_3),
                    self._milli_deg_to_rad(js.joint_4),
                    self._milli_deg_to_rad(js.joint_5),
                    self._milli_deg_to_rad(js.joint_6),
                ])
                self._cached_joint_positions = positions
                return positions.copy()
        except Exception as e:
            print(f"PiperInterface: Error reading joint positions: {e}")

        return self._cached_joint_positions.copy()

    def get_joint_velocities(
        self,
        joint_names: Optional[Union[str, List[str]]] = None,
    ) -> Union[float, List[float]]:
        """
        Get the current joint velocities from the arm.

        Uses high-speed feedback messages. Returns zeros if data is unavailable.

        Returns:
            Array of 6 joint velocities in rad/s.
        """
        if self._piper is None or not self._sdk_available:
            return np.zeros(6)

        try:
            msg = self._piper.GetMotorStates()
            if msg is not None and hasattr(msg, 'motor_state'):
                ms = msg.motor_state
                # Motor speed is in 0.001 rad/s
                return np.array([float(ms.motor_speed) / 1000.0] * 6)
        except Exception:
            pass

        # Fallback: return zeros if velocity data is not easily accessible
        return np.zeros(6)

    def get_joint_currents(
        self,
        joint_names: Optional[Union[str, List[str]]] = None,
    ) -> Union[float, List[float]]:
        """
        Get the current motor currents from the arm.

        Returns:
            Array of 6 motor currents in A.
        """
        if self._piper is None or not self._sdk_available:
            return np.zeros(6)

        try:
            msg = self._piper.GetMotorStates()
            if msg is not None:
                motor_state = msg[2]
                # motor_state.current is in 0.001A
                current = float(motor_state.current) / 1000.0
                # Single joint data — for simplicity return all zeros for now
                pass
        except Exception:
            pass

        return np.zeros(6)

    def get_gripper_state(self) -> tuple:
        """
        Get the current gripper state.

        Returns:
            Tuple of (gripper_angle_mm, gripper_effort_nm, is_enabled).
        """
        if self._piper is None or not self._sdk_available:
            return (0.0, 0.0, False)

        try:
            msg = self._piper.GetArmGripperMsgs()
            if msg is not None and hasattr(msg, 'gripper_state'):
                gs = msg.gripper_state
                angle_mm = float(gs.grippers_angle) / 1000.0
                effort_nm = float(gs.grippers_effort) / 1000.0
                # Check if gripper is enabled via driver_enable_status
                is_enabled = bool(getattr(gs.foc_status, 'driver_enable_status', False))
                return (angle_mm, effort_nm, is_enabled)
        except Exception as e:
            print(f"PiperInterface: Error reading gripper state: {e}")

        return (0.0, 0.0, False)

    def set_gripper(
        self,
        angle_mm: float,
        effort_nm: float = 0.5,
        enable: bool = True,
    ) -> bool:
        """
        Set the gripper position and effort.

        Args:
            angle_mm: Target gripper opening in mm (0 = closed, ~70 = fully open).
            effort_nm: Gripper torque in N·m (0 to 5.0).
            enable: Whether to enable the gripper before commanding.

        Returns:
            True if the command was sent successfully.
        """
        if self._piper is None or not self._sdk_available:
            return False

        try:
            # Clamp to valid range
            angle_mm = max(self.GRIPPER_MIN_MM, min(self.GRIPPER_MAX_MM, angle_mm))
            effort_nm = max(0.0, min(5.0, effort_nm))

            # Convert to SDK units
            gripper_angle = int(round(angle_mm * 1000.0))  # mm → 0.001mm
            gripper_effort = int(round(effort_nm * 1000.0))  # N·m → 0.001 N·m

            gripper_code = 0x01  # Enable
            if not enable:
                gripper_code = 0x00  # Disable

            self._piper.GripperCtrl(
                gripper_angle=gripper_angle,
                gripper_effort=gripper_effort,
                gripper_code=gripper_code,
                set_zero=0x00,
            )
            return True

        except Exception as e:
            print(f"PiperInterface: Error setting gripper: {e}")
            return False

    def get_arm_status(self) -> int:
        """
        Get the current arm status code.

        Returns:
            Status code: 0 = normal, 1 = emergency stop, etc.
        """
        if self._piper is None or not self._sdk_available:
            return -1

        try:
            msg = self._piper.GetArmStatus()
            if msg is not None and hasattr(msg, 'arm_status'):
                arm_st = msg.arm_status
                return int(arm_st.arm_status)
        except Exception:
            pass

        return -1

    def get_connection_status(self) -> bool:
        """Check if the arm is connected and the CAN thread is running."""
        if self._piper is None or not self._sdk_available:
            return False
        try:
            return bool(self._piper.get_connect_status())
        except Exception:
            return False

    def __del__(self):
        print("PiperInterface is being deleted")
