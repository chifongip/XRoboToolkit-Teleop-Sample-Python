import time
from typing import TYPE_CHECKING

import numpy as np
import xrobotoolkit_sdk as xrt

if TYPE_CHECKING:
    from xrobotoolkit_teleop.common.waist_tracking import WaistSample


class XrClient:
    """Client for the XrClient SDK to interact with XR devices."""

    def __init__(self):
        """Initializes the XrClient and the SDK."""
        xrt.init()
        print("XRoboToolkit SDK initialized.")

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Returns the pose of the specified device by name.
        Valid names: "left_controller", "right_controller", "headset".
        Pose is [x, y, z, qx, qy, qz, qw], passed through from the SDK.
        The teleoperation adapter expects native XR tracking-world axes:
        +X right, +Y up, +Z backward; positions are in meters.
        """
        if name == "left_controller":
            return xrt.get_left_controller_pose()
        elif name == "right_controller":
            return xrt.get_right_controller_pose()
        elif name == "headset":
            return xrt.get_headset_pose()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_controller', 'right_controller', 'headset'."
            )

    def get_key_value_by_name(self, name: str) -> float:
        """Returns the trigger/grip value by name (float).
        Valid names: "left_trigger", "right_trigger", "left_grip", "right_grip".
        """
        if name == "left_trigger":
            return xrt.get_left_trigger()
        elif name == "right_trigger":
            return xrt.get_right_trigger()
        elif name == "left_grip":
            return xrt.get_left_grip()
        elif name == "right_grip":
            return xrt.get_right_grip()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
            )

    def get_button_state_by_name(self, name: str) -> bool:
        """Returns the button state by name (bool).
        Valid names: "A", "B", "X", "Y",
                      "left_menu_button", "right_menu_button",
                      "left_axis_click", "right_axis_click"
        """
        if name == "A":
            return xrt.get_A_button()
        elif name == "B":
            return xrt.get_B_button()
        elif name == "X":
            return xrt.get_X_button()
        elif name == "Y":
            return xrt.get_Y_button()
        elif name == "left_menu_button":
            return xrt.get_left_menu_button()
        elif name == "right_menu_button":
            return xrt.get_right_menu_button()
        elif name == "left_axis_click":
            return xrt.get_left_axis_click()
        elif name == "right_axis_click":
            return xrt.get_right_axis_click()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'A', 'B', 'X', 'Y', "
                "'left_menu_button', 'right_menu_button', 'left_axis_click', 'right_axis_click'."
            )

    def get_timestamp_ns(self) -> int:
        """Returns the current timestamp in nanoseconds (int)."""
        return xrt.get_time_stamp_ns()

    def get_hand_tracking_state(self, hand: str) -> np.ndarray | None:
        """Returns the hand tracking state for the specified hand.
        Valid hands: "left", "right".
        State is a 27 x 7 numpy array, where each row is [x, y, z, qx, qy, qz, qw] for each joint.
        Returns None if hand tracking is inactive (low quality).
        """
        if hand.lower() == "left":
            if not xrt.get_left_hand_is_active():
                return None
            return xrt.get_left_hand_tracking_state()
        elif hand.lower() == "right":
            if not xrt.get_right_hand_is_active():
                return None
            return xrt.get_right_hand_tracking_state()
        else:
            raise ValueError(f"Invalid hand: {hand}. Valid hands are: 'left', 'right'.")

    def get_joystick_state(self, controller: str) -> list[float]:
        """Returns the joystick state for the specified controller.
        Valid controllers: "left", "right".
        State is a list with shape (2) representing [x, y] for each joystick.
        """
        if controller.lower() == "left":
            return xrt.get_left_axis()
        elif controller.lower() == "right":
            return xrt.get_right_axis()
        else:
            raise ValueError(f"Invalid controller: {controller}. Valid controllers are: 'left', 'right'.")

    def get_motion_tracker_data(self) -> dict:
        """Returns a dictionary of motion tracker data, where the keys are the tracker serial numbers.
        Each value is a dictionary containing the pose, velocity, and acceleration of the tracker.
        """
        num_motion_data = xrt.num_motion_data_available()
        if num_motion_data == 0:
            return {}

        poses = xrt.get_motion_tracker_pose()
        velocities = xrt.get_motion_tracker_velocity()
        accelerations = xrt.get_motion_tracker_acceleration()
        serial_numbers = xrt.get_motion_tracker_serial_numbers()

        tracker_data = {}
        for i in range(num_motion_data):
            serial = serial_numbers[i]
            tracker_data[serial] = {
                "pose": poses[i],
                "velocity": velocities[i],
                "acceleration": accelerations[i],
            }

        return tracker_data

    def get_body_tracking_data(self) -> dict | None:
        """Returns complete body tracking data or None if unavailable.

        Returns:
            Dict with keys: 'poses', 'velocities', 'accelerations'
            - poses: (24, 7) array [x,y,z,qx,qy,qz,qw] for each joint
            - velocities: (24, 6) array [vx,vy,vz,wx,wy,wz] for each joint
            - accelerations: (24, 6) array [ax,ay,az,wax,way,waz] for each joint
        """
        if not xrt.is_body_data_available():
            return None

        return {
            "poses": xrt.get_body_joints_pose(),
            "velocities": xrt.get_body_joints_velocity(),
            "accelerations": xrt.get_body_joints_acceleration(),
        }

    def get_pelvis_timestamp_ns(self) -> int:
        """Return the SDK's per-joint timestamp for the pelvis (body joint 0)."""
        return xrt.get_body_joints_timestamp()[0]

    @staticmethod
    def _body_frame_timestamps():
        try:
            aggregate = xrt.get_body_timestamp_ns()
        except (AttributeError, RuntimeError):
            aggregate = None
        try:
            head = xrt.get_body_joints_timestamp()[15]
        except (AttributeError, IndexError, RuntimeError, TypeError):
            head = None
        return (aggregate if type(aggregate) is int else None, head if type(head) is int else None)

    def get_waist_sample(self) -> "WaistSample":
        """Best-effort stable read using the unmodified SDK's public getters.

        Bracket pose reads with packet and body timestamps. The SDK can
        retain cached fields, so this does not guarantee an atomic snapshot.
        """
        from xrobotoolkit_teleop.common.waist_tracking import WaistSample

        for _ in range(3):
            observed_at = time.monotonic()
            if not xrt.is_body_data_available():
                raise ValueError("body tracking unavailable")
            before = xrt.get_time_stamp_ns()
            body_before = self._body_frame_timestamps()
            body = np.asarray(xrt.get_body_joints_pose(), dtype=float)
            controllers = {
                side: np.asarray(self.get_pose_by_name(f"{side}_controller"), dtype=float).copy()
                for side in ("left", "right")
            }
            body_after = self._body_frame_timestamps()
            after = xrt.get_time_stamp_ns()
            if before == after and body_before == body_after and xrt.is_body_data_available():
                if body.shape != (24, 7):
                    raise ValueError("invalid body pose array")
                return WaistSample(
                    pelvis_pose=body[0].copy(),
                    controllers=controllers,
                    body_timestamp_ns=body_before[0],
                    xr_timestamp_ns=before,
                    observed_at=observed_at,
                    hip_positions=body[1:3, :3].copy(),  # PICO body joints: left hip, right hip.
                    body_head_timestamp_ns=body_before[1],
                )
        raise ValueError("XR packet or body timestamps changed during waist sample read")

    def close(self):
        xrt.close()
