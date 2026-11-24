"""Agibot X2 fixed-base model and optional Omnihand Pro controls."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.ik import create_backend
from xrobotoolkit_teleop.simulation.agibot_x2_hand_config import HAND_CLOSED
from xrobotoolkit_teleop.simulation.fixed_base_teleop_controller import FixedBaseTeleopController
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class AgibotX2TeleopController(FixedBaseTeleopController):
    ROBOT_LABEL = "X2"
    HAND_CLOSED = HAND_CLOSED

    @staticmethod
    def _create_xr_client():
        return XrClient()

    def _create_backend(self, spec):
        return create_backend(self.ik_backend, spec)

    def _collision_pairs_path(self):
        filename = "collisions_omnihand.json" if self.hand_joint_names else "collisions.json"
        return str(Path(ASSET_PATH) / "agibot/x2" / filename)

    def _configure_model(self, root):
        movable = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
        arm_names = {name for name in movable if name.startswith(("left_", "right_"))}
        arm_joints = {
            f"{side}_arm": sorted(n for n in arm_names if n.startswith(f"{side}_")) for side in ("left", "right")
        }
        if any(len(names) != 7 for names in arm_joints.values()):
            raise ValueError("X2 requires seven joints per arm")
        return arm_joints, sorted(movable - arm_names)

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
            joint = self._position_servo_joint(i)
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
        if mj_names != arm_names | set(self.hand_joint_names):
            raise ValueError("Omnihand Pro requires x2_omnihand_ik.urdf with matching articulated hand joints")
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

    def _update_hand_commands(self, previous_commands):
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
