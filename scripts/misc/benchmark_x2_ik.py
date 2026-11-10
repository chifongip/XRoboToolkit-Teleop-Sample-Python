"""Replay a deterministic XR trajectory without a headset or viewer."""

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import tyro
from meshcat import transformations as tf

from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class ReplayXR:
    def __init__(self):
        self.timestamp = 1
        self.poses = {
            name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for name in ("headset", "left_controller", "right_controller")
        }

    def get_timestamp_ns(self):
        self.timestamp += 1
        return self.timestamp

    def get_pose_by_name(self, name):
        return self.poses[name].copy()

    def get_key_value_by_name(self, name):
        return 1.0 if name.endswith("grip") else 0.0

    def get_button_state_by_name(self, name):
        return False

    def close(self):
        pass


def main(ik_backend: Literal["placo", "lerobot"] = "placo", with_omnihand: bool = False, steps: int = 200):
    if steps < 4:
        raise ValueError("steps must be at least 4")
    assets = Path(ASSET_PATH) / "agibot/x2"
    scene, urdf = ("scene_omnihand.xml", "x2_omnihand_ik.urdf") if with_omnihand else ("scene.xml", "x2_fixed.urdf")
    setup_started = time.perf_counter()
    with (
        patch("xrobotoolkit_teleop.common.base_teleop_controller.XrClient", ReplayXR),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        controller = AgibotX2TeleopController(str(assets / scene), str(assets / urdf), ik_backend=ik_backend)
    setup_ms = 1000 * (time.perf_counter() - setup_started)
    timings, errors, rotation_errors, changes = [], [], [], []
    failures = 0
    try:
        # Report the first solve separately from the steady-state trajectory.
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            controller.step_control()
        startup_ms = 1000 * (time.perf_counter() - started)
        for index in range(steps):
            phase = 2 * np.pi * index / steps
            for side, sign in (("left", 1), ("right", -1)):
                pose = controller.xr_client.poses[f"{side}_controller"]
                pose[:3] = [sign * 0.02 * np.sin(phase), 0.02 * (1 - np.cos(phase)), -0.04 * np.sin(phase)]
                angle = 0.15 * np.sin(phase)
                pose[3:] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            before = controller.mj_data.ctrl.copy()
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                controller.step_control()
            timings.append(1000 * (time.perf_counter() - started))
            failures += int(not controller._ik_valid)
            changes.append(max(abs(controller.mj_data.ctrl[i] - before[i]) for i in controller.arm_actuators))
            for name, config in controller.manipulator_config.items():
                actual, quaternion = controller._get_link_pose(config["link_name"])
                errors.append(np.linalg.norm(actual - controller.effector_task[name].T_world_frame[:3, 3]))
                target_quaternion = tf.quaternion_from_matrix(controller.effector_task[name].T_world_frame)
                rotation_errors.append(2 * np.arccos(np.clip(abs(np.dot(quaternion, target_quaternion)), 0, 1)))
        print(
            json.dumps(
                {
                    "backend": ik_backend,
                    "omnihand": with_omnihand,
                    "steps": steps,
                    "setup_ms": setup_ms,
                    "first_cycle_ms": startup_ms,
                    "cycle_ms": dict(zip(("median", "p95", "max"), map(float, np.percentile(timings, [50, 95, 100])))),
                    "cycles_over_20ms": sum(t > 20 for t in timings),
                    "failed_cycles": failures,
                    "position_error_m": {"mean": float(np.mean(errors)), "max": float(max(errors))},
                    "orientation_error_rad": {
                        "mean": float(np.mean(rotation_errors)),
                        "max": float(max(rotation_errors)),
                    },
                    "max_arm_command_step_rad": float(max(changes)),
                },
                indent=2,
            )
        )
    finally:
        if controller.lerobot_ik is not None:
            controller.lerobot_ik.close()
        controller.xr_client.close()


if __name__ == "__main__":
    tyro.cli(main)
