"""Cycle X2 from home to a T pose through mocked XR input, without an XR service."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
from mujoco import viewer as mj_viewer
import numpy as np
from meshcat import transformations as tf
import tyro

from xrobotoolkit_teleop.ik import BackendName
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class OfflineXR:
    """One advancing timestamp per control frame; both grips stay engaged."""

    def __init__(self):
        self.timestamp_ns = 1
        self.closed = False
        self.poses = {
            name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for name in ("headset", "left_controller", "right_controller")
        }

    def advance(self, dt):
        self.timestamp_ns += round(dt * 1e9)

    def get_timestamp_ns(self):
        return self.timestamp_ns

    def get_pose_by_name(self, name):
        return self.poses[name].copy()

    def get_key_value_by_name(self, name):
        return float(name.endswith("grip"))

    def get_button_state_by_name(self, name):
        return False

    def close(self):
        self.closed = True


def run_replay(
    ik_backend="placo",
    with_omnihand=False,
    cycles=2,
    move_seconds=2.0,
    hold_seconds=1.0,
    visualize=False,
    visualize_ik=False,
):
    """Exercise XR mapping, the selected IK backend, command checks, and physics.

    MuJoCo FK on separate reference data generates the desired poses. Actual
    robot joints are driven only by the controller's position servos. T-pose
    elbow targets are straight, so this also exercises a near-singular endpoint.
    """
    ik_backend = getattr(ik_backend, "value", ik_backend)
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    for name, value in (("move_seconds", move_seconds), ("hold_seconds", hold_seconds)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    assets = Path(ASSET_PATH) / "agibot/x2"
    scene, urdf = ("scene_omnihand.xml", "x2_omnihand_ik.urdf") if with_omnihand else ("scene.xml", "x2_fixed.urdf")
    with ExitStack() as cleanup:
        # Patch the client factory before construction: the native SDK is never initialized.
        with patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient", OfflineXR):
            with redirect_stdout(sys.stderr if visualize_ik else io.StringIO()):
                c = AgibotX2TeleopController(
                    str(assets / scene), str(assets / urdf), ik_backend=ik_backend, visualize_ik=visualize_ik
                )
        cleanup.callback(c.close)
        viewer = None
        if visualize:
            viewer = cleanup.enter_context(mj_viewer.launch_passive(c.mj_model, c.mj_data))
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135, -15, 2.5
            viewer.cam.lookat = [0, 0, 0.7]

        model = c.mj_model
        reference_data = mujoco.MjData(model)
        home = model.key("home").qpos.copy()
        t_pose = home.copy()
        for side, sign in (("left", 1), ("right", -1)):
            for joint, value in (("shoulder_pitch", 0.0), ("shoulder_roll", sign * 1.55), ("elbow", 0.0)):
                t_pose[model.joint(f"{side}_{joint}_joint").qposadr[0]] = value
        frames = {side: model.body(f"{side}_hand_center").id for side in ("left", "right")}

        def hand_frames(q):
            reference_data.qpos[:] = q
            mujoco.mj_forward(model, reference_data)
            result = {}
            for side, body in frames.items():
                T = np.eye(4)
                T[:3, :3] = reference_data.xmat[body].reshape(3, 3)
                T[:3, 3] = reference_data.xpos[body]
                result[side] = T
            return result

        reference = hand_frames(home)
        arm_indices = list(c.arm_actuators)
        command_limits = model.actuator_ctrlrange[arm_indices]
        max_steps = np.array(
            [c.robot.model.velocity_limit[c.robot.get_joint_v_offset(n)] * c.dt for n in c.arm_actuators.values()]
        )
        endpoints, violations = [], []
        failed_cycles = 0
        checked_cycles = 0
        max_step = 0.0
        max_mapping_error = 0.0

        def check(condition, message):
            if not condition and message not in violations and len(violations) < 10:
                violations.append(message)

        def advance(q):
            nonlocal failed_cycles, checked_cycles, max_step, max_mapping_error
            target = hand_frames(q)
            basis = R_HEADSET_TO_WORLD
            for side, T in target.items():
                delta = np.eye(4)
                delta[:3, :3] = basis.T @ T[:3, :3] @ reference[side][:3, :3].T @ basis
                c.xr_client.poses[f"{side}_controller"] = np.r_[
                    basis.T @ (T[:3, 3] - reference[side][:3, 3]),
                    np.roll(tf.quaternion_from_matrix(delta), -1),
                ]
            if viewer is not None and not viewer.is_running():
                raise KeyboardInterrupt
            started = time.monotonic()
            before = c.mj_data.ctrl[arm_indices].copy()
            c.xr_client.advance(c.dt)
            with viewer.lock() if viewer is not None else nullcontext():
                with redirect_stdout(io.StringIO()):
                    c.step_control()
            checked_cycles += 1
            failed_cycles += int(not c._ik_valid)
            if not c._ik_valid:
                check(False, c.status)
            commands = c.mj_data.ctrl[arm_indices]
            steps = np.abs(commands - before)
            max_step = max(max_step, float(np.max(steps)))
            if not all(np.isfinite(values).all() for values in (c.mj_data.qpos, c.mj_data.qvel, c.mj_data.ctrl)):
                raise RuntimeError("Offline replay produced nonfinite state or commands")
            check(np.all(steps <= max_steps + 1e-8), "arm command velocity bound exceeded")
            check(np.all(commands >= command_limits[:, 0] - 1e-8), "arm command below position bounds")
            check(np.all(commands <= command_limits[:, 1] + 1e-8), "arm command above position bounds")
            for side in target:
                actual_target = c.effector_task[f"{side}_arm"].T_world_frame
                if not np.isfinite(actual_target).all():
                    raise RuntimeError("Offline replay produced a nonfinite IK target")
                max_mapping_error = max(max_mapping_error, float(np.max(np.abs(actual_target - target[side]))))
            if viewer is not None:
                viewer.sync(state_only=True)
            if visualize or visualize_ik:
                time.sleep(max(0.0, c.dt - (time.monotonic() - started)))
            return target

        # Engage at the exact reference pose before sending any movement.
        advance(home)
        move_steps, hold_steps = math.ceil(move_seconds / c.dt), math.ceil(hold_seconds / c.dt)
        for cycle in range(cycles):
            for start, end, label in ((home, t_pose, "T"), (t_pose, home, "home")):
                for index in range(1, move_steps + 1):
                    alpha = (1 - np.cos(np.pi * index / move_steps)) / 2
                    advance(start + alpha * (end - start))
                for _ in range(hold_steps):
                    target = advance(end)
                with viewer.lock() if viewer is not None else nullcontext():
                    mujoco.mj_forward(model, c.mj_data)
                    positions, rotations = [], []
                    for side, body in frames.items():
                        positions.append(float(np.linalg.norm(c.mj_data.xpos[body] - target[side][:3, 3])))
                        R = c.mj_data.xmat[body].reshape(3, 3).T @ target[side][:3, :3]
                        rotations.append(float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                    endpoint = {
                        "cycle": cycle + 1,
                        "pose": label,
                        "max_position_error_m": max(positions),
                        "max_orientation_error_rad": max(rotations),
                        "arm_commands_rad": c.mj_data.ctrl[arm_indices].tolist(),
                        "hand_positions_m": {side: c.mj_data.xpos[body].tolist() for side, body in frames.items()},
                        "hand_orientations_wxyz": {
                            side: c.mj_data.xquat[body].tolist() for side, body in frames.items()
                        },
                    }
                    c._update_robot_state()
                    endpoint["min_measured_clearance_m"] = float(c.collision_checker.distances().min())
                    endpoints.append(endpoint)
                    check(max(positions) <= 0.02, f"{label} endpoint position error exceeds 20 mm")
                    check(max(rotations) <= 0.10, f"{label} endpoint orientation error exceeds 0.10 rad")
                    check(
                        endpoint["min_measured_clearance_m"] >= c.collision_margin - 1e-6,
                        f"{label} endpoint measured clearance is below the collision margin",
                    )

        joint_variation, position_drift, orientation_drift = 0.0, 0.0, 0.0
        for label in ("T", "home"):
            visits = [e for e in endpoints if e["pose"] == label]
            commands = np.array([e["arm_commands_rad"] for e in visits])
            joint_variation = max(joint_variation, float(np.max(np.abs(commands - commands[0]))))
            for side in frames:
                positions = np.array([e["hand_positions_m"][side] for e in visits])
                position_drift = max(position_drift, float(np.linalg.norm(positions - positions[0], axis=1).max()))
                quaternions = np.array([e["hand_orientations_wxyz"][side] for e in visits])
                angles = 2 * np.arccos(np.clip(np.abs(quaternions @ quaternions[0]), 0, 1))
                orientation_drift = max(orientation_drift, float(angles.max()))
        # The task constrains hand poses, leaving redundant joint choices free.
        # Report joint variation, and check repeatability of the measured task.
        check(position_drift <= 0.005, "repeated endpoint positions drift by more than 5 mm")
        check(orientation_drift <= 0.02, "repeated endpoint orientations drift by more than 0.02 rad")
        check(max_mapping_error <= 1e-6, "mock XR input does not reproduce the reference hand poses")
        return {
            "backend": ik_backend,
            "omnihand": with_omnihand,
            "cycles": cycles,
            "control_steps": checked_cycles,
            "arm_joint_names": list(c.arm_actuators.values()),
            "simulated_seconds": checked_cycles * c.dt,
            "passed": not violations,
            "violations": violations,
            "failed_cycles": failed_cycles,
            "max_arm_command_step_rad": max_step,
            "max_target_mapping_error": max_mapping_error,
            "max_endpoint_joint_variation_rad": joint_variation,
            "max_endpoint_position_drift_m": position_drift,
            "max_endpoint_orientation_drift_rad": orientation_drift,
            "endpoints": endpoints,
        }


def main(
    ik_backend: BackendName = BackendName.placo,
    with_omnihand: bool = False,
    cycles: int = 2,
    move_seconds: float = 2.0,
    hold_seconds: float = 1.0,
    visualize: bool = False,
    visualize_ik: bool = False,
):
    """Offline home -> T -> home test; exits with status 1 on a failed check.

    Args:
        ik_backend: Registered IK backend to exercise.
        with_omnihand: Use the articulated-hand scene; fingers remain open.
        cycles: Number of complete home -> T -> home cycles.
        move_seconds: Simulated duration of each smooth transition.
        hold_seconds: Simulated settling time at each endpoint.
        visualize: Open the MuJoCo physics viewer and pace the replay in real time.
        visualize_ik: Open the shared browser IK viewer and pace the replay in real time.
    """
    try:
        result = run_replay(
            ik_backend=ik_backend,
            with_omnihand=with_omnihand,
            cycles=cycles,
            move_seconds=move_seconds,
            hold_seconds=hold_seconds,
            visualize=visualize,
            visualize_ik=visualize_ik,
        )
    except KeyboardInterrupt:
        print("Offline replay interrupted before completion.", file=sys.stderr)
        raise SystemExit(130) from None
    except RuntimeError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
