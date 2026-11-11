"""Compare exhaustive and selective bridge checks without XR/hardware.

Run with PYTHONPATH=. in the native XR environment. Timings exclude the
independent exhaustive validation of every accepted command.
"""

import argparse
import json
import time

import numpy as np

from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame
from xrobotoolkit_teleop.utils.collision_checks import CollisionChecker


def initial_state(retargeter):
    home = dict(
        zip(
            retargeter.arm_names,
            np.array([0.35, 0.1, 0, -0.87, 0, 0, 0, 0.35, -0.1, 0, -0.87, 0, 0, 0], dtype=np.float32).astype(float),
            strict=True,
        )
    )
    home.update(dict.fromkeys(retargeter.hand_names, 0.0))
    measured = dict.fromkeys(retargeter.joint_names, 0.0)
    measured.update(home)
    for side in ("left", "right"):
        measured.update({f"{side}_knee_joint": 0.3, f"{side}_hip_pitch_joint": -0.1, f"{side}_ankle_pitch_joint": -0.2})
    return measured, home


def replay(scenario, selective, steps, backend="lerobot"):
    r = ExternalStateRetargeter(robot="x2_omnihand", ik_backend=backend)
    try:
        if not callable(getattr(r.robot, "distance_subset", None)):
            raise RuntimeError(f"{backend} requires optional native bindings for selective queries")
        measured, home = initial_state(r)
        if scenario == "recovery":
            measured.update(left_shoulder_pitch_joint=2.0, left_shoulder_roll_joint=-0.06)
        previous = {name: measured[name] for name in r.command_names}
        recovery_measurements = measured.copy()
        recovery_previous = previous.copy()
        poses = {
            name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for name in ("headset", "left_controller", "right_controller")
        }
        limits = r.limits | r.profile.hands.limits
        oracle = CollisionChecker(r.robot)
        # Validate every path sample exactly, including those that conservative
        # bounds could certify. Disabling subset queries alone is insufficient.
        oracle.can_bound = False
        timings, collision_times, pair_counts = [], [], []
        failures = 0
        depth, collision_seconds, pairs = 0, 0.0, 0

        def instrument(method):
            def call(*args, **kwargs):
                nonlocal depth, collision_seconds
                outer = depth == 0
                started = time.perf_counter()
                depth += 1
                try:
                    return method(*args, **kwargs)
                finally:
                    depth -= 1
                    if outer:
                        collision_seconds += time.perf_counter() - started

            return call

        for name in ("clearances", "distances", "linearized_clearance", "clearance_bounds"):
            setattr(r.collision_checker, name, instrument(getattr(r.collision_checker, name)))
        native_full, native_subset = r.robot.distances, r.robot.distance_subset

        def full_query():
            nonlocal pairs
            pairs += r.collision_checker.pair_count
            return native_full()

        def subset_query(indices):
            nonlocal pairs
            pairs += len(indices)
            return native_subset(indices)

        r.robot.distances = full_query
        r.robot.distance_subset = subset_query if selective else None
        for index in range(steps + 6):
            if scenario == "recovery":
                # Repeat a penetrating snapshot so warmup cannot hide the cost
                # of recovery's exhaustive finite-difference queries.
                measured = recovery_measurements.copy()
                previous = recovery_previous.copy()
                r.backend.reset()
            phase = max(0, index - 5) / steps * 2 * np.pi
            for side, sign in (("left", 1), ("right", -1)):
                poses[f"{side}_controller"][0] = sign * 0.015 * np.sin(phase)
            trigger = (0.5 * (1 - np.cos(phase))) if scenario == "closing" else 0.0
            frame = XRFrame(
                index + 1,
                poses,
                dict.fromkeys(("left", "right"), float(index >= 2)),
                triggers=dict.fromkeys(("left", "right"), trigger),
            )
            for name, value in measured.items():
                r.robot.set_joint(name, value)
            measured_q = r.robot.state.q.copy()
            previous_q = measured_q.copy()
            for name, value in previous.items():
                previous_q[r.robot.get_joint_offset(name)] = value
            r._project_hands(previous_q)
            collision_seconds, pairs = 0.0, 0
            started = time.perf_counter()
            result = r.step(measured, previous, home, limits, intent="TRACK", frame=frame, control_dt=0.02)
            elapsed = 1000 * (time.perf_counter() - started)
            if index >= 6:
                timings.append(elapsed)
                collision_times.append(1000 * collision_seconds)
                pair_counts.append(pairs)
                failures += result is None
            if result is not None:
                candidate = measured_q.copy()
                for name, value in result.items():
                    candidate[r.robot.get_joint_offset(name)] = value
                r._project_hands(candidate)
                checker, saved = r.collision_checker, r.robot.state.q.copy()
                try:
                    r.collision_checker = oracle
                    oracle.clear_cache()
                    r._validate_motion(measured_q, previous_q, candidate)
                finally:
                    r.collision_checker = checker
                    r.robot.state.q = saved
                    r.robot.update_kinematics()
                if not r.acknowledge(result):
                    raise RuntimeError("Replay acknowledgement failed")
                previous = result
                fraction = 0.25 if scenario == "servo_lag" else 1.0
                next_q = measured_q + fraction * (candidate - measured_q)
                r._project_hands(next_q)
                measured = {name: float(next_q[r.robot.get_joint_offset(name)]) for name in r.joint_names}
        return {
            "backend": backend,
            "scenario": scenario,
            "selective": selective,
            "steps": steps,
            "retarget_ms": dict(zip(("median", "p95"), np.percentile(timings, [50, 95]).tolist())),
            "collision_ms": dict(zip(("median", "p95"), np.percentile(collision_times, [50, 95]).tolist())),
            "exact_pairs_median": float(np.median(pair_counts)),
            "failed_cycles": int(failures),
        }
    finally:
        r.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ik-backend", choices=("placo", "lerobot"), default="lerobot")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["stationary", "closing", "servo_lag", "recovery"],
        choices=["stationary", "closing", "servo_lag", "recovery"],
    )
    args = parser.parse_args()
    if args.steps < 4 or args.repeats < 1:
        parser.error("steps must be at least 4 and repeats at least 1")
    for repeat in range(args.repeats):
        for scenario in args.scenarios:
            for selective in (False, True) if repeat % 2 == 0 else (True, False):
                print(
                    json.dumps({"repeat": repeat, **replay(scenario, selective, args.steps, args.ik_backend)}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
