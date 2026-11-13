"""Measure collision-disabled LeRobot retargeting without XR or robot commands.

Optionally compare the working tree against a Git revision's solver, transaction
backend, and numeric model. Run from the repository root in the native environment.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import subprocess
import time
from types import ModuleType
from unittest.mock import patch

import numpy as np

from xrobotoolkit_teleop.common.teleop_settings import TeleopSettings
from xrobotoolkit_teleop.ik import lerobot_backend
from xrobotoolkit_teleop.integrations.retargeter import ExternalStateRetargeter, XRFrame

ROOT = Path(__file__).resolve().parents[2]


def baseline_factory(ref):
    """Load the three changed components without modifying repository imports."""
    modules = {}
    for name in ("backend", "lerobot_solver", "pinocchio_model"):
        path = f"xrobotoolkit_teleop/ik/{name}.py"
        source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)
        module = ModuleType(f"xrobotoolkit_teleop.ik._latency_baseline_{name}")
        module.__package__ = "xrobotoolkit_teleop.ik"
        exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
        modules[name] = module

    def make(spec):
        solver = modules["lerobot_solver"].LeRobotIKSolver(spec)
        robot = modules["pinocchio_model"].PinocchioModel(spec)
        return modules["backend"].TransactionalBackend(spec, robot, solver)

    return make


@contextmanager
def implementation(factory):
    if factory is None:
        yield
    else:
        with patch.object(lerobot_backend, "LeRobotBackend", factory):
            yield


def measure(scenario, steps, robot):
    r = ExternalStateRetargeter(TeleopSettings(collision_checking=False), robot=robot, ik_backend="lerobot")
    try:
        home = dict(
            zip(
                r.arm_names,
                np.array([0.35, 0.1, 0, -0.87, 0, 0, 0, 0.35, -0.1, 0, -0.87, 0, 0, 0], dtype=np.float32).astype(float),
                strict=True,
            )
        )
        home.update(dict.fromkeys(r.hand_names, 0.0))
        measured = dict.fromkeys(r.joint_names, 0.0)
        measured.update(home)
        previous = home.copy()
        poses = {
            name: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for name in ("headset", "left_controller", "right_controller")
        }
        limits = r.limits | (r.profile.hands.limits if r.hand_names else {})
        solver = r.backend.solver
        native_run = solver._run_optimizer
        solves, records, commands, errors = [], [], [], []

        def run(*args):
            warm = solver._warm is not None
            started = time.perf_counter()
            result = native_run(*args)
            solves.append((1000 * (time.perf_counter() - started), solver.last_stats["iter_count"], warm))
            return result

        solver._run_optimizer = run
        for index in range(steps + 6):
            phase = max(0, index - 5) / steps * 2 * np.pi
            for side, sign in (("left", 1), ("right", -1)):
                poses[f"{side}_controller"][0] = sign * 0.015 * np.sin(phase)
            trigger = 0.5 * (1 - np.cos(phase)) if scenario == "closing" else 0.0
            frame = XRFrame(
                index + 1,
                poses,
                dict.fromkeys(("left", "right"), float(index >= 2)),
                triggers=dict.fromkeys(("left", "right"), trigger),
            )
            count = len(solves)
            started = time.perf_counter()
            result = r.step(measured, previous, home, limits, intent="TRACK", frame=frame, control_dt=0.02)
            elapsed = 1000 * (time.perf_counter() - started)
            if index >= 6:
                if result is None or len(solves) != count + 1:
                    raise RuntimeError(f"Expected one successful solve per measured frame: {r.hold_reason}")
                records.append((elapsed, *solves[-1]))
                commands.append([result[name] for name in r.command_names])
                if not np.isfinite(commands[-1]).all():
                    raise RuntimeError("Nonfinite replay command")
                for name, value in result.items():
                    cap = (
                        min(r.settings.arm_velocity_limit, r.velocities[name])
                        if name in r.arm_names
                        else min(r.settings.hand_velocity_limit, r.profile.hands.velocities[name])
                    )
                    bounded = limits[name][0] <= value <= limits[name][1]
                    if not bounded or abs(value - previous[name]) > 0.02 * cap + 1e-6:
                        raise RuntimeError(f"Replay command violates bounds: {name}")
                # Telemetry is outside retarget timing for both implementations.
                fk = solver.forward_kinematics(measured | result)
                errors.append(
                    (
                        max(np.linalg.norm(p[:3, 3] - r.tasks[k].T_world_frame[:3, 3]) for k, p in fk.items()),
                        max(
                            np.arccos(
                                np.clip((np.trace(p[:3, :3].T @ r.tasks[k].T_world_frame[:3, :3]) - 1) / 2, -1, 1)
                            )
                            for k, p in fk.items()
                        ),
                    )
                )
            if result is not None:
                if not r.acknowledge(result):
                    raise RuntimeError("Replay acknowledgement failed")
                previous = result
                fraction = 0.25 if scenario == "servo_lag" else 1.0
                for name, value in result.items():
                    measured[name] += fraction * (value - measured[name])
        values = np.asarray(records)
        errors = np.asarray(errors)
        return dict(
            scenario=scenario,
            steps=steps,
            retarget_ms=dict(zip(("median", "p95", "max"), np.percentile(values[:, 0], [50, 95, 100]).tolist())),
            optimizer_ms=dict(zip(("median", "p95"), np.percentile(values[:, 1], [50, 95]).tolist())),
            iterations=dict(zip(("median", "p95"), np.percentile(values[:, 2], [50, 95]).tolist())),
            warm_frames=int(values[:, 3].sum()),
            over20_frames=int(np.count_nonzero(values[:, 0] > 20)),
            failed_cycles=0,
            capacities=[solver._rows, solver._soft_rows],
            position_error_m={"mean": float(errors[:, 0].mean()), "max": float(errors[:, 0].max())},
            orientation_error_rad={"mean": float(errors[:, 1].mean()), "max": float(errors[:, 1].max())},
        ), np.asarray(commands)
    finally:
        r.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Git revision before the optimization, e.g. HEAD before committing")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--robot", choices=("x2", "x2_omnihand"), default="x2_omnihand")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 4 or args.repeats < 1:
        parser.error("steps must be at least 4 and repeats must be positive")
    factories = {"current": None}
    if args.baseline_ref:
        factories = {"baseline": baseline_factory(args.baseline_ref), **factories}
    records = []
    for repeat in range(args.repeats):
        for scenario in ("stationary", "closing", "servo_lag"):
            commands, pair = {}, []
            variants = list(factories) if repeat % 2 == 0 else list(reversed(factories))
            for variant in variants:
                with implementation(factories[variant]):
                    record, commands[variant] = measure(scenario, args.steps, args.robot)
                record.update(variant=variant, repeat=repeat)
                pair.append(record)
            if "baseline" in commands:
                difference = float(np.max(np.abs(commands["current"] - commands["baseline"])))
                if difference > 1e-5:
                    raise RuntimeError(f"Replay command difference exceeds tolerance: {difference} rad")
                for record in pair:
                    record["max_command_difference_rad"] = difference
            for record in pair:
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.output:
        import casadi

        args.output.write_text(
            json.dumps(
                dict(
                    machine=platform.machine(),
                    python=platform.python_version(),
                    casadi=casadi.__version__,
                    baseline_ref=args.baseline_ref,
                    baseline_commit=(
                        subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
                        if args.baseline_ref
                        else None
                    ),
                    robot=args.robot,
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
