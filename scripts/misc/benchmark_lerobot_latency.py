"""Measure collision-disabled LeRobot retargeting without XR or robot commands.

Optionally compare the working tree against a Git revision's retargeter, solver,
transaction backend, and numeric model. Run from the repository root in the native environment.
"""

import argparse
from contextlib import contextmanager, ExitStack
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
    """Load a compatible revision's backend, model, solver, and retargeter."""
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

    path = "xrobotoolkit_teleop/integrations/retargeter.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)
    module = ModuleType("xrobotoolkit_teleop.integrations._latency_baseline_retargeter")
    module.__package__ = "xrobotoolkit_teleop.integrations"
    # dataclass resolves annotations through the defining module.
    import sys

    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
    return make, module.ExternalStateRetargeter


@contextmanager
def implementation(factory):
    if factory is None:
        yield
    else:
        make, retargeter = factory
        with ExitStack() as stack:
            stack.enter_context(patch.object(lerobot_backend, "LeRobotBackend", make))
            stack.enter_context(patch.dict(globals(), ExternalStateRetargeter=retargeter))
            yield


def measure(scenario, steps, robot, active_arms="both", ack_every=1, ack_delay_steps=0):
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
        pending = None
        completions = 0
        streaming = ack_every != 1 or ack_delay_steps != 0

        def complete(transaction, positions):
            if not r.complete_proposal(transaction, positions):
                raise RuntimeError("Replay acknowledgement failed")
            return positions

        def run(*args):
            warm = solver._warm is not None
            started = time.perf_counter()
            result = native_run(*args)
            solves.append((1000 * (time.perf_counter() - started), solver.last_stats["iter_count"], warm))
            return result

        solver._run_optimizer = run
        for index in range(steps + 6):
            if pending is not None and pending[0] <= index:
                previous = complete(*pending[1:])
                pending = None
                completions += int(index >= 6)
            fraction = 0.25 if scenario == "servo_lag" else 1.0
            for name, value in previous.items():
                measured[name] += fraction * (value - measured[name])
            phase = max(0, index - 5) / steps * 2 * np.pi
            for side, sign in (("left", 1), ("right", -1)):
                poses[f"{side}_controller"][0] = sign * 0.015 * np.sin(phase)
            trigger = 0.5 * (1 - np.cos(phase)) if scenario == "closing" else 0.0
            frame = XRFrame(
                index + 1,
                poses,
                {side: float(index >= 2 and active_arms in ("both", side)) for side in ("left", "right")},
                triggers=dict.fromkeys(("left", "right"), trigger),
            )
            count = len(solves)
            started = time.perf_counter()
            result = r.step(
                measured, previous, home, limits, intent="TRACK", frame=frame, control_dt=0.02, streaming=streaming
            )
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
                    if name in r.arm_names and not r.active[r._joint_groups[name]] and value != previous[name]:
                        raise RuntimeError(f"Replay changed an inactive arm command: {name}")
                # Telemetry is outside retarget timing for both implementations.
                fk = {k: p for k, p in solver.forward_kinematics(measured | result).items() if r.active[k]}
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
                transaction = r.detach_proposal()
                if pending is None and index % ack_every == 0:
                    if ack_delay_steps:
                        pending = (index + ack_delay_steps, transaction, result)
                    else:
                        previous = complete(transaction, result)
                        completions += int(index >= 6)
                else:
                    r.discard_proposal(transaction)
        if pending is not None:
            r.discard_proposal(pending[1])
        values = np.asarray(records)
        errors = np.asarray(errors)
        return dict(
            scenario=scenario,
            active_arms=active_arms,
            ack_every=ack_every,
            ack_delay_steps=ack_delay_steps,
            completed_frames=completions,
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
    parser.add_argument("--active-arms", choices=("both", "left", "right"), default="both")
    parser.add_argument("--ack-every", type=int, default=1, help="Select one proposal every N steps for application")
    parser.add_argument("--ack-delay-steps", type=int, default=0, help="Delay selected application by N control steps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 4 or args.repeats < 1 or args.ack_every < 1 or args.ack_delay_steps < 0:
        parser.error("steps must be at least 4; repeats/ack-every positive; ack-delay-steps nonnegative")
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
                    record, commands[variant] = measure(
                        scenario, args.steps, args.robot, args.active_arms, args.ack_every, args.ack_delay_steps
                    )
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
                    active_arms=args.active_arms,
                    ack_every=args.ack_every,
                    ack_delay_steps=args.ack_delay_steps,
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
