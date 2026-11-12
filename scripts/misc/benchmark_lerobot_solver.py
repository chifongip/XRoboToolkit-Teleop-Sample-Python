"""Sequential IPOPT ablations with exhaustive command-path validation.

Use --baseline-source with a saved pre-change lerobot_solver.py for an exact
baseline. Each output line describes one scenario, variant, and repeat.
"""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import time
from types import ModuleType
from unittest.mock import patch

import numpy as np

from xrobotoolkit_teleop.ik import lerobot_backend, lerobot_solver

# Resolve the sibling helper by file: ROS also installs a top-level scripts package.
_replay_spec = importlib.util.spec_from_file_location(
    "_benchmark_bridge_collisions", Path(__file__).with_name("benchmark_bridge_collisions.py")
)
_replay_module = importlib.util.module_from_spec(_replay_spec)
_replay_spec.loader.exec_module(_replay_module)
replay = _replay_module.replay


def measure(scenario, variant, steps, baseline):
    cls = baseline if variant == "baseline" else lerobot_solver.LeRobotIKSolver
    records = []
    measured_records = []
    cycle_start = 0
    original = cls.solve

    def solve(self, problem):
        started = time.perf_counter()
        result = original(self, problem)
        elapsed = 1000 * (time.perf_counter() - started)
        stats = self.last_stats if hasattr(self, "last_stats") else self.opti.stats()
        poses = self.forward_kinematics(problem.measured | result)
        errors = [np.linalg.norm(pose[:3, 3] - problem.targets[key][:3, 3]) for key, pose in poses.items()]
        rotations = [
            np.arccos(np.clip((np.trace(pose[:3, :3].T @ problem.targets[key][:3, :3]) - 1) / 2, -1, 1))
            for key, pose in poses.items()
        ]
        records.append(
            (
                elapsed,
                stats["iter_count"],
                self._rows,
                getattr(self, "_soft_rows", self._rows),
                max(errors),
                max(rotations),
                max(abs(result[n] - problem.previous[n]) for n in result),
                1000 * sum(v for k, v in stats.items() if k.startswith("t_wall_nlp_")),
            )
        )
        return result

    def observe_cycle(index, result):
        nonlocal cycle_start
        cycle_records = records[cycle_start:]
        cycle_start = len(records)
        if index >= 6:
            if result is None or len(cycle_records) != 1:
                raise RuntimeError("Solver benchmark requires one successful solve per measured cycle")
            measured_records.append(cycle_records[0])

    with ExitStack() as stack:
        stack.enter_context(patch.object(lerobot_backend, "LeRobotIKSolver", cls))
        if variant != "baseline":
            for attr, enabled in (
                ("_warm_starts", variant in ("warm", "all")),
                ("_compact_slack", variant in ("slack", "all")),
                ("_function_calls", variant in ("function", "all")),
            ):
                stack.enter_context(patch.object(cls, attr, enabled))
        stack.enter_context(patch.object(cls, "solve", solve))
        summary = replay(scenario, True, steps, "lerobot", observe_cycle=observe_cycle)
    r = np.asarray(measured_records)
    if len(r) != steps:
        raise RuntimeError("Expected one successful solve per measured frame")
    return dict(
        summary,
        variant=variant,
        solver_ms=dict(zip(("median", "p95"), np.percentile(r[:, 0], [50, 95]).tolist())),
        iterations=dict(zip(("median", "p95"), np.percentile(r[:, 1], [50, 95]).tolist())),
        capacities_max=r[:, 2:4].max(axis=0).astype(int).tolist(),
        evaluation_ms=dict(zip(("median", "p95"), np.percentile(r[:, 7], [50, 95]).tolist())),
        position_error_m={"mean": float(r[:, 4].mean()), "max": float(r[:, 4].max())},
        orientation_error_rad={"mean": float(r[:, 5].mean()), "max": float(r[:, 5].max())},
        max_command_step_rad=float(r[:, 6].max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "warm", "slack", "function", "all"),
        default=["baseline", "warm", "slack", "function", "all"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("stationary", "closing", "servo_lag", "recovery"),
        default=["stationary", "closing", "servo_lag", "recovery"],
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 4 or args.repeats < 1:
        parser.error("steps must be at least 4 and repeats at least 1")
    baseline = None
    if "baseline" in args.variants:
        if args.baseline_source is None:
            parser.error("baseline requires --baseline-source pointing to the pre-change solver")
        module = ModuleType("xrobotoolkit_teleop.ik._benchmark_baseline")
        module.__package__ = "xrobotoolkit_teleop.ik"
        exec(compile(args.baseline_source.read_text(), str(args.baseline_source), "exec"), module.__dict__)
        baseline = module.LeRobotIKSolver
    for repeat in range(args.repeats):
        for scenario in args.scenarios:
            variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                print(json.dumps(dict(repeat=repeat, **measure(scenario, variant, args.steps, baseline))), flush=True)


if __name__ == "__main__":
    main()
