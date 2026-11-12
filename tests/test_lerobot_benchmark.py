"""Benchmark accounting checks without optional native solver dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "scripts/misc/benchmark_lerobot_solver.py"
SPEC = importlib.util.spec_from_file_location("benchmark_lerobot_solver", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class FakeSolver:
    _warm_starts = _compact_slack = _function_calls = False
    _rows, _soft_rows = 16, 0

    def __init__(self):
        self.opti = SimpleNamespace(stats=lambda: {"iter_count": 0})

    def solve(self, problem):
        return {"axis": problem.command}

    def forward_kinematics(self, positions):
        pose = np.eye(4)
        pose[0, 3] = positions["axis"]
        return {"tool": pose}


class LeRobotBenchmarkTests(unittest.TestCase):
    def test_benchmark_import_ignores_foreign_scripts_package(self):
        module = importlib.util.module_from_spec(SPEC)
        with patch.dict(sys.modules, {"scripts": None}):
            SPEC.loader.exec_module(module)
        self.assertTrue(callable(module.replay))

    def measure(self, attempts=1, held=False):
        def replay(scenario, selective, steps, backend, observe_cycle):
            solver = benchmark.lerobot_backend.LeRobotIKSolver()
            for index in range(steps + 6):
                problem = SimpleNamespace(
                    measured={"axis": 0.0},
                    previous={"axis": 0.0},
                    targets={"tool": np.eye(4)},
                    command=100.0 if index < 6 else 0.0,
                )
                result = None
                count = attempts if index == 6 else 1
                if not (held and index == 6):
                    for _ in range(count):
                        result = solver.solve(problem)
                observe_cycle(index, result)
            return {"failed_cycles": 0}

        with (
            patch.object(benchmark.lerobot_solver, "LeRobotIKSolver", FakeSolver),
            patch.object(benchmark, "replay", side_effect=replay),
        ):
            return benchmark.measure("stationary", "all", 4, None)

    def test_warmup_records_are_excluded_by_cycle(self):
        result = self.measure()
        self.assertEqual(result["position_error_m"], {"mean": 0.0, "max": 0.0})
        self.assertEqual(result["capacities_max"], [16, 0])

    def test_retry_is_not_silently_counted_as_another_measured_cycle(self):
        with self.assertRaisesRegex(RuntimeError, "one successful solve per measured cycle"):
            self.measure(attempts=2)

    def test_held_cycle_does_not_reuse_an_earlier_successful_solve(self):
        with self.assertRaisesRegex(RuntimeError, "one successful solve per measured cycle"):
            self.measure(held=True)
