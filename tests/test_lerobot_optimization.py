"""Native regression checks for transactional IPOPT execution optimizations."""

from dataclasses import replace
import os
import pickle
import unittest
from unittest.mock import Mock, patch

import numpy as np

import test_ik_backends as contracts
from xrobotoolkit_teleop.ik.types import LinearConstraint


@unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires native LeRobot dependencies")
class LeRobotOptimizationTests(unittest.TestCase):
    def fixture(self):
        controller = contracts.BackendContracts.make_controller(self, "lerobot")
        problem = contracts.BackendContracts._tracking_problem(self, controller)
        return controller.backend, problem

    def test_warm_state_commits_only_on_acceptance_and_does_not_alias(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        self.assertIsNone(solver._warm)
        backend.accept(proposal, proposal.positions)
        self.assertIsNotNone(solver._warm)
        snapshot = solver.capture_state()
        saved = pickle.dumps(snapshot)
        solver._warm["q"][0] += 1
        self.assertEqual(pickle.dumps(snapshot), saved)
        solver.restore_state(snapshot)
        solver._warm["dual"][0] += 1
        self.assertEqual(pickle.dumps(snapshot), saved)
        solver.restore_state(snapshot)
        backend.reject(backend.propose(problem))
        self.assertEqual(pickle.dumps(solver.capture_state()), saved)
        solver.reset(frozenset([solver.joint_names[0]]))
        self.assertIsNone(solver._warm)

    def test_changed_applied_command_invalidates_warm_state(self):
        backend, problem = self.fixture()
        proposal = backend.propose(problem)
        applied = {n: (v + problem.previous[n]) / 2 for n, v in proposal.positions.items()}
        backend.accept(proposal, applied)
        self.assertIsNone(backend.solver._warm)

    def test_no_soft_graph_and_independent_capacity_growth(self):
        backend, problem = self.fixture()
        solver = backend.solver
        backend.propose(problem)
        self.assertEqual(solver._soft_rows, 0)
        self.assertEqual(solver.opti.nx, solver.model.nq)
        rows = solver._rows + 1
        hard = LinearConstraint(np.zeros((rows, solver.model.nq)), np.ones(rows))
        backend.propose(replace(problem, hard=hard))
        self.assertGreaterEqual(solver._rows, rows)
        self.assertEqual(solver._soft_rows, 0)
        hard.b[-1] = -1
        before = pickle.dumps(solver.capture_state())
        with self.assertRaises(RuntimeError):
            backend.propose(replace(problem, hard=hard))
        self.assertEqual(pickle.dumps(solver.capture_state()), before)

    def test_soft_padding_and_cached_variant_switch(self):
        backend, problem = self.fixture()
        solver = backend.solver
        plain = backend.propose(problem)
        backend.accept(plain, plain.positions)
        no_soft_function = solver._optimizer
        # Retain zero-Jacobian supplied rows, including a negative RHS.
        A = np.zeros((3, solver.model.nq))
        A[0, 0] = 1
        soft = LinearConstraint(A, np.array([10.0, -0.001, 1.0]))
        proposal = backend.propose(replace(problem, soft=soft))
        backend.accept(proposal, proposal.positions)
        self.assertEqual(solver._soft_rows, 4)
        self.assertAlmostEqual(solver._warm["slack"][3], 0, places=8)
        self.assertGreaterEqual(solver._warm["slack"][1], 0.001 - 1e-6)
        backend.propose(problem)
        self.assertIs(solver._optimizer, no_soft_function)
        self.assertEqual(solver._soft_rows, 0)
        self.assertEqual(len(solver._variants), 2)
        # Snapshot layout must remain compatible after switching variants.
        backend.propose(replace(problem, soft=soft))

    def test_converted_and_direct_calls_match_from_identical_state(self):
        backend, problem = self.fixture()
        solver = backend.solver
        for soft in (problem.soft, LinearConstraint(np.zeros((1, len(problem.previous))), np.array([-0.001]))):
            current = replace(problem, soft=soft)
            # Check both cold and accepted warm state with identical inputs.
            for warm in (False, True):
                if warm:
                    seed = backend.propose(current)
                    backend.accept(seed, seed.positions)
                converted = backend.propose(current)
                with patch.object(solver, "_function_calls", False):
                    direct = backend.propose(current)
                np.testing.assert_allclose(
                    list(converted.positions.values()), list(direct.positions.values()), atol=1e-8
                )

    def test_nonfinite_and_unsuccessful_outputs_restore_accepted_state(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        before = pickle.dumps(solver.capture_state())
        for result, success in (
            ([np.full(solver.model.nq, np.nan), np.zeros(solver.opti.ng)], True),
            ([np.zeros(solver.model.nq), np.zeros(solver.opti.ng)], False),
        ):
            fake = Mock(return_value=result)
            fake.stats.return_value = {"success": success}
            # Cache owns the callable selected before every solve.
            cached = solver._variants[False]
            with patch.dict(cached, _optimizer=fake), self.assertRaises(RuntimeError):
                backend.propose(problem)
            self.assertEqual(pickle.dumps(solver.capture_state()), before)

    def test_changed_rows_and_bounds_clear_corresponding_duals(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        solver._warm["dual"][:] = 123
        hard = LinearConstraint(problem.hard.A.copy(), problem.hard.b + 0.001)
        lower = problem.lower.copy()
        lower[0] += 1e-5
        callable_ = solver._optimizer
        spy = Mock(wraps=callable_)
        cached = solver._variants[False]
        with patch.dict(cached, _optimizer=spy):
            backend.propose(replace(problem, hard=hard, lower=lower))
        dual = spy.call_args.args[-1]
        self.assertEqual(dual[solver.indices[0]], 0)
        np.testing.assert_array_equal(dual[solver.model.nq :], 0)

    def test_engagement_and_home_invalidate_raw_initial_guess(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        for changed in (replace(problem, active=frozenset()), replace(problem, joint_target=problem.previous)):

            def fail(*args):
                self.assertIsNone(solver._warm)
                raise RuntimeError("probe")

            with patch.object(solver, "_run_optimizer", side_effect=fail) as run:
                with self.assertRaises(RuntimeError):
                    backend.propose(changed)
                run.assert_called_once()
            self.assertIsNotNone(solver._warm)  # Failed proposals restore accepted state.

    def test_raw_initial_guess_is_reused_and_clipped_to_current_bounds(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        solver._warm["q"][0] = problem.upper[0] + 1
        callable_ = solver._optimizer
        spy = Mock(wraps=callable_)
        with patch.dict(solver._variants[False], _optimizer=spy):
            backend.propose(problem)
        expected = np.clip(
            solver._warm["q"], problem.lower[np.argsort(solver.indices)], problem.upper[np.argsort(solver.indices)]
        )
        np.testing.assert_array_equal(spy.call_args.args[-2], expected)

    def test_iteration_limit_raises_and_rolls_back_converted_function(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        before = pickle.dumps(solver.capture_state())
        solver.opti.solver(
            "ipopt",
            {
                "ipopt": {"max_iter": 0, "tol": 1e-6, "print_level": 0, "sb": "yes"},
                "print_time": False,
                "error_on_fail": True,
            },
        )
        limited = solver.opti.to_function("limited_ik", solver._inputs + solver._guesses, solver._outputs)
        with patch.dict(solver._variants[False], _optimizer=limited), self.assertRaises(RuntimeError):
            backend.propose(problem)
        self.assertEqual(pickle.dumps(solver.capture_state()), before)
        self.assertFalse(solver.last_stats["success"])

    def test_float32_acknowledgement_preserves_warm_start(self):
        backend, problem = self.fixture()
        proposal = backend.propose(problem)
        rounded = {name: float(np.float32(value)) for name, value in proposal.positions.items()}
        backend.accept(proposal, rounded)
        self.assertIsNotNone(backend.solver._warm)

    def test_constant_soft_constraint_can_use_slack_without_a_jacobian(self):
        backend, problem = self.fixture()
        soft = LinearConstraint(np.zeros((1, len(problem.previous))), np.array([-0.001]))
        proposal = backend.propose(replace(problem, soft=soft))
        backend.accept(proposal, proposal.positions)
        self.assertGreaterEqual(backend.solver._warm["slack"][0], 0.001 - 1e-6)

    def test_malformed_problem_does_not_mutate_state_or_allocate_graphs(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        state = pickle.dumps(solver.capture_state())
        graph = solver._graph
        n = len(problem.previous)
        malformed = (
            replace(problem, lower=np.zeros((n, 1))),
            replace(problem, upper=np.full(n, np.nan)),
            replace(problem, upper=np.full(n, 1 + 1j)),
            replace(problem, lower=np.full(n, "0")),
            replace(problem, hard=LinearConstraint(np.ones((1, n), dtype=complex), np.ones(1))),
            replace(problem, soft=LinearConstraint(np.ones((1, n)), np.array(["1"]))),
            replace(problem, hard=LinearConstraint(np.ones(n), np.ones(2))),
            replace(problem, soft=LinearConstraint(np.zeros((2, n)), np.ones((2, 1)))),
            replace(problem, hard=LinearConstraint(np.full((1, n), np.inf), np.ones(1))),
            replace(problem, soft=LinearConstraint(np.zeros((1, n)), np.array([np.nan]))),
        )
        for invalid in malformed:
            with self.subTest(problem=invalid), self.assertRaises(ValueError):
                backend.propose(invalid)
            self.assertEqual(pickle.dumps(solver.capture_state()), state)
            self.assertIs(solver._graph, graph)

    def test_failed_graph_growth_restores_symbols_and_can_retry(self):
        backend, problem = self.fixture()
        solver = backend.solver
        proposal = backend.propose(problem)
        backend.accept(proposal, proposal.positions)
        graph, q, opti = solver._graph, solver.q, solver.opti
        state = pickle.dumps(solver.capture_state())
        rows = solver._rows + 1
        larger = replace(problem, hard=LinearConstraint(np.zeros((rows, len(problem.previous))), np.ones(rows)))

        # Fail after the builder has replaced live symbols, as a compiler failure would.
        def fail(*args):
            solver._build_optimizer(__import__("casadi"))
            raise RuntimeError("graph compilation failed")

        with patch.object(solver, "_build_constraint_optimizer", side_effect=fail), self.assertRaises(RuntimeError):
            backend.propose(larger)
        self.assertIs(solver._graph, graph)
        self.assertIs(solver.opti, opti)
        self.assertIs(solver.q, q)
        self.assertEqual(pickle.dumps(solver.capture_state()), state)
        backend.propose(larger)

    def test_switching_to_no_soft_graph_clears_soft_symbols(self):
        backend, problem = self.fixture()
        soft = LinearConstraint(np.zeros((1, len(problem.previous))), np.ones(1))
        backend.propose(replace(problem, soft=soft))
        self.assertIsNotNone(backend.solver.slack)
        backend.propose(problem)
        self.assertIsNone(backend.solver.slack)
        self.assertIsNone(backend.solver.soft_A)
        self.assertIsNone(backend.solver.soft_b)
        self.assertIsNone(backend.solver.slack_upper)

    def test_incompatible_bounds_are_rejected_before_graph_selection(self):
        backend, problem = self.fixture()
        solver = backend.solver
        before = pickle.dumps(solver.capture_state())
        invalid = replace(problem, upper=problem.lower - 1)
        with patch.object(solver, "_select_optimizer", side_effect=AssertionError("unexpected graph allocation")):
            with self.assertRaisesRegex(RuntimeError, "Incompatible joint command bounds"):
                backend.propose(invalid)
        self.assertEqual(pickle.dumps(solver.capture_state()), before)

    def test_padding_activation_clears_changed_slack_bound_multiplier(self):
        backend, problem = self.fixture()
        solver = backend.solver
        n = len(problem.previous)
        soft = LinearConstraint(np.zeros((3, n)), np.ones(3))
        proposal = backend.propose(replace(problem, soft=soft))
        backend.accept(proposal, proposal.positions)
        old_dual = solver._warm["dual"].copy()
        bound_start = solver.model.nq + solver._rows + solver._soft_rows
        solver._warm["dual"][bound_start + 3] = 123.0
        old_dual[bound_start + 3] = 0.0
        spy = Mock(wraps=solver._optimizer)
        # The fourth row equals the previous padding, but its upper bound changes.
        expanded = LinearConstraint(np.zeros((4, n)), np.ones(4))
        with patch.dict(solver._variants[True], _optimizer=spy):
            backend.propose(replace(problem, soft=expanded))
        np.testing.assert_array_equal(spy.call_args.args[-1], old_dual)
