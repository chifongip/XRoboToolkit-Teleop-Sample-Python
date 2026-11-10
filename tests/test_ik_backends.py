"""Backend contracts, independent loading, shared geometry, and visualization."""

from dataclasses import replace
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from test_agibot_x2_simulation import ASSETS, FakeXR
from xrobotoolkit_teleop.ik import BACKENDS
from xrobotoolkit_teleop.ik.backend import IKSolver, TransactionalBackend
from xrobotoolkit_teleop.ik.registry import BackendRegistration
from xrobotoolkit_teleop.ik.types import LinearConstraint, VisualizationSnapshot
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController


class DummySolver(IKSolver):
    """Minimal third method: offer a bounded hold/joint target if feasible."""

    def __init__(self, spec):
        self.spec = spec

    def solve(self, problem):
        q = np.array([(problem.joint_target or problem.previous)[n] for n in self.spec.joints])
        q = np.clip(q, problem.lower, problem.upper)
        if np.any(problem.hard.A @ q > problem.hard.b + 1e-6):
            raise RuntimeError("Dummy cannot satisfy this pose")
        return dict(zip(self.spec.joints, q))


class DummyBackend(TransactionalBackend):
    def __init__(self, spec):
        from xrobotoolkit_teleop.ik.placo_model import PlacoModel

        super().__init__(spec, PlacoModel(spec), DummySolver(spec))


class BackendContracts(unittest.TestCase):
    def make_controller(self, backend="placo", hands=False, **kwargs):
        scene, urdf = ("scene_omnihand.xml", "x2_omnihand_ik.urdf") if hands else ("scene.xml", "x2_fixed.urdf")
        with patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient", FakeXR):
            c = AgibotX2TeleopController(str(ASSETS / scene), str(ASSETS / urdf), ik_backend=backend, **kwargs)
        c.xr_client.grips.update(left_trigger=0.0, right_trigger=0.0)
        self.addCleanup(c.close)
        return c

    def test_registered_third_method_needs_no_controller_changes(self):
        with patch.dict(BACKENDS, dummy=BackendRegistration("test_ik_backends:DummyBackend", "Test method")):
            c = self.make_controller("dummy")
            c.xr_client.grips["left_grip"] = 1
            before = c.mj_data.ctrl.copy()
            c.step_control()
            self.assertTrue(c._ik_valid, c.status)
            np.testing.assert_array_equal(c.mj_data.ctrl, before)

    def test_unsupported_capabilities_fail_before_xr_connection(self):
        with (
            patch.dict(BACKENDS, dummy=BackendRegistration("test_ik_backends:DummyBackend", "Incomplete", frozenset())),
            patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient") as xr,
        ):
            with self.assertRaisesRegex(ValueError, "required objectives"):
                AgibotX2TeleopController(str(ASSETS / "scene.xml"), str(ASSETS / "x2_fixed.urdf"), ik_backend="dummy")
            xr.assert_not_called()

    def test_visualization_uses_the_shared_renderer(self):
        with patch("xrobotoolkit_teleop.ik.visualization.IKViewer") as viewer:
            c = self.make_controller(visualize_ik=True)
            c.step_control()
            viewer.assert_called_once()
            self.assertEqual(viewer.return_value.update.call_count, 1)
            snapshot = viewer.return_value.update.call_args.args[0]
            self.assertIsInstance(snapshot, VisualizationSnapshot)
            for i, name in c.arm_actuators.items():
                self.assertEqual(snapshot.positions[name], c.mj_data.ctrl[i])

    def test_headless_does_not_create_a_viewer(self):
        with patch("xrobotoolkit_teleop.ik.visualization.IKViewer") as viewer:
            c = self.make_controller()
            c.step_control()
            viewer.assert_not_called()

    def _tracking_problem(self, controller):
        controller.active.update(left_arm=True, right_arm=True)
        controller._update_robot_state()
        problem = controller._make_problem(controller._command_bounds())
        targets = {name: pose.copy() for name, pose in problem.targets.items()}
        targets["left_arm"][0, 3] += 0.02
        return replace(problem, targets=targets)

    def test_proposals_preserve_model_and_solver_state_until_acceptance(self):
        c = self.make_controller()
        problem = self._tracking_problem(c)
        configuration = c.robot.state.q.copy()
        history = pickle.dumps(c.backend.solver.capture_state())
        proposal = c.backend.propose(problem)
        np.testing.assert_array_equal(c.robot.state.q, configuration)
        self.assertEqual(pickle.dumps(c.backend.solver.capture_state()), history)
        self.assertGreater(max(abs(proposal.positions[n] - problem.previous[n]) for n in problem.previous), 1e-5)
        c.backend.reject(proposal)
        np.testing.assert_array_equal(c.robot.state.q, configuration)
        self.assertEqual(pickle.dumps(c.backend.solver.capture_state()), history)
        proposal = c.backend.propose(problem)
        # The final collision gate can shorten a step; accept the applied command.
        applied = {n: (proposal.positions[n] + problem.previous[n]) / 2 for n in problem.previous}
        c.backend.accept(proposal, applied)
        for name, value in applied.items():
            self.assertEqual(c.robot.get_joint(name), value)

    def test_solver_failure_restores_model_and_accepted_history(self):
        c = self.make_controller()
        problem = self._tracking_problem(c)
        proposal = c.backend.propose(problem)
        c.backend.accept(proposal, proposal.positions)
        configuration = c.robot.state.q.copy()
        history = pickle.dumps(c.backend.solver.capture_state())
        pose = c.robot.get_T_world_frame("left_hand_center")

        def fail(problem):
            c.robot.set_joint("left_elbow_joint", -0.3)
            c.robot.update_kinematics()
            c.backend.solver.reset()
            raise RuntimeError("solver failure")

        with patch.object(c.backend.solver, "solve", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "solver failure"):
                c.backend.propose(problem)
        np.testing.assert_array_equal(c.robot.state.q, configuration)
        np.testing.assert_allclose(c.robot.get_T_world_frame("left_hand_center"), pose, atol=1e-12)
        self.assertEqual(pickle.dumps(c.backend.solver.capture_state()), history)

    def test_malformed_solver_output_is_rejected_without_changing_state(self):
        c = self.make_controller()
        problem = self._tracking_problem(c)
        configuration = c.robot.state.q.copy()
        history = pickle.dumps(c.backend.solver.capture_state())
        name = c.backend.spec.joints[0]
        cases = []
        for value in (np.nan, [problem.previous[name]], "invalid", problem.upper[0] + 1):
            positions = problem.previous.copy()
            positions[name] = value
            cases.append(positions)
        missing = problem.previous.copy()
        del missing[name]
        cases.extend((missing, {**problem.previous, "unexpected": 0.0}, {n: [v] for n, v in problem.previous.items()}))
        for positions in cases:
            with self.subTest(positions=positions), patch.object(c.backend.solver, "solve", return_value=positions):
                with self.assertRaises(RuntimeError):
                    c.backend.propose(problem)
                np.testing.assert_array_equal(c.robot.state.q, configuration)
                self.assertEqual(pickle.dumps(c.backend.solver.capture_state()), history)

    def test_backend_close_is_idempotent_and_rejects_further_operations(self):
        c = self.make_controller()
        problem = self._tracking_problem(c)
        proposal = c.backend.propose(problem)
        with patch.object(c.backend.solver, "close", wraps=c.backend.solver.close) as close:
            c.backend.close()
            c.backend.close()
            close.assert_called_once()
        operations = (
            lambda: c.backend.propose(problem),
            lambda: c.backend.accept(proposal, proposal.positions),
            lambda: c.backend.reject(proposal),
            lambda: c.backend.reset(),
        )
        for operation in operations:
            with self.assertRaisesRegex(RuntimeError, "backend is closed"):
                operation()

    def test_invalid_urdf_limits_fail_before_backend_or_xr_startup(self):
        for key, value in (("lower", "nan"), ("upper", "inf"), ("velocity", "nan"), ("velocity", "0")):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                root = ET.parse(ASSETS / "x2_fixed.urdf")
                root.find("joint[@name='left_elbow_joint']/limit").set(key, value)
                path = Path(tmp) / "invalid.urdf"
                root.write(path)
                with (
                    patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.create_backend") as backend,
                    patch("xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient") as xr,
                ):
                    with self.assertRaisesRegex(ValueError, "Invalid URDF limits"):
                        AgibotX2TeleopController(str(ASSETS / "scene.xml"), str(path))
                    backend.assert_not_called()
                    xr.assert_not_called()

    def test_large_finite_xr_quaternions_normalize_without_overflow(self):
        c = self.make_controller()
        quaternion = np.array([0.1, -0.2, 0.3, 0.4])
        c.xr_client.poses["left_controller"][3:] = quaternion * 1e308
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            pose = c._read_valid_xr_pose("left_controller")
        np.testing.assert_allclose(pose[3:], quaternion / np.linalg.norm(quaternion), atol=1e-15)

    def test_cleanup_continues_after_viewer_failure_and_is_idempotent(self):
        c = self.make_controller()
        c._ik_viewer = Mock()
        c._ik_viewer.close.side_effect = RuntimeError("viewer disconnected")
        with patch.object(c.backend, "close") as backend, patch.object(c.xr_client, "close") as xr:
            with self.assertRaisesRegex(RuntimeError, "viewer disconnected"):
                c.close()
            backend.assert_called_once()
            xr.assert_called_once()
            c.close()
            backend.assert_called_once()

    def test_renderer_loads_both_bundled_models(self):
        from xrobotoolkit_teleop.ik.visualization import IKViewer

        for hands in (False, True):
            c = self.make_controller(hands=hands)
            viewer = Mock()
            viewer.__getitem__ = Mock(return_value=Mock())
            renderer = IKViewer(c.robot_urdf_path, viewer=viewer, open_browser=False)
            renderer.update(c.visualization_snapshot())
            self.assertGreater(len(renderer.visuals), 10)
            renderer.close()

    def test_renderer_displays_failure_and_accepted_geometry(self):
        from xrobotoolkit_teleop.ik.visualization import IKViewer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot.urdf"
            path.write_text(
                '<robot name="test"><link name="body"><visual><geometry><box size="1 1 1"/>'
                "</geometry></visual></link></robot>"
            )
            viewer = Mock()
            node = Mock()
            viewer.__getitem__ = Mock(return_value=node)
            renderer = IKViewer(path, viewer=viewer, open_browser=False)
            pose = np.eye(4)
            pose[0, 3] = 0.4
            renderer.update(
                VisualizationSnapshot({"body": pose}, {"target": pose}, {}, "hold: rejected", {"tip": pose})
            )
            np.testing.assert_array_equal(node.set_transform.call_args_list[0].args[0], pose)
            self.assertEqual(renderer._status, "hold: rejected")
            self.assertIn("status", [call.args[0] for call in viewer.__getitem__.call_args_list])

    def test_owned_renderer_cleanup_releases_server_when_delete_fails(self):
        import zmq
        from xrobotoolkit_teleop.ik.visualization import IKViewer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.urdf"
            path.write_text('<robot name="empty"><link name="body"/></robot>')
            with patch("xrobotoolkit_teleop.ik.visualization.meshcat.Visualizer") as factory:
                renderer = IKViewer(path, open_browser=False)
            viewer = factory.return_value
            viewer.delete.side_effect = zmq.Again()
            with self.assertRaises(zmq.Again):
                renderer.close()
            viewer.window.zmq_socket.close.assert_called_once_with(linger=0)
            viewer.window.server_proc.terminate.assert_called_once()
            viewer.window.server_proc.wait.assert_called_once_with(timeout=2)
            self.assertEqual(viewer.window.zmq_socket.setsockopt.call_count, 2)
            renderer.close()
            viewer.delete.assert_called_once()


@unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires optional Pinocchio/CasADi build")
class NativeBackendContracts(BackendContracts):
    def make_controller(self, backend="lerobot", hands=False, **kwargs):
        return super().make_controller(backend, hands, **kwargs)

    def test_restored_filter_state_does_not_alias_its_snapshot(self):
        c = self.make_controller()
        problem = self._tracking_problem(c)
        proposal = c.backend.propose(problem)
        c.backend.accept(proposal, proposal.positions)
        state = c.backend.solver.capture_state()
        snapshot = pickle.dumps(state)
        c.backend.solver.restore_state(state)
        c.backend.reset(frozenset(c.backend.spec.joints))
        # Mutating live state must not change a snapshot that could be restored.
        c.backend.solver._history[0][0] += 0.1
        c.backend.solver._previous[0] += 0.1
        self.assertEqual(pickle.dumps(state), snapshot)

    def test_obstructed_homing_reaches_home_with_and_without_servo_dynamics(self):
        # Reuse the same blocked-path fixture and per-step invariants as Placo.
        from test_agibot_x2_collisions import X2CollisionTests

        case = X2CollisionTests("test_return_avoids_obstructed_direct_path_and_reaches_home")
        case.make_controller = lambda omnihand=False, **kw: self.make_controller("lerobot", omnihand, **kw)
        case.test_return_avoids_obstructed_direct_path_and_reaches_home()

    def test_constraint_capacity_grows_without_dropping_rows(self):
        c = self.make_controller()
        c._update_robot_state()
        problem = c._make_problem(c._command_bounds())
        rows = c.backend.solver._rows + 1
        A, b = np.zeros((rows, len(c.backend.spec.joints))), np.ones(rows)
        b[-1] = -1  # An impossible final row must be retained after growth.
        with self.assertRaises(RuntimeError):
            c.backend.propose(replace(problem, hard=LinearConstraint(A, b)))
        self.assertGreaterEqual(c.backend.solver._rows, rows)
        self.assertIsNone(c.backend.solver._previous)
        proposal = c.backend.propose(problem)
        np.testing.assert_allclose(list(proposal.positions.values()), list(problem.previous.values()), atol=1e-8)

    def test_backends_load_independently_in_fresh_processes(self):
        root = Path(__file__).resolve().parents[1]
        for backend, blocked in (("lerobot", ("placo", "placo_utils")), ("placo", ("pinocchio", "casadi", "coal"))):
            program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError("Unexpected dependency: " + fullname)
sys.meta_path.insert(0, Block())
from scripts.misc.benchmark_x2_ik import ReplayXR
from unittest.mock import patch
from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController
with patch('xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller.XrClient', ReplayXR):
    c=AgibotX2TeleopController('assets/agibot/x2/scene.xml','assets/agibot/x2/x2_fixed.urdf',ik_backend=BACKEND)
try:
    c.step_control()
    assert c._ik_valid, c.status
finally:
    c.backend.close(); c.xr_client.close()
"""
            # Load the replay by file path: some installations also contain a
            # top-level scripts package unrelated to this repository.
            program = program.replace(
                "from scripts.misc.benchmark_x2_ik import ReplayXR",
                """
import importlib.util
spec=importlib.util.spec_from_file_location('replay','scripts/misc/benchmark_x2_ik.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
ReplayXR=module.ReplayXR""",
            )
            result = subprocess.run(
                [sys.executable, "-c", f"BLOCKED={blocked!r}\nBACKEND={backend!r}\n" + program],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_collision_adapters_agree_on_distances_placements_and_gradients(self):
        for hands in (False, True):
            controllers = [self.make_controller(name, hands) for name in ("placo", "lerobot")]
            for pitch in (0.35, 1.7, 2.0):
                records = []
                for c in controllers:
                    c.robot.set_joint("left_shoulder_pitch_joint", pitch)
                    # Parallel flat faces have nonunique nearest points and
                    # valid but different subgradients. Exercise differentiable
                    # placements for Jacobian parity instead.
                    c.robot.set_joint("left_wrist_roll_joint", 0.17)
                    c.robot.set_joint("right_wrist_roll_joint", -0.21)
                    c.robot.update_kinematics()
                    c.collision_checker.clear_cache()
                    distances = c.collision_checker.distances()
                    near, J = c.collision_checker.linearized_clearance(c.collision_trigger)
                    rows = dict(zip(np.flatnonzero(distances < c.collision_trigger), J))
                    objects = c.robot.collision_model.objects
                    pairs = {}
                    for i, (a, b) in enumerate(c.collision_checker.pair_indices):
                        key = tuple(sorted((objects[a].name, objects[b].name)))
                        witness = c.collision_checker._witnesses[i]
                        points = np.array([witness.pointA, witness.pointB])
                        if objects[a].name > objects[b].name:
                            points = points[::-1]
                        pairs[key] = (
                            distances[i],
                            (
                                None
                                if i not in rows
                                else np.array(
                                    [rows[i][c.robot.get_joint_v_offset(n)] for n in controllers[0].backend.spec.joints]
                                )
                            ),
                            points,
                        )
                    placements = c.robot.collision_placements(c.robot.state.q)
                    records.append((pairs, {obj.name: placements[i] for i, obj in enumerate(objects)}))
                self.assertEqual(records[0][0].keys(), records[1][0].keys())
                for key, (distance, gradient, points) in records[0][0].items():
                    other, other_gradient, other_points = records[1][0][key]
                    self.assertAlmostEqual(distance, other, delta=2e-6)
                    if gradient is not None and np.allclose(points, other_points, atol=1e-5, rtol=0):
                        np.testing.assert_allclose(gradient, other_gradient, atol=2e-4, rtol=0)
                    else:
                        # On flat/symmetric surfaces the witnesses and contact
                        # subgradient are nonunique. Both must realize the same
                        # signed-distance magnitude; do not require identical points.
                        self.assertAlmostEqual(np.linalg.norm(points[1] - points[0]), abs(distance), delta=1e-5)
                        self.assertAlmostEqual(
                            np.linalg.norm(other_points[1] - other_points[0]), abs(other), delta=1e-5
                        )
                for key, transform in records[0][1].items():
                    np.testing.assert_allclose(transform, records[1][1][key], atol=1e-6, rtol=0)
                for joint in controllers[0].backend.spec.joints:
                    matrices = []
                    for c in controllers:
                        columns = [c.robot.get_joint_v_offset(n) for n in controllers[0].backend.spec.joints]
                        matrices.append(c.robot.joint_jacobian(joint, "world")[:, columns])
                    np.testing.assert_allclose(*matrices, atol=1e-6, rtol=0)

    def test_velocity_bounds_are_independent_of_servo_lag(self):
        for name in ("placo", "lerobot"):
            for frequency in (25.0, 100.0):
                c = self.make_controller(name, control_frequency=frequency, arm_velocity_limit=0.05)
                c.xr_client.grips["left_grip"] = 1
                c.step_control()
                c.mj_data.qpos[c.mj_model.joint("left_elbow_joint").qposadr[0]] -= 0.03
                mujoco.mj_forward(c.mj_model, c.mj_data)
                before = c.mj_data.ctrl.copy()
                c.step_control()
                self.assertTrue(c._ik_valid, c.status)
                self.assertLessEqual(np.max(np.abs(c.mj_data.ctrl - before)), 0.05 / frequency + 1e-9)

    def test_rejected_combined_proposals_do_not_commit_history(self):
        c = self.make_controller("lerobot", True)
        c.xr_client.grips["left_grip"] = 1
        c._update_ik()
        self.assertIsNotNone(c._pending_proposal)
        self.assertIsNone(c.backend.solver._previous)
        c._send_command()
        history = np.array(c.backend.solver._history)
        reference = c.backend.solver._previous.copy()
        c.xr_client.poses["left_controller"][2] -= 0.02
        c._update_ik()

        def reject(previous):
            c.mj_data.ctrl[:] = previous
            c._ik_valid = False

        with patch.object(c, "_validate_combined_commands", side_effect=reject):
            c._send_command()
        self.assertFalse(c._ik_valid)
        np.testing.assert_array_equal(np.array(c.backend.solver._history), history)
        np.testing.assert_array_equal(c.backend.solver._previous, reference)

    def test_penetration_recovery_preserves_clearance_and_speed_bounds(self):
        for backend in ("placo", "lerobot"):
            for hands in (False, True):
                c = self.make_controller(backend, hands)
                for name, value in (("left_shoulder_pitch_joint", 2.0), ("left_shoulder_roll_joint", -0.06)):
                    c.mj_data.qpos[c.mj_model.joint(name).qposadr[0]] = value
                for a, name in c.arm_actuators.items():
                    c.mj_data.ctrl[a] = c.mj_data.qpos[c.mj_model.joint(name).qposadr[0]]
                mujoco.mj_forward(c.mj_model, c.mj_data)
                c._update_robot_state()
                initial = c.collision_checker.distances()
                self.assertLess(initial.min(), 0)
                c.xr_client.grips["left_grip"] = 1
                for _ in range(6):
                    c._update_robot_state()
                    before = c.collision_checker.distances()
                    commands = c.mj_data.ctrl.copy()
                    c._update_ik()
                    self.assertTrue(c._ik_valid, c.status)
                    c._send_command()
                    self.assertTrue(c._ik_valid, c.status)
                    self.assertTrue(
                        np.all(c.collision_checker.distances() >= np.minimum(before, c.collision_margin) - 1e-6)
                    )
                    for a, name in c.arm_actuators.items():
                        speed = c.robot.model.velocity_limit[c.robot.get_joint_v_offset(name)]
                        self.assertLessEqual(abs(c.mj_data.ctrl[a] - commands[a]), speed * c.dt + 1e-8)
                        c.mj_data.qpos[c.mj_model.joint(name).qposadr[0]] = c.mj_data.ctrl[a]
                    mujoco.mj_forward(c.mj_model, c.mj_data)
                self.assertGreater(c.collision_checker.distances().min(), 0)
