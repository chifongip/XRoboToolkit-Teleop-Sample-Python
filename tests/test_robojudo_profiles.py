"""Profile extension and acknowledged solver-history tests without XR hardware."""

import copy
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_robojudo_x2 import MemoryContext
from xrobotoolkit_teleop.common.x2_defaults import DEFAULTS
from xrobotoolkit_teleop.ik.backend import IKSolver, TransactionalBackend
from xrobotoolkit_teleop.ik.placo_model import PlacoModel
from xrobotoolkit_teleop.ik.types import IKProblem, LinearConstraint, RobotSpec
from xrobotoolkit_teleop.integrations.retargeter import XRFrame
from xrobotoolkit_teleop.integrations.robojudo import RoboJuDoBridge
from xrobotoolkit_teleop.integrations.robot_profiles import PROFILES, ControlGroup, RobotProfile


class HistorySolver(IKSolver):
    def __init__(self):
        self.history = 0
        self.applied = None

    def solve(self, problem):
        self.history += 1
        return {"axis": float(np.clip(problem.previous["axis"] + 0.001, problem.lower[0], problem.upper[0]))}

    def capture_state(self):
        return self.history

    def restore_state(self, state):
        self.history = state

    def accept(self, positions):
        self.applied = positions.copy()

    def reset(self, joints=None):
        self.history = 0


@contextmanager
def synthetic_profile(settings):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "robot.urdf"
        path.write_text("""<robot name="fixture">
<link name="pedestal"><collision><origin xyz="0 0 -2"/><geometry><sphere radius="0.1"/></geometry></collision></link>
<link name="tool"><collision><origin xyz="1 0 0"/><geometry><sphere radius="0.1"/></geometry></collision></link>
<joint name="axis" type="revolute"><parent link="pedestal"/><child link="tool"/>
<axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
        pairs = Path(directory) / "collisions.json"
        pairs.write_text(json.dumps([["pedestal", "tool"]]))
        group = ControlGroup(("axis",), "tool", "wand", "squeeze")
        spec = RobotSpec(
            str(path),
            ("axis",),
            {"reach": "tool"},
            {"reach": ("axis",)},
            {"axis": (-2.0, 2.0)},
            {"axis": 0.0},
            str(pairs),
            1 / settings.control_frequency,
        )
        yield RobotProfile(spec, ("axis",), {"reach": group}, "pedestal", {"axis": 1.0})


class ProfileBridgeTests(unittest.TestCase):
    def setUp(self):
        self.solver = HistorySolver()

        def factory(name, spec):
            self.assertEqual(name, "fixture")
            return TransactionalBackend(spec, PlacoModel(spec), self.solver)

        with patch("xrobotoolkit_teleop.integrations.retargeter.create_backend", side_effect=factory):
            self.bridge = RoboJuDoBridge(
                profile=synthetic_profile(DEFAULTS), ik_backend="fixture", context=MemoryContext()
            )
        self.addCleanup(self.bridge.close)
        self.state = dict(
            schema_version=1,
            type="upper_body_state",
            session_id="fixture",
            generation=1,
            state_id=0,
            command_id=0,
            intent="TRACK",
            valid=True,
            positions={"axis": 0.0},
            joint_age_s={"axis": 0.0},
            last_commanded_positions={"axis": 0.0},
            default_positions={"axis": 0.0},
            position_limits={"axis": [-2.0, 2.0]},
            control_dt=0.02,
            arm_velocity_limit=1.0,
            return_velocity_limit=0.5,
            feedback_timeout_s=0.1,
        )
        self.frame = XRFrame(0, {"headset": [0, 0, 0, 0, 0, 0, 1], "wand": [0, 0, 0, 0, 0, 0, 1]}, {"reach": 0.0})

    def poll(self, ack=False):
        if ack:
            self.state["command_id"] += 1
            self.state["applied_bridge_id"] = self.bridge.bridge_id
            self.state["applied_frame_id"] = self.bridge.commands.frames[-1]["frame_id"]
            self.state["last_commanded_positions"] = self.bridge.commands.frames[-1]["positions"].copy()
        self.state["state_id"] += 1
        self.frame.timestamp_ns += 1
        self.bridge.feedback.messages.append(copy.deepcopy(self.state))
        with patch.object(self.bridge, "_xr_frame", return_value=self.frame):
            return self.bridge.poll()

    def engage_and_propose(self):
        self.assertFalse(self.poll())
        self.assertTrue(self.poll())  # released idle command in streaming mode
        self.frame.grips["reach"] = 1.0
        self.assertTrue(self.poll())  # clutch anchor has no solver history
        self.assertTrue(self.poll(ack=True))
        self.assertIsNotNone(next(reversed(self.bridge._outstanding.values()))[1][0])
        self.assertEqual(self.solver.history, 0)

    def test_different_joint_count_mapping_and_pending_history(self):
        self.engage_and_propose()
        self.assertEqual(set(self.bridge.commands.frames[-1]["positions"]), {"axis"})
        self.assertEqual(set(self.bridge.retargeter.target_poses), {"reach"})
        self.assertTrue(self.poll(ack=True))
        self.assertEqual(self.solver.history, 1)
        self.assertEqual(self.solver.applied, self.state["last_commanded_positions"])
        self.assertTrue(self.poll())  # unchanged baseline never blocks publishing
        self.assertEqual(self.solver.history, 1)

    def test_accept_failure_resets_history_without_stopping_publication(self):
        self.engage_and_propose()
        with patch.object(self.bridge.retargeter.backend, "accept", side_effect=RuntimeError("accept failed")):
            self.assertTrue(self.poll(ack=True))
        self.assertEqual(self.bridge.status, "tracking")
        self.assertIsNone(self.bridge.retargeter._pending)
        self.assertEqual(self.solver.history, 0)

    def test_backtracked_quantized_positions_are_committed_exactly(self):
        self.engage_and_propose()
        validate = self.bridge.retargeter._validate_motion
        attempts = []

        def reject_full(measured, previous, candidate):
            attempts.append(candidate.copy())
            if len(attempts) == 1:
                raise RuntimeError("full step rejected")
            validate(measured, previous, candidate)

        with patch.object(self.bridge.retargeter, "_validate_motion", side_effect=reject_full):
            self.assertTrue(self.poll(ack=True))
        positions = self.bridge.commands.frames[-1]["positions"]
        self.assertEqual(len(attempts), 2)
        self.assertLess(positions["axis"] - self.state["last_commanded_positions"]["axis"], 0.001)
        self.assertEqual(float(np.float32(positions["axis"])), positions["axis"])
        self.assertTrue(self.poll(ack=True))
        self.assertEqual(self.solver.applied, positions)

    def test_expired_completion_history_is_discarded_without_stopping(self):
        self.engage_and_propose()
        self.state["feedback_timeout_s"] = 1.0
        with patch("time.monotonic", return_value=next(reversed(self.bridge._outstanding.values()))[2] + 0.11):
            self.assertTrue(self.poll(ack=True))
        self.assertEqual(self.solver.history, 0)
        self.assertIsNone(self.solver.applied)

    def test_retargeter_rejecting_completion_does_not_stop_publishing(self):
        self.engage_and_propose()
        with patch.object(self.bridge.retargeter, "complete_proposal", return_value=False):
            self.assertTrue(self.poll(ack=True))
        self.assertEqual(self.bridge.status, "tracking")
        self.assertIsNone(self.bridge.retargeter._pending)

    def test_cleanup_runs_even_if_pending_rejection_fails(self):
        self.engage_and_propose()
        retargeter = self.bridge.retargeter
        path = Path(retargeter.profile.spec.urdf_path)
        with patch.object(retargeter.backend, "reject", side_effect=RuntimeError("reject failed")):
            with self.assertRaisesRegex(RuntimeError, "reject failed"):
                self.bridge.close()
        self.assertFalse(path.exists())
        self.assertIsNone(retargeter._pending)
        retargeter.close()

    def test_acknowledgement_requires_an_open_pending_proposal(self):
        self.assertFalse(self.bridge.retargeter.acknowledge({"axis": 0.0}))
        self.bridge.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.bridge.retargeter.acknowledge({"axis": 0.0})

    def test_mismatched_acknowledgement_rejects_history(self):
        self.engage_and_propose()
        self.state["command_id"] += 1
        self.assertTrue(self.poll())  # new clutch anchor, not a commit
        self.assertEqual(self.solver.history, 0)
        self.assertIsNone(self.solver.applied)

    def test_generation_change_and_invalid_feedback_drop_pending(self):
        self.engage_and_propose()
        self.state["generation"] += 1
        self.assertFalse(self.poll())
        self.assertIsNone(self.bridge.retargeter._pending)
        self.assertEqual(self.solver.history, 0)
        self.state["positions"] = {"wrong": 0.0}
        self.assertFalse(self.poll())
        self.assertIsNone(self.solver.applied)

    def test_send_failure_rolls_back_solver_and_holds(self):
        self.engage_and_propose()
        with patch.object(self.bridge.commands, "send_json", side_effect=RuntimeError("send failed")):
            self.assertFalse(self.poll(ack=True))
        self.assertEqual(self.bridge.status, "hold")
        self.assertIsNone(self.bridge.retargeter._pending)
        self.assertEqual(self.solver.history, 0)

    def test_deadline_expiry_rejects_pending_and_reanchors(self):
        self.engage_and_propose()
        with patch("time.monotonic", return_value=next(reversed(self.bridge._outstanding.values()))[2] + 0.11):
            self.assertTrue(self.poll())
        self.assertEqual(self.solver.history, 0)
        self.assertIsNone(self.solver.applied)

    def test_overrun_does_not_commit_or_publish(self):
        self.engage_and_propose()
        original = self.bridge.retargeter.step
        with patch("time.monotonic", return_value=100.0) as clock:

            def slow(*args, **kwargs):
                result = original(*args, **kwargs)
                clock.return_value = 100.2
                return result

            count = len(self.bridge.commands.frames)
            with patch.object(self.bridge.retargeter, "step", side_effect=slow):
                self.assertFalse(self.poll(ack=True))
        self.assertEqual(len(self.bridge.commands.frames), count)
        self.assertIsNone(self.bridge.retargeter._pending)
        self.assertEqual(self.solver.history, 0)

    def test_backend_failure_cleans_profile_before_any_socket_or_xr_use(self):
        paths = []

        @contextmanager
        def tracked_profile():
            with synthetic_profile(DEFAULTS) as profile:
                paths.append(Path(profile.spec.urdf_path))
                yield profile

        from unittest.mock import Mock

        context = Mock()
        xr = Mock()
        with patch(
            "xrobotoolkit_teleop.integrations.retargeter.create_backend", side_effect=ImportError("native bindings")
        ):
            with self.assertRaisesRegex(ImportError, "native bindings"):
                RoboJuDoBridge(profile=tracked_profile(), context=context, xr_client=xr, ik_backend="lerobot")
        context.socket.assert_not_called()
        xr.get_timestamp_ns.assert_not_called()
        xr.close.assert_called_once()
        self.assertFalse(paths[0].exists())

    def test_runtime_defaults_do_not_mutate_reusable_profile(self):
        with synthetic_profile(DEFAULTS) as profile:
            bridge = RoboJuDoBridge(profile=nullcontext(profile), context=MemoryContext())
            try:
                result = bridge.retargeter.step(
                    {"axis": 0.0}, {"axis": 0.0}, {"axis": 0.25}, {"axis": (-2.0, 2.0)}, intent="HOME"
                )
                self.assertIsNotNone(result)
                self.assertEqual(profile.spec.initial_positions, {"axis": 0.0})
                self.assertEqual(bridge.retargeter.backend.spec.initial_positions, {"axis": 0.25})
            finally:
                bridge.close()

    def test_profile_resources_close_after_backend(self):
        path = Path(self.bridge.retargeter.profile.spec.urdf_path)
        backend = self.bridge.retargeter.backend
        self.assertTrue(path.exists())
        self.bridge.close()
        self.bridge.close()
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.reset()

    def test_native_placo_supports_low_dof_profile(self):
        bridge = RoboJuDoBridge(profile=synthetic_profile(DEFAULTS), context=MemoryContext())
        self.addCleanup(bridge.close)
        retargeter = bridge.retargeter
        frame = XRFrame(0, {"headset": [0, 0, 0, 0, 0, 0, 1], "wand": [0, 0, 0, 0, 0, 0, 1]}, {"reach": 0.0})
        for timestamp, grip in [(1, 0.0), (2, 0.0), (3, 1.0), (4, 1.0)]:
            frame.timestamp_ns = timestamp
            frame.grips["reach"] = grip
            if timestamp == 4:
                frame.poses["wand"] = [0, 0, 0, 0, np.sin(0.025), 0, np.cos(0.025)]
            proposal = retargeter.step(
                {"axis": 0.0}, {"axis": 0.0}, {"axis": 0.0}, {"axis": (-2.0, 2.0)}, intent="TRACK", frame=frame
            )
        self.assertIsNotNone(proposal, retargeter.hold_reason)
        self.assertGreater(abs(proposal["axis"]), 0.001)
        self.assertLessEqual(abs(proposal["axis"]), 0.02)
        self.assertTrue(retargeter.acknowledge(proposal))

    def test_profile_registry_loads_without_core_changes(self):
        with patch.dict(PROFILES, {"fixture": "test_robojudo_profiles:synthetic_profile"}):
            with patch(
                "xrobotoolkit_teleop.integrations.retargeter.create_backend",
                side_effect=lambda name, spec: TransactionalBackend(spec, PlacoModel(spec), HistorySolver()),
            ):
                bridge = RoboJuDoBridge(robot="fixture", context=MemoryContext())
                self.addCleanup(bridge.close)
        self.assertEqual(bridge.retargeter.joint_names, ["axis"])

    def test_invalid_profile_rejected_before_backend_construction(self):
        with synthetic_profile(DEFAULTS) as profile:
            with self.assertRaisesRegex(ValueError, "disjoint"):
                replace(profile, groups={"a": profile.groups["reach"], "b": profile.groups["reach"]}).validate()
        with patch("xrobotoolkit_teleop.integrations.retargeter.create_backend") as backend:
            with self.assertRaisesRegex(ValueError, "Unknown robot"):
                RoboJuDoBridge(robot="missing", context=MemoryContext())
            backend.assert_not_called()


@unittest.skipUnless(os.environ.get("X2_TEST_LEROBOT") == "1", "requires optional Pinocchio/CasADi build")
class NativeLeRobotDeploymentTests(unittest.TestCase):
    def test_full_body_symbolic_fk_matches_observed_model(self):
        from test_robojudo_x2 import fixture
        from xrobotoolkit_teleop.common.x2_arm_retargeter import X2ArmRetargeter

        retargeter = X2ArmRetargeter(ik_backend="lerobot")
        self.addCleanup(retargeter.close)
        measured, home = fixture(retargeter)
        measured.update(waist_yaw_joint=0.3, waist_pitch_joint=0.1)
        retargeter.step(measured, home, home, retargeter.limits, intent="HOME")
        solver = retargeter.backend.solver
        body = solver.pin.neutral(solver.full_model)
        for name, value in measured.items():
            body[solver.full_model.idx_qs[solver.full_model.getJointId(name)]] = value
        targets = [retargeter.robot.get_T_world_frame(g.frame) for g in retargeter.profile.groups.values()]
        ep, er = solver.errors(solver._vector(home), body, *targets)
        np.testing.assert_allclose(np.asarray(ep), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(er), 0.0, atol=1e-10)
        for key, expected in zip(retargeter.profile.groups, targets, strict=True):
            np.testing.assert_allclose(solver.forward_kinematics(measured)[key], expected, atol=1e-10)
        neutral = solver.pin.neutral(solver.full_model)
        ep, _ = solver.errors(solver._vector(home), neutral, *targets)
        self.assertGreater(np.linalg.norm(np.asarray(ep)), 0.01)

    def test_native_backend_supports_non_x2_joint_count(self):
        with synthetic_profile(DEFAULTS) as profile:
            from xrobotoolkit_teleop.ik import create_backend

            backend = create_backend("lerobot", profile.spec)
            self.addCleanup(backend.close)
            self.assertEqual(backend.solver.model.nq, 1)
            target = np.eye(4)
            angle = 0.05
            target[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            empty = LinearConstraint(np.zeros((0, 1)), np.zeros(0))
            problem = IKProblem(
                {"axis": 0.0},
                {"axis": 0.0},
                {"reach": target},
                frozenset({"axis"}),
                np.array([-0.02]),
                np.array([0.02]),
                empty,
                empty,
                0.02,
            )
            proposal = backend.propose(problem)
            self.assertGreater(proposal.positions["axis"], 0.001)
            self.assertLessEqual(proposal.positions["axis"], 0.02)
            self.assertIsNone(backend.solver._previous)
            backend.accept(proposal, proposal.positions)
            self.assertAlmostEqual(backend.solver._previous[0], proposal.positions["axis"])


if __name__ == "__main__":
    unittest.main()
