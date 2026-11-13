"""LeRobot G1's nonlinear IK objective and filter for named scalar joints.

Algorithm reference: Hugging Face LeRobot's G1_29_ArmIK (Apache-2.0).
This module intentionally imports the optional native dependencies only on use.
"""

from collections import deque

import numpy as np

from .backend import IKSolver


def load_dependencies():
    try:
        import casadi
        import pinocchio
        from pinocchio import casadi as cpin

        if not casadi.has_nlpsol("ipopt"):
            raise ImportError("CasADi cannot load its IPOPT plugin")
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise ImportError(
            "The lerobot backend requires compatible Pinocchio CasADi bindings and CasADi/IPOPT. "
            "See docs/lerobot_ik.md for environment setup; Placo remains available with --ik-backend placo."
        ) from exc
    return pinocchio, cpin, casadi


class G1IKSolver:
    """Reference G1 objective and filter with position bounds.

    Production LeRobotIKSolver adds velocity and collision constraints to this
    mathematical helper. Its backend owns transactions with the controller.

    solve() commits filter history only after a finite bounded solution. A failed
    solve raises RuntimeError so the controller can retain its servo commands.
    All public joint dictionaries use URDF joint names and radians.
    """

    FILTER_WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])  # oldest to newest, as in LeRobot

    def __init__(self, urdf_path, joint_names, position_limits, frames=None, groups=None):
        self.pin, cpin, ca = load_dependencies()
        self.joint_names = tuple(joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("LeRobot IK requires distinct controlled joints")
        full = self.pin.buildModelFromUrdf(str(urdf_path))
        self.full_model = full
        self.body_configuration = self.pin.neutral(full)
        self._full_indices = {name: full.idx_qs[full.getJointId(name)] for name in full.names[1:]}
        self._full_data = full.createData()
        missing = set(self.joint_names) - set(full.names)
        if missing:
            raise ValueError(f"IK model is missing arm joints: {sorted(missing)}")
        locked = [j for j in range(1, full.njoints) if full.names[j] not in self.joint_names]
        self.model = self.pin.buildReducedModel(full, locked, self.pin.neutral(full))
        if self.model.nq != len(self.joint_names) or self.model.nv != len(self.joint_names):
            raise ValueError("LeRobot IK requires scalar controlled joints")
        self.indices = np.array([self.model.idx_qs[self.model.getJointId(n)] for n in self.joint_names])
        self.arm_indices = {
            side: np.array([i for n, i in zip(self.joint_names, self.indices, strict=True) if n in names])
            for side, names in (
                groups
                or {side: [n for n in self.joint_names if n.startswith(f"{side}_")] for side in ("left", "right")}
            ).items()
        }
        self._active_keys = {key: key if groups is not None else f"{key}_arm" for key in self.arm_indices}
        if any(not len(indices) for indices in self.arm_indices.values()):
            raise ValueError("LeRobot IK requires nonempty control groups")
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        for name, index in zip(self.joint_names, self.indices, strict=True):
            lower, upper = position_limits[name]
            self.lower[index] = max(lower, self.lower[index])
            self.upper[index] = min(upper, self.upper[index])
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Incompatible finite joint position limits")
        frame_names = frames or {side: f"{side}_hand_center" for side in ("left", "right")}
        self.frame_ids = {key: self.model.getFrameId(frame) for key, frame in frame_names.items()}
        self._full_frame_ids = {key: full.getFrameId(frame) for key, frame in frame_names.items()}
        if any(i >= self.model.nframes for i in self.frame_ids.values()):
            raise ValueError("IK model is missing a configured end-effector frame")
        # Keep observed, uncontrolled joints in symbolic FK. Reducing them at
        # neutral would give the wrong targets when a deployment torso moves.
        cm = cpin.Model(full)
        cd = cm.createData()
        cq = ca.SX.sym("q", self.model.nq)
        body = ca.SX.sym("body", full.nq)
        configuration = ca.SX(body)
        for name, index in zip(self.joint_names, self.indices, strict=True):
            configuration[full.idx_qs[full.getJointId(name)]] = cq[index]
        targets = {side: ca.SX.sym(f"target_{side}", 4, 4) for side in self.frame_ids}
        cpin.framesForwardKinematics(cm, cd, configuration)
        translation = ca.vertcat(
            *[cd.oMf[fid].translation - targets[key][:3, 3] for key, fid in self._full_frame_ids.items()]
        )
        rotation = ca.vertcat(
            *[cpin.log3(cd.oMf[fid].rotation @ targets[key][:3, :3].T) for key, fid in self._full_frame_ids.items()]
        )
        # Pinocchio's symbolic log3 contains inactive branches with singular
        # derivatives at axis-aligned rotations. Reverse AD can propagate 0*NaN
        # through them; forward AD preserves the same objective without that path.
        self.errors = ca.Function(
            "ik_errors", [cq, body, *targets.values()], [translation, rotation], {"enable_reverse": False}
        )
        self._build_optimizer(ca)
        self._history = deque(maxlen=4)
        self._previous = None
        self._active = set()

    def _build_optimizer(self, ca):
        """Create a solver graph without changing accepted filter state."""
        self.opti = ca.Opti()
        self.q = self.opti.variable(self.model.nq)
        self.reference = self.opti.parameter(self.model.nq)
        self.targets = {side: self.opti.parameter(4, 4) for side in self.frame_ids}
        self.bound_lower = self.opti.parameter(self.model.nq)
        self.bound_upper = self.opti.parameter(self.model.nq)
        self.body = self.opti.parameter(self.full_model.nq)
        ep, er = self.errors(self.q, self.body, *self.targets.values())
        self.tracking_cost = (
            50 * ca.sumsqr(ep) + ca.sumsqr(er) + 0.02 * ca.sumsqr(self.q) + 0.1 * ca.sumsqr(self.q - self.reference)
        )
        self.opti.minimize(self.tracking_cost)
        self.opti.subject_to(self.opti.bounded(self.bound_lower, self.q, self.bound_upper))
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 50, "tol": 1e-6},
                "print_time": False,
                "calc_lam_p": False,
            },
        )

    def _vector(self, positions):
        q = np.empty(self.model.nq)
        for name, index in zip(self.joint_names, self.indices, strict=True):
            q[index] = positions[name]
        if not np.isfinite(q).all():
            raise ValueError("IK joint positions must be finite")
        return q

    def reset(self, joints=None):
        if joints is not None:
            self._active.difference_update(
                key
                for key, indices in self.arm_indices.items()
                if any(n in joints for n, i in zip(self.joint_names, self.indices, strict=True) if i in indices)
            )
            return
        self._history.clear()
        self._previous = None
        self._active.clear()

    def close(self):
        self.reset()

    def _full_configuration(self, positions):
        configuration = self.body_configuration.copy()
        # Require all controlled joints; omitted observed joints retain neutral.
        for name in self.joint_names:
            configuration[self._full_indices[name]] = positions[name]
        for name, value in positions.items():
            if name in self._full_indices:
                configuration[self._full_indices[name]] = value
        if not np.isfinite(configuration).all():
            raise ValueError("Observed body configuration must be finite")
        return configuration

    def forward_kinematics(self, positions):
        configuration = self._full_configuration(positions)
        self.pin.framesForwardKinematics(self.full_model, self._full_data, configuration)
        return {key: self._full_data.oMf[fid].homogeneous.copy() for key, fid in self._full_frame_ids.items()}

    def solve(self, measured_positions, previous_positions, targets, active_arms):
        measured = self._vector(measured_positions)
        held = self._vector(previous_positions)
        active = {side for side in self.frame_ids if active_arms.get(self._active_keys[side], False)}
        # Releases take effect even if this solve fails. New engagements are
        # committed only on success, so a retry still rebases from measurements.
        self._active.intersection_update(active)
        reference = held.copy() if self._previous is None else self._previous.copy()
        history = [q.copy() for q in self._history]
        # Rebase only changed arms, preserving the other arm's filter continuity.
        for side, indices in self.arm_indices.items():
            if side not in active:
                reference[indices] = held[indices]
                for sample in history:
                    sample[indices] = held[indices]
            elif side not in self._active:
                reference[indices] = measured[indices]
                for sample in history:
                    sample[indices] = measured[indices]
        lower, upper = self.lower.copy(), self.upper.copy()
        for group, indices in self.arm_indices.items():
            if group not in active:
                if np.any(held[indices] < lower[indices]) or np.any(held[indices] > upper[indices]):
                    raise RuntimeError(f"Held joints outside IK limits: {group}")
                lower[indices] = upper[indices] = held[indices]
        validated_targets = {}
        for side in self.targets:
            target = np.asarray(targets[side], dtype=float)
            if target.shape != (4, 4) or not np.isfinite(target).all():
                raise ValueError(f"Invalid {side} IK target")
            validated_targets[side] = target
        try:
            solution = np.asarray(
                self._run_optimizer(
                    reference, lower, upper, self._full_configuration(measured_positions), validated_targets
                )
            ).reshape(self.model.nq)
        except RuntimeError as exc:
            raise RuntimeError("LeRobot IK did not converge; holding commands") from exc
        if not np.isfinite(solution).all() or np.any(solution < lower - 1e-6) or np.any(solution > upper + 1e-6):
            raise RuntimeError("LeRobot IK returned an invalid joint configuration")
        solution = np.clip(solution, lower, upper)
        if not history or not np.array_equal(solution, history[-1]):
            history = (history + [solution])[-4:]
        filtered = history[-1].copy() if len(history) < 4 else np.asarray(history).T @ self.FILTER_WEIGHTS
        filtered = np.clip(filtered, lower, upper)
        self._history = deque(history, maxlen=4)
        self._previous = filtered.copy()
        self._active = active
        return {n: float(filtered[i]) for n, i in zip(self.joint_names, self.indices, strict=True)}

    def _run_optimizer(self, reference, lower, upper, body, targets):
        """Reference Opti path; production uses a converted function."""
        for side, target in targets.items():
            self.opti.set_value(self.targets[side], target)
        self.opti.set_value(self.body, body)
        self.opti.set_initial(self.q, np.clip(reference, lower, upper))
        self.opti.set_value(self.reference, reference)
        self.opti.set_value(self.bound_lower, lower)
        self.opti.set_value(self.bound_upper, upper)
        return self.opti.solve().value(self.q)


class LeRobotIKSolver(G1IKSolver, IKSolver):
    """Transactional warm starts and cached graphs for collision row layouts."""

    # Private switches support controlled benchmark comparisons, not runtime UI.
    _warm_starts = True
    _compact_slack = True
    _function_calls = True

    _INITIAL_CAPACITY = 16
    _SLACK_LIMIT = 1000.0
    _APPLIED_WARM_TOLERANCE = 1e-7

    def __init__(self, spec):
        self.spec = spec
        super().__init__(spec.urdf_path, spec.joints, spec.limits, spec.frames, spec.groups)
        self._warm = None
        self._backend_owns_rollback = False
        self._variants = {}
        self._select_optimizer(0, 0)

    def _select_optimizer(self, hard_rows, soft_rows):
        if not self._compact_slack:
            soft_rows = max(self._INITIAL_CAPACITY, hard_rows, soft_rows)
        kind = (bool(hard_rows), bool(soft_rows))
        cached = self._variants.get(kind)
        hard_capacity = max(self._INITIAL_CAPACITY, 2 ** (hard_rows - 1).bit_length()) if hard_rows else 0
        soft_capacity = 2 ** (soft_rows - 1).bit_length() if soft_rows else 0
        # Keep one high-water capacity per variant; switches never rebuild a graph.
        if cached is not None:
            hard_capacity = max(hard_capacity, cached["_rows"])
            soft_capacity = max(soft_capacity, cached["_soft_rows"])
            if (hard_capacity, soft_capacity) == (cached["_rows"], cached["_soft_rows"]):
                self._activate_optimizer(cached)
                return
        previous = getattr(self, "_graph", None)
        try:
            graph = self._build_constraint_optimizer(hard_capacity, soft_capacity)
        except Exception:
            # Graph compilation is not part of a proposal's accepted state.
            if previous is not None:
                self._activate_optimizer(previous)
            raise
        self._variants[kind] = graph
        self._activate_optimizer(graph)

    def _activate_optimizer(self, graph):
        for name, value in graph.items():
            setattr(self, name, value)
        self._graph = graph

    @staticmethod
    def _solver_options(warm=False):
        return {
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 50,
                "tol": 1e-6,
                "warm_start_init_point": "yes" if warm else "no",
            },
            "print_time": False,
            "calc_lam_p": False,
            "error_on_fail": True,
        }

    def _build_constraint_optimizer(self, hard_capacity, soft_capacity):
        _, _, ca = load_dependencies()
        self._build_optimizer(ca)
        self._rows, self._soft_rows = hard_capacity, soft_capacity
        self._layout = (hard_capacity, soft_capacity)
        self.mode = self.opti.parameter()
        self.home = self.opti.parameter(self.model.nq)
        self.hard_A = self.opti.parameter(hard_capacity, self.model.nq)
        self.hard_b = self.opti.parameter(hard_capacity)
        if hard_capacity:
            self.opti.subject_to(self.hard_A @ self.q <= self.hard_b)
        objective = ca.if_else(self.mode > 0.5, ca.sumsqr(self.q - self.home), self.tracking_cost, True)
        inputs = [
            self.reference,
            self.bound_lower,
            self.bound_upper,
            self.body,
            *self.targets.values(),
            self.mode,
            self.home,
            self.hard_A,
            self.hard_b,
        ]
        guesses = [self.q]
        outputs = [self.q]
        self.soft_A = self.soft_b = self.slack = self.slack_upper = None
        if soft_capacity:
            self.soft_A = self.opti.parameter(soft_capacity, self.model.nq)
            self.soft_b = self.opti.parameter(soft_capacity)
            self.slack = self.opti.variable(soft_capacity)
            self.slack_upper = self.opti.parameter(soft_capacity)
            self.opti.subject_to(self.soft_A @ self.q - self.soft_b <= self.slack)
            self.opti.subject_to(self.opti.bounded(0, self.slack, self.slack_upper))
            objective += 1e6 * ca.sumsqr(self.slack)
            inputs += [self.soft_A, self.soft_b, self.slack_upper]
            guesses += [self.slack]
            outputs += [self.slack]
        self.opti.minimize(objective)
        self._inputs = inputs
        self._guesses = guesses + [self.opti.lam_g]
        self._outputs = outputs + [self.opti.lam_g]
        # Differentiate the MX graph before expansion. Expanding the entire NLP
        # first bypasses errors' forward-AD policy and can produce NaN gradients
        # in Pinocchio log3's inactive branches during obstructed homing.
        x, p = self.opti.x, self.opti.p
        lam_f = ca.MX.sym("lam_f")
        lam_g = ca.MX.sym("lam_g", self.opti.ng)
        gradient = ca.gradient(self.opti.f, x)
        hessian = ca.hessian(lam_f * self.opti.f + ca.dot(lam_g, self.opti.g), x)[0]
        self._derivative_options = {
            "grad_f": ca.Function("nlp_grad_f", [x, p], [self.opti.f, gradient]).expand(),
            "hess_lag": ca.Function("nlp_hess_l", [x, p, lam_f, lam_g], [ca.triu(hessian)]).expand(),
        }
        # Use an independent Opti copy: to_function caches its solver, so changing
        # options on an already converted Opti can silently reuse the old solver.
        self._cold_opti = self.opti.copy()
        self._cold_opti.solver("ipopt", dict(self._solver_options(), **self._derivative_options))
        self._cold_optimizer = self._cold_opti.to_function("solve_ik_cold", self._inputs + self._guesses, self._outputs)
        self.opti.solver("ipopt", dict(self._solver_options(warm=self._warm_starts), **self._derivative_options))
        self._optimizer = self.opti.to_function("solve_ik_warm", self._inputs + self._guesses, self._outputs)
        names = (
            "opti",
            "q",
            "reference",
            "targets",
            "bound_lower",
            "bound_upper",
            "body",
            "tracking_cost",
            "mode",
            "home",
            "hard_A",
            "hard_b",
            "_rows",
            "_soft_rows",
            "_layout",
            "_optimizer",
            "_cold_optimizer",
            "_cold_opti",
            "soft_A",
            "soft_b",
            "slack",
            "slack_upper",
            "_inputs",
            "_guesses",
            "_outputs",
            "_derivative_options",
        )
        return {name: getattr(self, name) for name in names}

    def reset(self, joints=None):
        self._warm = None
        super().reset(joints)

    def capture_state(self):
        return (
            [q.copy() for q in self._history],
            None if self._previous is None else self._previous.copy(),
            self._active.copy(),
            self._copy_warm(self._warm),
        )

    def restore_state(self, state):
        history, previous, active, warm = state
        self._history = deque((q.copy() for q in history), maxlen=4)
        self._previous = None if previous is None else previous.copy()
        self._active = active.copy()
        self._warm = self._copy_warm(warm)

    @staticmethod
    def _copy_warm(warm):
        """Copy the known array layout without recursive generic dispatch."""
        if warm is None:
            return None
        return {
            key: tuple(a.copy() for a in value)
            if key in ("hard", "soft")
            else value.copy()
            if isinstance(value, np.ndarray)
            else value
            for key, value in warm.items()
        }

    def solve_in_transaction(self, problem):
        """Let the backend's proposal snapshot own rollback for this call."""
        previous = self._backend_owns_rollback
        self._backend_owns_rollback = True
        try:
            return self.solve(problem)
        finally:
            self._backend_owns_rollback = previous

    def _constraint_parameters(self, constraint, rows):
        if len(constraint.b) > rows:
            raise ValueError("Too many configured collision rows")
        A, b = np.zeros((rows, self.model.nq)), np.ones(rows)
        A[: len(constraint.b), self.indices] = constraint.A
        b[: len(constraint.b)] = constraint.b
        return A, b

    def _run_optimizer(self, reference, lower, upper, body, targets):
        problem = self._problem
        hard = self._constraint_parameters(problem.hard, self._rows)
        soft = self._constraint_parameters(problem.soft, self._soft_rows)
        warm = self._warm if self._warm_starts else None
        compatible = warm is not None and warm["layout"] == self._layout
        self.last_stats["warm_start_used"] = compatible
        slack_upper = np.zeros(self._soft_rows)
        slack_upper[: len(problem.soft.b)] = self._SLACK_LIMIT
        same_soft = compatible and all(np.array_equal(a, b) for a, b in zip(warm["soft"], soft))
        initial = np.clip(warm["q"] if compatible else reference, lower, upper)
        dual = warm["dual"].copy() if compatible else np.zeros(self.opti.ng)
        nq = self.model.nq
        # Opti orders multipliers as joint bounds, hard rows, soft rows, slack bounds.
        # Collision rows have no stable IDs: reuse a block only on exact equality.
        if compatible:
            unchanged = (warm["lower"] == lower) & (warm["upper"] == upper)
            dual[np.flatnonzero(~unchanged)] = 0
            for old, new, start, count in (
                (warm["hard"], hard, nq, self._rows),
                (warm["soft"], soft, nq + self._rows, self._soft_rows),
            ):
                if not all(np.array_equal(a, b) for a, b in zip(old, new)):
                    dual[start : start + count] = 0
            slack_bounds_start = nq + self._rows + self._soft_rows
            if not same_soft:
                dual[slack_bounds_start:] = 0
            else:
                # Padded A=0, b=1 rows can become supplied rows without changing
                # the matrix. Their slack upper bounds still change from 0 to 1000.
                changed = np.flatnonzero(warm["slack_upper"] != slack_upper)
                dual[slack_bounds_start + changed] = 0
        args = [
            reference,
            lower,
            upper,
            body,
            *targets.values(),
            float(problem.joint_target is not None),
            self._vector(problem.joint_target or problem.previous),
            *hard,
        ]
        if self._soft_rows:
            # A constant soft inequality can still require positive slack.
            slack = np.clip(soft[0] @ initial - soft[1], 0, slack_upper)
            if same_soft:
                slack = np.maximum(slack, np.clip(warm["slack"], 0, slack_upper))
            args += [*soft, slack_upper, initial, slack, dual]
        else:
            args += [initial, dual]
        if self._function_calls:
            # Resets and layout changes need IPOPT's ordinary initialization.
            optimizer = self._optimizer if compatible else self._cold_optimizer
            result = optimizer(*args)
            self.last_stats = optimizer.stats()
        else:
            opti = self.opti if compatible else self._cold_opti
            for symbol, value in zip(self._inputs, args):
                opti.set_value(symbol, value)
            for symbol, value in zip(self._guesses, args[len(self._inputs) :]):
                opti.set_initial(symbol, value)
            solution = opti.solve()
            result = [solution.value(symbol) for symbol in self._outputs]
            self.last_stats = opti.stats()
        self.last_stats["warm_start_used"] = compatible
        if not self.last_stats.get("success", False):
            raise RuntimeError("IPOPT did not succeed")
        arrays = [np.asarray(value).reshape(-1) for value in result]
        if not all(np.isfinite(value).all() for value in arrays):
            raise RuntimeError("Nonfinite optimizer output")
        self._warm = dict(
            layout=self._layout,
            q=arrays[0].copy(),
            dual=arrays[-1].copy(),
            slack=arrays[1].copy() if self._soft_rows else np.empty(0),
            lower=lower.copy(),
            upper=upper.copy(),
            hard=hard,
            soft=soft,
            slack_upper=slack_upper,
        )
        return arrays[0]

    def _validate_problem(self, problem):
        """Reject malformed numerical inputs before allocating or modifying a graph."""
        n = len(self.joint_names)
        for label, bounds in (("lower", problem.lower), ("upper", problem.upper)):
            values = np.asarray(bounds)
            if values.shape != (n,) or values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                raise ValueError(f"Invalid {label} joint command bounds")
        for label, constraint in (("hard", problem.hard), ("soft", problem.soft)):
            A, b = np.asarray(constraint.A), np.asarray(constraint.b)
            if b.ndim != 1 or A.shape != (b.size, n):
                raise ValueError(f"Invalid {label} collision constraint dimensions")
            if A.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
                raise ValueError(f"Non-real or nonnumeric {label} collision constraint")
            if not np.isfinite(A).all() or not np.isfinite(b).all():
                raise ValueError(f"Nonfinite {label} collision constraint")

    def solve(self, problem):
        self._validate_problem(problem)
        state = None if self._backend_owns_rollback else self.capture_state()
        self.last_stats = {"success": False}
        lower, upper = self.lower.copy(), self.upper.copy()
        self._problem = problem
        try:
            self.lower[self.indices] = np.maximum(self.lower[self.indices], problem.lower)
            self.upper[self.indices] = np.minimum(self.upper[self.indices], problem.upper)
            if np.any(self.lower > self.upper):
                raise RuntimeError("Incompatible joint command bounds")
            self._select_optimizer(len(problem.hard.b), len(problem.soft.b))
            active = {key: any(n in problem.active for n in names) for key, names in self.spec.groups.items()}
            if {key for key, value in active.items() if value} != self._active:
                self._warm = None
            if problem.joint_target is not None:
                self.reset()
            return super().solve(problem.measured, problem.previous, problem.targets, active)
        except Exception:
            if state is not None:
                self.restore_state(state)
            raise
        finally:
            self.lower, self.upper = lower, upper
            self._problem = None

    def accept(self, positions):
        applied = self._vector(positions)
        # Bridge commands round-trip through float32. Ignore that representation
        # noise (below the 1e-6 command validation tolerance), not shortened steps.
        if self._previous is None or not np.allclose(
            applied, self._previous, rtol=0, atol=self._APPLIED_WARM_TOLERANCE
        ):
            self._warm = None
        self._previous = applied
