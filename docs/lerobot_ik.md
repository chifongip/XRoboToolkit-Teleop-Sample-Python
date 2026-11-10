# LeRobot IK for the X2 MuJoCo demo

Select the backend when starting the demo:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --ik-backend lerobot
python scripts/simulation/teleop_agibot_x2_mujoco.py --ik-backend lerobot --with-omnihand
```

`placo` remains the default. RoboJuDo's external-state bridge continues to use
its existing solver; it does not accept this option.

## Behavior

`lerobot` adapts the method in Hugging Face LeRobot's
[`G1_29_ArmIK`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/unitree_g1/g1_kinematics.py).
It uses X2 URDF geometry, 14 named arm joints, and the existing hand-center
frames. Other joints are locked in the optimization. Fingers retain their
separate trigger controls and speed limits.

Pinocchio supplies symbolic forward kinematics and rotation-log errors.
CasADi/IPOPT minimizes the same G1 objective:

```text
50 * squared_position_error
+ squared_rotation_error
+ 0.02 * squared_joint_angles
+ 0.1 * squared_change_from_previous_filtered_solution
```

The solve uses 50 maximum iterations and tolerance `1e-6`. Its four-sample
filter preserves LeRobot's chronological weights `[0.4, 0.3, 0.2, 0.1]`, with
the oldest sample receiving the largest weight. Before four distinct samples
arrive, the latest solution is used directly. Duplicate raw solutions do not
advance the window. A uniformly changing signal has two samples of filter
delay (40 ms at 50 Hz), in addition to solver and servo delays.

Both backends enforce intersected URDF/actuator position bounds and arm
velocity limits. Velocity bounds use the **previous accepted servo command**,
not the measured joint position; servo lag cannot enlarge the command step.
Inactive arms hold their servo commands exactly. Fingers retain independent
trigger control and bounded speeds.

Both backends use the same collision policy: cached exact distance witnesses,
linearized clearance constraints, nonworsening penetration recovery, and sampled
checks of motion from both measured joints and previous servo targets to the
outgoing commands. These checks include held finger targets and passive finger
couplings. `--arm-velocity-limit`, `--collision-margin`, and
`--collision-trigger` apply to both methods. The checks use the fitted collision
primitives and sampled paths; they are not continuous collision detection.

Placo queries its native geometry and solves a QP. LeRobot queries
Pinocchio/Coal directly and includes the shared constraints in CasADi/IPOPT;
it never invokes Placo for collision checking or command projection. Its
constraint capacity grows geometrically when more nearby pairs appear, retaining
every supplied row. Growth rebuilds the optimizer graph and can add a cold-cycle
cost. LeRobot retains its G1 tracking objective and filter; Placo retains its
manipulability and posture objectives.

Clutch engagement rebases the newly active arm from measured joints. Release
holds it exactly, and tracking loss resets history. A proposal only commits
filter history after the final arm/finger command check accepts it. Rejected
proposals retain the last accepted history; a collision rejection resets the
filter and retries once. Failure holds the previous commands.

Holding Y first tries a bounded joint step toward home. If obstructed, the
selected backend optimizes a joint-target objective under the same constraints
and command checks. Return speed is bounded by `--return-velocity-limit`, the
overall arm cap, and URDF velocity limits. Homing resets filter history. Outputs
drive the existing position actuators; G1 gravity feedforward torques are not
applied.

## Backend modules and visualization

The X2 controller uses `xrobotoolkit_teleop/ik/registry.py` to lazily load the
selected method. The shared request/proposal types and numeric `RobotAdapter`
protocol live in `ik/types.py`; native objects stay in model adapters and solver
engines. Both methods use the same composition:

| Layer | Placo | LeRobot |
| --- | --- | --- |
| Backend assembly | `placo_backend.py` | `lerobot_backend.py` |
| Observed model adapter | `placo_model.py` | `pinocchio_model.py` |
| Solver engine | `placo_solver.py` | `lerobot_solver.py` |
| Transaction lifecycle | `backend.py:TransactionalBackend` | `backend.py:TransactionalBackend` |

`IKSolver` defines solving, state snapshots, acceptance, reset, and cleanup for
both engines. The LeRobot engine keeps its G1 objective/filter helper internal
to the solver module; its backend does not inherit the numerical solver.
The shared backend validates solver output, restores the observed model and
accepted solver history after every proposal or failure, and commits the exact
applied commands only on acceptance. Closing either backend is idempotent and
prevents further operations. Native solver details are available through the
backend's solver engine for diagnostics; the controller uses the common API.

Runtime imports are independent. The repository still installs Placo as a base
dependency for its other robot demos; this change does not reorganize package extras.
`ik/safety.py` implements the common collision policy. X2 joint names, frames,
actuator limits, and hand couplings are configured by the controller. Other
robot demos and RoboJuDo keep their existing implementations.

To add a method:

1. Implement an `IKSolver` engine and numeric `RobotAdapter`, then assemble
   them with `TransactionalBackend`. Stateless engines inherit the default
   history hooks; stateful engines implement snapshot, restore, accept, and reset.
   This supplies the `IKBackend` protocol: `propose`, `accept`, `reject`,
   `reset(joints=None)`, and `close` with the same lifecycle as existing methods.
2. Register a lazy `module:factory` in `BACKENDS` with supported capabilities.
   CLI choices are generated from the registry when the module is imported.
3. Honor absolute command bounds, hard/soft linear constraints, pose targets,
   and joint-target homing. Keep proposal history pending until `accept`.
4. Add adapter parity, command-bound, rejection, collision, and import-isolation
   tests. A dummy third method already tests selection without controller edits.

`--visualize-ik` opens a shared Meshcat browser view of the accepted model pose,
end-effector frames, targets, and status for either backend. MuJoCo remains the
physics viewer. No extra browser server starts in headless operation.

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --ik-backend lerobot --visualize-ik
```

## Offline pose-cycle test

Replay default → T pose → default using mock XR input, with either backend:

```bash
# Headless: JSON report, exit status 0 on success and 1 on a failed check.
python scripts/misc/test_x2_ik_offline.py --ik-backend placo
python scripts/misc/test_x2_ik_offline.py --ik-backend lerobot --with-omnihand

# Watch the real simulated motion in MuJoCo; optionally add the browser IK view.
python scripts/misc/test_x2_ik_offline.py --ik-backend lerobot --visualize
python scripts/misc/test_x2_ik_offline.py --ik-backend placo --with-omnihand --visualize --visualize-ik
```

No headset or XR PC Service is required. The harness replaces the XR client
before construction, so it never initializes the native XR SDK. A separate
MuJoCo data instance generates reference hand poses from joint configurations.
Those poses become native XR positions and XYZW orientations relative to the
initial clutch references. Both grips stay held, and frame timestamps advance
once per control cycle. The normal XR mapping, selected IK backend, collision
checks, command acceptance, servos, and MuJoCo dynamics remain active.

The T reference sets shoulder pitch to zero, shoulder roll to ±1.55 rad, and
elbows to zero, exercising near-horizontal arms and a straight-elbow endpoint.
Other joints retain their default reference values; Omnihand fingers remain open.
Each transition uses a cosine ramp. Defaults are two complete cycles, two
simulated seconds per transition, and one second of settling at each endpoint.
Adjust these with `--cycles`, `--move-seconds`, and `--hold-seconds`. Shortening
the motion or settling time can cause tracking checks to fail.

Checks require finite states and commands, no rejected IK cycles, bounded arm
command speeds and positions, accurate XR target mapping, and measured endpoint
clearance above the collision margin. Endpoint hand errors must remain within
20 mm and 0.10 rad. Repeated visits must agree within 5 mm and 0.02 rad. The
report includes joint variation separately: redundant arm joint configurations
can vary while the hand pose remains accurate, especially near the straight-arm
T pose. Identical joint configurations are not enforced by the hand-pose task.

`--visualize` opens the MuJoCo window with target markers. `--visualize-ik`
opens the shared browser viewer with accepted commands, end-effector frames,
targets, and status. Either flag paces the replay at the control period when
possible; solver overruns make playback slower. Closing the MuJoCo window or
pressing Ctrl-C stops the replay and releases its resources.
An interrupted replay exits with status 130, so an incomplete run cannot be
mistaken for a passing check. Runtime failures produce a JSON error and exit
with status 1. Browser startup messages go to stderr, keeping the final JSON
report on stdout.

Run the automated regression, which uses fake windows for visualization checks:

```bash
X2_TEST_LEROBOT=1 python -m unittest discover -s tests -p test_x2_offline_replay.py -v
```

It covers both scenes and both installed backends, verifies that no real XR
client starts, and confirms that a backend which only holds commands fails the
pose test. LeRobot cases are skipped unless `X2_TEST_LEROBOT=1` is set.

## Native environment

Do the dependency work in a copy of the working XR environment. This backend
requires `pinocchio.casadi`, which is absent from the installed pip `pin`
wheel; installing pip `casadi` alone does not add it.

The build helper targets Pinocchio **4.1.0**, EigenPy **3.13.0**, and Placo
**0.11.0**, with an ABI-compatible CasADi build. Use Conda CasADi **3.6.7**;
the pip CasADi 3.7.0 wheel uses a different C++ string ABI from this stack.
Do not mix pip and Conda CasADi libraries in the same environment.

Create a fresh environment with the tested dependency versions:

```bash
conda create -n xr-lerobot -c conda-forge python=3.10 pip numpy=2.2.6 casadi=3.6.7
conda activate xr-lerobot
python -m pip install -e ".[x2]" pin==4.1.0 placo==0.11.0 eigenpy==3.13.0 \
  cmake ninja cmeel-eigen==3.4.1 jrl-cmakemodules==2.1.0
curl -L --fail -o /tmp/pinocchio-4.1.0.tar.gz \
  https://github.com/stack-of-tasks/pinocchio/archive/refs/tags/v4.1.0.tar.gz
tar -xf /tmp/pinocchio-4.1.0.tar.gz -C /tmp
python scripts/misc/build_lerobot_bindings.py \
  --source /tmp/pinocchio-4.1.0 \
  --build /tmp/x2-lerobot-native \
  --output /tmp/x2-lerobot-python
export PYTHONPATH="/tmp/x2-lerobot-python${PYTHONPATH:+:$PYTHONPATH}"
```

Building the overlay alone does not register it with Python. The export above
applies to the current shell. To install the verified overlay permanently in
the active environment, run:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import sys
import sysconfig

source = Path('/tmp/x2-lerobot-python')
destination = Path(sys.prefix) / 'lerobot-python'
registration = Path(sysconfig.get_paths()['purelib']) / '00_lerobot_pinocchio.pth'
if destination.exists() or registration.exists():
    raise SystemExit('Installation already exists; inspect before replacing it')
shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
registration.write_text("import os, sys; sys.path.insert(0, os.path.join(sys.prefix, 'lerobot-python'))\n")
print('Installed:', destination)
PY
env -u PYTHONPATH python -c 'import placo; import pinocchio; import pinocchio.casadi'
```

The `.pth` registration makes new Python processes use the patched bindings,
including in new terminals after activating this environment. The native
libraries, CasADi installation, and overlay must belong to the same compatible
environment. The original wheel's binding files remain in `cmeel.prefix`.
If cloning an already configured environment, rebuild the bindings against the
clone and replace its copied registration: compiled library search paths can
still point to the original environment.

The helper builds a Python overlay against the installed numeric Pinocchio
libraries and leaves installed packages untouched. It also adds the required
`StdVec_Scalar` and `StdVec_Vec3s` aliases to Pinocchio's bindings so importing
Placo first can reuse Placo's registered vector classes. The temporary setup
requires the overlay directory. After permanent installation, the environment's
copy is sufficient and the temporary build/output directories are no longer
needed. The extension's runtime search paths refer to the dependency environment;
rebuild if that environment changes.
When using CasADi from a separate prefix, set `CASADIPATH` to that prefix's
`lib` directory so IPOPT and its dependencies load from the same installation.
The backend checks IPOPT availability before opening the XR client.

Check both import orders in separate processes before running the demo:

```bash
python -c 'import placo; import pinocchio; import pinocchio.casadi; import casadi, mujoco'
python -c 'import pinocchio; import pinocchio.casadi; import placo, casadi, mujoco'
python -m pip install scipy  # independent numerical reference used by the optional tests
X2_TEST_LEROBOT=1 python -m unittest discover -s tests -v
```

The optional tests compare IPOPT against independently computed numeric
least squares, check both X2 models and controller behavior, and verify the
original filtering. Without `X2_TEST_LEROBOT=1`, native backend tests are
skipped so the regular suite works without the optional dependencies.

Replay the same deterministic input for each backend without a headset:

```bash
python scripts/misc/benchmark_x2_ik.py --ik-backend placo
python scripts/misc/benchmark_x2_ik.py --ik-backend lerobot
python scripts/misc/benchmark_x2_ik.py --ik-backend lerobot --with-omnihand
```

The replay reports initial-cycle cost, median/p95/max cycle time, 20 ms
overruns, failed cycles, position error, and maximum arm command changes.
Cycle time includes physics and controller work. Native solver initialization
can exceed one 50 Hz period; use the measured results for the selected scene.

## Validation results

Validated on 2026-11-10 on an Intel Core i9-14900HX with Python 3.10.22,
NumPy 2.2.6, Placo 0.11.0, Pinocchio 4.1.0, and Conda CasADi 3.6.7.
Both native import orders and an IPOPT smoke solve passed. The full suite passed
**162 tests in 363.1 seconds** with `X2_TEST_LEROBOT=1` after aligning the backend structures.
The final scalar-command and snapshot-isolation cleanup also passed all
36 backend contract tests and six offline replay tests. The optional suite
covers independent backend imports with the other solver blocked, collision
adapter distances/placements/Jacobians, command bounds despite servo lag,
penetration recovery, rejected proposals without filter-history commits,
obstructed homing with servo dynamics, shared renderer assets and cleanup,
URDF limit validation, large-quaternion normalization, and offline pose cycles
with interruption/nonfinite-state failures. Shared lifecycle tests also verify
proposal/failure rollback, exact applied-command acceptance, and closed-backend
behavior for both methods. Asset
regeneration checks compare XML structure and identifiers exactly, with an
absolute `1e-12` tolerance for computed geometry and inertia values.

The following replay used 200 control cycles at a simulated 50 Hz, with the
same position/orientation trajectory for each backend. Timing includes physics
and command processing; setup and the first cycle are excluded from these
percentiles. [Full measurements](lerobot_ik_benchmark.json) include cold-cycle
cost, orientation error, and maximum commanded joint steps.

| Backend | Model | Median cycle | p95 cycle | Max cycle | Mean position error | Cycles >20 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| placo | Arms | 5.42 ms | 5.64 ms | 16.28 ms | 3.66 mm | 0/200 |
| placo | Omnihand | 13.97 ms | 24.28 ms | 25.41 ms | 5.85 mm | 19/200 |
| lerobot | Arms | 13.31 ms | 16.25 ms | 26.63 ms | 7.99 mm | 2/200 |
| lerobot | Omnihand | 26.11 ms | 35.83 ms | 39.91 ms | 9.83 mm | 200/200 |

All four replays completed with zero failed cycles. The LeRobot Omnihand replay
exceeded the 20 ms period on every cycle; selecting 50 Hz does not guarantee
that rate. Placo Omnihand and LeRobot arms also had occasional overruns.
These timings include the newly enabled shared collision checks and velocity
bounds. They replace the earlier measurements that omitted those protections
for LeRobot. First-cycle costs and optimizer graph growth can add further delay.
LeRobot had larger position error in this replay. The comparison includes
optimization, filtering, collision checks, and simulated servo tracking.
These measurements cover this synthetic trajectory and the bundled scenes;
they do not establish timing or tracking performance for every target.
