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

Tracking uses intersected URDF/actuator **joint position bounds only**. It
does not apply Placo's arm velocity constraints, manipulability objective,
posture objective, collision avoidance, or final arm/finger collision gates.
`--arm-velocity-limit`, `--collision-margin`, and `--collision-trigger` do not
constrain LeRobot tracking. Positive argument validation still applies.
MuJoCo's physical joint limits and actuator dynamics remain active.

Clutch release holds the released arm's commands exactly. Engagement starts
from its measured joints and rebases its filter history. Tracking loss holds
both arms and resets history. Failed/nonfinite solves retain valid commands
and history; failed optimizer iterates never enter the filter. Y returns the
arms directly toward the home joint configuration at `--return-velocity-limit`,
bounded by URDF velocities, without collision avoidance. Returning home also
resets the IK history. Outputs drive the existing position actuators; G1's
gravity feedforward torques are not applied.

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
Both native import orders and an IPOPT smoke solve passed. After review, the
installed `xr-robotics` environment passed **123 tests**, including all optional
tests, in 127.7 seconds. Asset regeneration checks compare XML structure and
identifiers exactly, with an absolute `1e-12` tolerance for computed geometry
and inertia values to accommodate floating-point serialization differences.

The following initial replay used 200 control cycles at a simulated 50 Hz, with the
same position/orientation trajectory for each backend. Timing includes physics
and command processing; setup and the first cycle are excluded from these
percentiles. [Full measurements](lerobot_ik_benchmark.json) include cold-cycle
cost, orientation error, and maximum commanded joint steps.

| Backend | Model | Median cycle | p95 cycle | Max cycle | Mean position error |
| --- | --- | ---: | ---: | ---: | ---: |
| placo | Arms | 3.50 ms | 3.65 ms | 4.86 ms | 3.66 mm |
| placo | Omnihand | 7.11 ms | 9.32 ms | 17.72 ms | 5.85 mm |
| lerobot | Arms | 4.10 ms | 4.81 ms | 5.06 ms | 7.99 mm |
| lerobot | Omnihand | 4.79 ms | 5.22 ms | 5.65 ms | 9.83 mm |

All four replays completed with zero failed cycles and zero steady-state
20 ms overruns. LeRobot had larger position error in this replay. The comparison
includes differences in optimization, filtering, and simulated servo tracking.
These measurements cover this synthetic trajectory and the bundled scenes;
they do not establish timing or tracking performance for every target.
