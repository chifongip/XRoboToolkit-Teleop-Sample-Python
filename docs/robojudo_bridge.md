# External-state RoboJuDo bridge

Run the bridge in the XR environment and RoboJuDo's existing X2 pipeline in its runtime environment:

```bash
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2 --ik-backend placo
# With the optional native dependencies configured:
python -m xrobotoolkit_teleop.integrations.robojudo --robot x2 --ik-backend lerobot
```

The original `integrations.robojudo_x2` command and `RoboJuDoX2Bridge` constructor remain available.
Both methods use the shared `ik` registry, numerical model adapters, collision policy, and transactional backend.
Placo remains the default. Its upstream manipulability objective can change redundant joint positions compared
with the earlier bridge, even when hand poses agree. LeRobot retains its own objective and filter.

The bridge uses observed robot state rather than a second physics simulation. X2 observes 31 joints and controls
14 arm joints. Torso-relative targets follow measured waist/body motion. LeRobot's symbolic forward kinematics
includes observed noncontrolled joints instead of freezing them at neutral. Arm velocity limits use acknowledged
commands, while collision checks sample paths from both measured and previously commanded positions.
Targets are quantized to float32 before the final checks. HOME works without an XR client.

## Layers and ownership

- `integrations/robojudo.py`: bounded ZMQ polling, schema validation, session/generation handling, deadlines,
  command acknowledgement, XR sampling, diagnostics, and cleanup.
- `integrations/retargeter.py`: clutch references, target transforms, bounds, collision constraints and path checks,
  HOME, and pending IK proposals. It uses the backend API without accessing native solver objects.
- `integrations/robot_profiles`: typed robot descriptions and lazy profile factories. X2 owns its deployment URDF,
  measured/controlled joints, frames, group mappings, and model limits here.
- `ik`: registered solver implementations and adapters, also used by standalone simulation.

There is at most one outstanding command. Solver history remains uncommitted while waiting for feedback. The bridge
commits only when fresh feedback has the same session/generation, exactly the next command ID, and the exact outgoing
joint positions within the original acknowledgement budget. Later feedback cannot extend that budget. This acknowledges successful environment submission, not physical
actuator arrival. A mismatch, timeout, outage, reset, failed send, or computation overrun discards the proposal and
rebases the clutch/history. Reconnection and engagement follow the existing release requirements.
The schema stays at version 1 and robot-owned deadlines remain authoritative. Only one publisher may own the endpoint.

## Add a robot

1. Add a module exposing a `load_profile(settings)` context manager. Yield a `RobotProfile` containing a `RobotSpec`,
   the complete observed joint set, disjoint `ControlGroup` entries, reference frame, and per-joint velocity limits.
   Each group names its controlled joints, end-effector frame, XR controller device and grip key.
   Choose a reference frame carried by observed, uncontrolled joints so it stays fixed within each IK solve. Group keys are
   arbitrary; they also label target poses. Initial positions cover all observed joints. The bridge replaces the
   controlled posture reference with the validated defaults supplied by RoboJuDo before solving.
2. Supply a matching URDF and explicit collision-pair file using geometry supported by the shared model adapters.
   If generating files, keep them alive for the entire context. Backend cleanup happens before context exit.
   Current adapters support scalar fixed-base joints and expose the shared leading world-pose layout.
3. Add a lazy `module:factory` entry in `PROFILES`. No bridge dispatch changes are necessary. XR input assumes native
   Y-up coordinates and robot reference axes X-forward, Y-left, Z-up; adapt device poses before this interface for
   other coordinate conventions. The current clutch uses headset heading, one controller/grip per group, and a
   configured home button. Other input/control modalities need a separate mapping implementation.
4. Configure RoboJuDo's existing upper-body controller with the same controlled joints and ensure feedback supplies
   every observed joint. RoboJuDo's existing target viewer supports X2's `left`/`right` labels relative to
   `torso_link`; disable `visualize_targets` for other layouts until their viewer is adapted. This affects display,
   not joint-command validation. Keep the arm-only protocol: hands and base commands are outside this integration.
5. Test feedback-name validation, moving reference frames, clutch mapping, limits/collision checks, HOME, and
   acknowledgement rollback. `tests/test_robojudo_profiles.py` demonstrates a one-joint profile with non-X2 names,
   a custom device mapping, and a stateful test solver.

Select a compatible registered IK method independently of the robot profile. Methods must support pose tracking,
joint-target HOME and linear constraints. LeRobot now sizes optimization to the configured scalar controlled joints.
Placo omits its 6D manipulability task for groups with fewer than six controlled joints, avoiding a singular objective.
LeRobot uses forward differentiation of its pose-error function to avoid NaN gradients from inactive rotation-log
branches at axis-aligned poses; its objective and filter are unchanged.
Adding another method uses `ik/registry.py`; adding another robot uses `robot_profiles/PROFILES`.

## Dependencies and validation

From a RoboJuDo-Plus checkout, create the single teleop environment with:

```bash
python scripts/install_xrobotoolkit_teleop.py
conda activate robojudo-xr
```

Setup installs Placo and LeRobot together under Python 3.11, builds compatible Pinocchio symbolic bindings,
and registers them permanently. It keeps RoboJuDo in its existing environment. An existing target requires
`--reuse`; use `--name NAME` for a different name. Conda and `g++` are prerequisites. The source archive is
checksum-verified, and logs/manifest remain under the target prefix's `share/robojudo-xr` directory.
See the [RoboJuDo setup guide](../../../docs/xrobotoolkit_teleop.md) for the complete installation flow.

Bare `pip install -e ".[x2]"` still installs Placo dependencies only. For a standalone source installation outside
RoboJuDo, the [manual native setup guide](lerobot_ik.md) describes the build process.
Do not mix pip and Conda CasADi libraries or copy compiled overlays between Python versions/environments.
Selecting an unavailable backend raises an actionable error before XR initialization and never silently falls back.

```bash
# From the submodule root, with this checkout first on the import path:
PYTHONPATH=. python -m unittest discover -s tests -p 'test_robojudo*.py'
PYTHONPATH=. X2_TEST_LEROBOT=1 python -m unittest discover -s tests -p test_robojudo_profiles.py
PYTHONPATH=. BRIDGE_TEST_BACKEND=lerobot python -m unittest discover -s tests -p test_robojudo_x2.py
```

From RoboJuDo, set `XROBOTOOLKIT_TEST_PYTHON` to the XR interpreter and optionally
`XROBOTOOLKIT_TEST_BACKEND=lerobot` for its two-process integration test. It prepends the bundled source checkout,
so another editable installation cannot silently select a different bridge.

This integration builds on upstream revision `813bf48`. Use the submodule revision pinned by RoboJuDo-Plus
to obtain the matching bridge, robot profiles, and native setup support.

### Validation

Both solvers were validated together in a clean setup-created Python 3.11.18 environment on Linux x86_64.
Checks cover the backend contracts, profile extension, bounds, acknowledgement history and failures, moving-body FK,
offline pose replay with both hand scenes, and two-process MuJoCo/ONNX integration for both solvers. RoboJuDo feedback,
visualization and installer regressions and Ruff checks are included. No hardware SDK or live XR service is initialized.

Run the full native suite from this submodule's root with the optional dependencies configured:

```bash
PYTHONPATH=. X2_TEST_LEROBOT=1 python -m unittest discover -s tests
```
