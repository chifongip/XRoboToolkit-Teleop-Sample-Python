# RoboJuDo-Plus bridge

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

## Guide map

- [Setup and validation](#setup-and-validation)
- [OmniHand profile](#omnihand-profile)
- [Runtime behavior and diagnostics](#runtime-behavior)
- [Layers and ownership](#layers-and-ownership)
- [Add a robot](#add-a-robot)
- [Collision implementation and benchmarks](#shared-collision-checking)

## Setup and validation

From a RoboJuDo-Plus checkout, create the single teleop environment with:

```bash
python scripts/install_xrobotoolkit_teleop.py
conda activate robojudo-xr
```

Setup installs Placo and LeRobot together under Python 3.11, builds compatible Pinocchio symbolic bindings,
and registers them permanently. It keeps RoboJuDo in its existing environment. An existing target requires
`--reuse`; use `--name NAME` for a different name. Conda and `g++` are prerequisites. The source archive is
checksum-verified, and logs/manifest remain under the target prefix's `share/robojudo-xr` directory.
See the [RoboJuDo setup guide](https://github.com/chifongip/RoboJuDo-Plus/blob/main/docs/xrobotoolkit_teleop.md) for the complete installation flow.

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

## Runtime behavior

The `x2` extra also provides a headless bridge using live measured robot state:

```bash
python -m xrobotoolkit_teleop.integrations.robojudo_x2 --help
python -m xrobotoolkit_teleop.integrations.robojudo_x2
```

Run it in this environment alongside RoboJuDo-Plus's `x2_xrobotoolkit` simulation
preset or `x2_xrobotoolkit_real` AimDK preset in the robot environment. The bridge
requires no RoboJuDo Python imports. It binds arm commands on port 8559 and
subscribes to measured-state feedback on port 8571 by default; both endpoints
are configurable. Start PC Service and headset streaming before enabling tracking.

The bridge observes all 31 robot joints and commands only the 14 arm joints.
Hand targets follow the measured torso, while grip release holds the last
commanded targets. It uses the same default settings and collision helpers as the
standalone demo: arm cap 8 rad/s, return cap 1 rad/s, 50 Hz, scale 1.0, collision
margin 0.02 m, activation distance 0.08 m, and XR timeout 0.25 s. URDF/deployment
limits remain authoritative; no additional receiver EMA is used.

Tracking holds on stale input and automatically rebases controller references
when fresh data returns, discarding motion made during a transient hold. Startup,
explicit mode changes, and a restarted bridge require grip release before engagement.
IK starts from acknowledged arm targets; measured servo lag cannot pin an ungripped
arm outside its command speed window. Placo combines the configured default posture with
its upstream manipulability objective; LeRobot retains its joint-angle objective and filter.
These methods can select different redundant-joint postures for matching hand poses.
Nonlinear collision rejection tries bounded shorter steps, each checked against
both measured and acknowledged configurations. Explicit teleop disable requests
a checked return to defaults, which can run without XR tracking. If the bridge is offline the robot retains
that request and holds until it recovers. XR `Y` held also requests a return;
loss of XR freshness ends that button-driven return. Walking, runtime modes,
and recording controls remain in RoboJuDo. For finger control, select the
[OmniHand profile](#omnihand-profile).

The reusable `common.x2_arm_retargeter.X2ArmRetargeter` accepts measured joint
positions, acknowledged arm targets, defaults, limits, intent, and optional
`X2XRFrame` input. It returns a complete float32 arm proposal or `None` to hold.
Direct users must call `acknowledge(applied_positions)` after confirmed application, or `reject()`
on rejection; publishing a proposal does not commit backend history. The bridge manages this lifecycle automatically.
It restores body DOFs in the bundled model for observation, masks them during
IK, and checks the final measured-to-target and previous-to-target paths without
advancing physics. Robot-owned session, generation, feedback, and command IDs
prevent old proposals from executing after mode changes or restarts.

## Timing diagnostics

For timing diagnostics, run the bridge with `--xr-diagnostics`. It reports SDK
observations and the latest policy, stepping, retargeting, and acknowledgement durations.
The measurement-age lower bound excludes transport and queueing; RoboJuDo performs
the authoritative deadline check using its own clock.

`retarget` includes XR SDK sampling as well as retargeting. The additional
`xr_sample` and `ik` p95 values separate SDK reads from backend proposal work
(including solver transactions). The `IK` line reports solve attempts, warm-start
use, and total IPOPT iterations per step. Idle/anchor frames do not reuse old solver
metrics. The synthetic latency benchmark excludes SDK reads and, by default,
engages both arms and acknowledges every command immediately. Its optional
single-arm/delayed-application replay is described in [the LeRobot guide](lerobot_ik.md).

Keeping one grip released previously reset LeRobot's whole warm start on every
frame. The retargeter now resets on an actual release transition, preserving the
continuing arm's accepted warm start. Completion still commits only the exact
proposal reported applied by the robot; publishing or discarding proposals cannot
advance solver history. Completion latency starts after publication and includes
robot application and acknowledgement delays, so faster IK alone cannot remove it.

To measure SDK updates without IK or ZMQ, stop the bridge and other SDK clients,
leave PC Service/headset streaming, and run:

```bash
python -m xrobotoolkit_teleop.integrations.xr_receive_rate --duration 30 --csv /tmp/xr-rate.csv
```

This samples timestamps at 200 Hz; it does not count every network packet.
It reports the observed rate, recent gap p95 (last 4096 updates), lifetime maximum
gaps, and timeout episodes, without comparing headset and host clocks.

## OmniHand profile

Select `--robot x2_omnihand` with either backend. This profile observes 69 joints, optimizes 14 arm joints,
and produces 24 additional active finger targets from independent controller triggers (grips clutch arms only).
The version 2 protocol requires the matching RoboJuDo OmniHand XR preset. The hardware SDK and serial-selected
HCAN adapters live on the robot; the XR environment has no vendor SDK dependency.

Hand targets use coordinated interpolation and nonlinear passive coupling for sampled collision paths. HOME
retains finger targets. Combined acknowledgement covers arm submission plus both hand writes. A failed finger
step can fall back to independently checked arm motion with fingers held. Existing arm-only profiles retain
version 1 and their behavior. See RoboJuDo's guide for local serial configuration, read-only verification,
worker lifecycle, hardware acceptance, and the additional MuJoCo model.

Run `PYTHONPATH=. python -m unittest discover -s tests -p test_robojudo_omnihand.py`; set
`BRIDGE_TEST_BACKEND=lerobot` to exercise the alternative solver.

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
Arm-only profiles use schema version 1; the OmniHand profile uses version 2. Robot-owned deadlines remain authoritative. Only one publisher may own the endpoint.


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
   not joint-command validation. Use schema version 1 for arm-only profiles. The bundled `x2_omnihand` profile
   uses version 2 for fingers; base commands remain outside this integration.
5. Test feedback-name validation, moving reference frames, clutch mapping, limits/collision checks, HOME, and
   acknowledgement rollback. `tests/test_robojudo_profiles.py` demonstrates a one-joint profile with non-X2 names,
   a custom device mapping, and a stateful test solver.

Select a compatible registered IK method independently of the robot profile. Methods must support pose tracking,
joint-target HOME and linear constraints. LeRobot now sizes optimization to the configured scalar controlled joints.
Placo omits its 6D manipulability task for groups with fewer than six controlled joints, avoiding a singular objective.
LeRobot uses forward differentiation of its pose-error function to avoid NaN gradients from inactive rotation-log
branches at axis-aligned poses; its objective and filter are unchanged.
Adding another method uses `ik/registry.py`; adding another robot uses `robot_profiles/PROFILES`.

## Shared collision checking

Both model adapters support optional `distance_subset(pair_indices)` queries, returning exact signed
distances and copied witnesses in requested pair order. The checker uses conservative primitive bounds to
exclude distant pairs before requesting exact results. `clearances(threshold)` caps those results at the
requested threshold; it keeps a separate cache from exhaustive `distances()` and its exact motion references.
Certified capped clearances also provide a separate motion-bound reference: subtracting each shape's maximum
displacement preserves a lower bound without treating capped values as exact distances.
An additional 1 micrometre guard includes borderline pairs in exact queries. Geometry, configured pairs,
clearance margins, and coupled-path sample spacing are unchanged. Finite-difference penetration recovery
continues to use exhaustive signed queries. Adapters without subset support retain exhaustive behavior.
Placo uses its own native geometry placements with Coal primitive distance queries; no second robot/FK model,
Pinocchio, or CasADi is required. Coal's Python bindings are included in the `x2` extra. A Placo-only installation
without those bindings, or a model with unsupported primitives, retains exhaustive queries. The shared X2
simulation controller also uses the threshold-aware checker for both backends.

Run the native subset/oracle regression tests with:

```bash
PYTHONPATH=.:tests X2_TEST_LEROBOT=1 python -m unittest test_collision_subset
```

Compare warm bridge retargeting times without a headset or robot:

```bash
PYTHONPATH=. python scripts/misc/benchmark_bridge_collisions.py --steps 60 --repeats 3
PYTHONPATH=. python scripts/misc/benchmark_bridge_collisions.py --ik-backend placo --steps 60 --repeats 3
```

This sequential benchmark alternates exhaustive and selective execution order. It reports retargeting and
collision median/p95 times, exact pair-query counts, and held cycles for stationary fingers, closing fingers,
servo lag, and repeated penetrating recovery snapshots. Every accepted command is independently validated
with exhaustive signed queries at every path sample outside the timed region. Setup and six warmup frames
are excluded. Run without concurrent tests or other substantial CPU workloads when comparing performance.

The timing tables below predate the stricter oracle: their reference validation used exhaustive baseline
distances but still allowed conservative bounds for path samples. The current benchmark disables bounds in
the oracle, so all sampled configurations receive exact signed-distance checks.

LeRobot validation on 2026-11-11 used 40 measured cycles per run and three repeats per mode/scenario,
with native Python 3.11 dependencies. The following are medians of the three run-level statistics;
retarget times exclude XR reads and transport. All 24 runs had zero held cycles and every accepted
command passed reference path validation. These are replay measurements, not live hardware guarantees.

| Scenario | Collision median, exhaustive → selective | Retarget median, exhaustive → selective | Retarget p95, exhaustive → selective |
| --- | ---: | ---: | ---: |
| Fingers stationary | 7.27 → 1.31 ms | 22.59 → 15.70 ms | 26.25 → 18.86 ms |
| Fingers closing | 8.14 → 2.14 ms | 24.31 → 17.71 ms | 27.54 → 19.88 ms |
| Servo lag | 14.53 → 1.45 ms | 30.19 → 15.94 ms | 34.49 → 18.50 ms |
| Repeated penetrating snapshot | 237.75 → 207.85 ms | 262.94 → 233.39 ms | 269.65 → 240.19 ms |

Normal tracking required a median 41–45.5 exact pair queries per cycle, versus 4,563 for exhaustive checking
(servo lag required 9,126 exhaustive queries). Recovery intentionally retains the expensive exhaustive
finite-difference derivatives and remains outside a 20 ms control budget.

The same 40-cycle, three-repeat comparison for Placo produced the following medians of run-level statistics.
All 24 runs had zero held measured cycles and all accepted commands passed reference path validation.

| Scenario | Collision median, exhaustive → selective | Retarget median, exhaustive → selective | Retarget p95, exhaustive → selective |
| --- | ---: | ---: | ---: |
| Fingers stationary | 8.46 → 1.18 ms | 10.09 → 2.32 ms | 11.07 → 2.54 ms |
| Fingers closing | 8.81 → 1.82 ms | 10.96 → 3.43 ms | 11.62 → 3.86 ms |
| Servo lag | 16.63 → 1.31 ms | 18.30 → 2.46 ms | 23.71 → 2.65 ms |
| Repeated penetrating snapshot | 274.35 → 242.93 ms | 276.81 → 245.25 ms | 283.17 → 250.48 ms |
