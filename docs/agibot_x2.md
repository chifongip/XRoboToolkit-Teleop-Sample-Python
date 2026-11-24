# Agibot X2 standalone simulation

Use this guide for the fixed-base MuJoCo demo. For live measured-state control through RoboJuDo-Plus, use the [bridge guide](robojudo_bridge.md). See the [IK backend guide](lerobot_ik.md) for LeRobot setup and the [asset reference](../assets/agibot/x2/README.md) for model provenance.

Run all commands below from this repository’s root.

## Installation

For the fixed-base X2 simulation, activate a Python 3.10+ environment and install
the `x2` extra from this repository's root:

```bash
python -m pip install -e ".[x2]"
```

For a standalone Placo-only installation, a fresh environment is recommended for dependency isolation. With Conda installed, create a dedicated
Python 3.11 environment and run these commands from this repository's root:

```bash
conda create -n x2-standalone python=3.11 pip -y
conda activate x2-standalone
python -m pip install -e ".[x2]"
```

If reusing an existing Python 3.10+ environment, activate it and run the pip command.
Activate the same environment in later terminals before running the sample. Install PC Service separately.

This installs the shared dependencies (NumPy, Meshcat, Placo, and the XRoboToolkit
SDK) plus MuJoCo, Tyro, and PyZMQ, including their transitive dependencies. It supports
both the arm-only and `--with-omnihand` simulated X2 examples. It does not install
other robot SDKs, Torch, camera packages, or dexterous-hand retargeting packages.
The SDK is installed through pip; no SDK source checkout or setup script is needed.

Verify the installation without a headset or graphical desktop:

```bash
python -c "from xrobotoolkit_teleop.simulation.agibot_x2_teleop_controller import AgibotX2TeleopController; print('OK')"
python -m pip check
```

Install and start [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)
separately, connect the headset/controllers, and use a graphical desktop to run:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py
```

A plain `pip install -e .` now installs only shared dependencies. Use
`python -m pip install -e ".[all]"` for the previous complete Python dependency
set, or `".[all,dev]"` to include development tools. Other hardware examples may
still require external SDKs and system setup. The legacy setup scripts request
the `all` extra and are not needed for X2-only installation.


## Run the simulation

The optional `--ik-backend lerobot` selects LeRobot's G1-style nonlinear IK
adapted to X2. Placo remains the default. See the [backend setup and behavior](lerobot_ik.md)
for native dependencies, shared command protection, and headless comparisons.

Activate the installed environment (for example, `conda activate x2-standalone`),
start XRoboToolkit PC Service, and connect the headset and two VR controllers:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py
```

The bundled X2 v1.3.0 model keeps the pelvis, legs, waist, and head fixed. Hold
the left or right **grip** to control the corresponding hand center's position and
orientation. The default `--tracking-reference fixed-heading` uses a fixed frame
aligned to your forward heading when the first grip engages. Use
[`--tracking-reference waist`](#waist-relative-reference) to compensate shared
waist and controller motion instead.
Moving your head alone leaves the hand targets unchanged. In the default mode,
turning your body while gripping moves the targets if it changes the controllers'
world poses.
Release to hold its last joint command; press again to start from
the current controller pose. Both arms can move simultaneously.

Hold **Y** to return both arms to the default pose; release Y to hold the
current joint commands. This overrides the arm grips while held and respects
a separate joint target speed limit of **1.0 rad/s** (about 57 degrees/s), bounded by
the arm and URDF limits. Set `--return-velocity-limit 0.25` for a slower return.
When a direct step is blocked, a collision-constrained posture solve seeks
clearance around the obstacle. HOME requires a live XR stream and uses the
robot's measured state and collision checks. Controller, headset, and pelvis
poses are unused. Omnihand fingers hold during HOME. Release both grips after
HOME to resume tracking.

To attach both Omnihand Pro hands, use:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --with-omnihand
```

Each **trigger** continuously closes its corresponding hand; release it to open.
Fingers work independently of the arm grips. The hands retain their nonlinear
finger coupling, mass, and inertia, with contacts disabled. This mode demonstrates
hand motion without object grasping. The default arm-only model assigns no trigger
action. The optional IK visualization includes the articulated hands and shows
the accepted commands, including coupled finger joints. See [hand asset provenance](../assets/omnihand_pro/README.md).

Both modes use `left_hand_center` and `right_hand_center` for IK and target
markers. These fixed frames sit at the bounding-box centers of the rigid hand
meshes in default mode, or the palm meshes with Omnihand Pro enabled. Their axes
remain parallel to the wrist-roll frames. Finger motion does not move the frame
within the hand. The Omnihand option automatically selects its matching
`x2_omnihand_ik.urdf`; custom scenes require a URDF with matching center frames.

Use `--scale-factor 0.5` for smaller translation movements, or
`--visualize-ik` to also open the shared browser IK visualization with either
backend. `--xml-path` and `--robot-urdf-path` accept matching alternative
fixed-body models. Use a custom
`--xml-path` separately from `--with-omnihand`. Alternative models must use the
same 14 arm joints, optionally with both Omnihand Pro hands, and joint position servos
with unit gear. Close the
MuJoCo viewer or press Ctrl-C to exit. A graphical desktop is needed for the
viewer; the headless checks below require neither a headset nor PC Service.

```bash
python -m unittest discover -s tests -v
```

Replay default → T pose → default without a headset or XR PC Service:

```bash
python scripts/misc/test_ik_offline.py --robot x2 --ik-backend placo --visualize
python scripts/misc/test_ik_offline.py --robot x2 --ik-backend lerobot --with-omnihand --visualize
```

`--visualize` shows the simulated robot in MuJoCo. Add `--visualize-ik` to open
the shared browser IK view, or omit both flags for a headless check. The replay
uses mock XR poses and real IK, collision checks, position servos, and physics.
It repeats two cycles by default; use `--cycles`, `--move-seconds`, and
`--hold-seconds` to adjust the motion. It prints a JSON report and exits with
status 1 if a check fails. See the [offline replay details](lerobot_ik.md#offline-pose-cycle-test).

The demo defaults to **50 Hz** control (20 ms per cycle). Set the target rate with
`--control-frequency`, for example:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --with-omnihand --control-frequency 100
```

Loop pacing and Placo's IK integration and command speed limits use the configured
period. Physics steps are at most 1 ms in the bundled scenes; rates such as 60 Hz
use slightly smaller steps so each cycle advances exactly one control period.
Collision checks can overrun the target period.

Both backends enforce the following arm speed and collision protections.
Each uses its own model adapter: Placo native geometry queries for `placo`,
and Pinocchio/Coal queries for `lerobot`. Shared code builds linearized
clearance constraints from cached exact witnesses, handles penetration recovery,
and validates sampled command paths. LeRobot does not call a Placo solver.
Arm speed bounds are centered on the previous accepted servo commands, so
servo lag does not increase the permitted command step.

Set the arm IK joint speed cap with `--arm-velocity-limit 2.0` (rad/s; default 8.0). Each arm joint
uses the lower of this value and its URDF limit (4.188 rad/s for wrist pitch/roll, 13.088 rad/s for
shoulder pitch/roll, and 15.077 rad/s for shoulder yaw, elbows, and wrist yaw).
With Omnihand Pro, `--hand-velocity-limit` separately caps actuated finger target
speeds (rad/s; default 2.0), also bounded by each joint's supplied hand URDF limit.
The former `--velocity-limit` option is now `--arm-velocity-limit`.

Self-collision avoidance is enabled for both backends and X2 models, using fitted cylinders
and compact boxes for the body and arms, with rounded torso sections. Each
Omnihand preserves its bundled URDF joint chains and collision primitives, updated
from all measured finger joints, including passive coupled joints.
`--collision-margin 0.02` sets the minimum commanded clearance in meters;
`--collision-trigger 0.08` activates avoidance within 8 cm and must exceed the
margin. End-effector tracking yields to this hard constraint. The controller
also checks sampled motion from measured positions and previous servo targets
to the outgoing targets, including the held targets of released arms. Invalid
solutions hold the previous commands. Custom IK URDFs must provide collision
geometry and joints for the same configured links. The Omnihand IK model includes
14 arm and 38 observed finger joints; finger DOFs are fixed during arm IK.
Every changed arm/hand command is checked as a combined configuration, including
held finger targets. Sampled finger motion follows the source MJCF polynomial
coupling, with passive-joint tracking error carried from the measured state.
If measured clearance drops below the margin, holding the affected arm's grip
allows gradual recovery within the joint speed limits. Existing violations must
improve without worsening other pairs or creating new violations. Recovery also
checks intermediate poses and shortens steps when needed; releasing the grip
continues to hold that arm.

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --with-omnihand --arm-velocity-limit 4.0 --hand-velocity-limit 1.0
```

Finger position commands ramp together toward trigger targets, including after
stream recovery. Each hand uses a common progress step bounded by its slowest
moving joint, keeping the thumb and fingers coordinated during reversals.
This bounds commanded speeds, not actual physics velocities; passive
fingers still follow their coupling constraints and have no separate speed clamp.
Control requires an advancing XR stream timestamp. If it stops advancing for
250 ms, joint commands hold and clutch references reset. Omnihand Pro commands
hold during stream outages and resume trigger control when fresh packets arrive.
Non-finite trigger values hold only the
affected hand, and finite values are clamped to the trigger's 0–1 range.
The SDK exposes a shared stream timestamp, so
this watchdog detects stream outages, not individual devices dropping out while
other packets continue arriving. Invalid active controller poses also hold
both arms' commands and reset both clutch references, so recovery discards
controller motion made during the hold.

### Fixed-heading reference

When the first grip engages in `fixed-heading` mode, the headset's forward direction
projected onto the horizontal plane sets the control heading. Look forward at engagement. This
heading stays fixed while either grip is held: later head translation, pitch,
roll, or yaw cannot shift targets. Both arms share the same heading even if their
grips engage at different times. Release **both grips** before repositioning;
the next engagement captures a new heading and rebases each arm at its measured
hand pose, without a target jump. Returning with Y also ends the current session.
The headset is needed only to capture the heading; invalid poses or a vertical
forward direction hold commands until a valid heading can be captured.
Enable headset streaming in the XR app before engaging a grip. The SDK's shared
timestamp cannot distinguish a cached headset pose from a live, stationary one;
re-engaging with headset streaming disabled can capture an old heading.
Tracking outages rebase controller deltas but preserve the locked session heading.

The XR adapter expects native PICO tracking-world coordinates: +X right, +Y up,
+Z backward, with poses `[x, y, z, qx, qy, qz, qw]` and positions in meters.
In `fixed-heading` mode, the session heading is removed before the fixed basis
conversion
`[robot_x, robot_y, robot_z] = [-vr_z, -vr_x, vr_y]`. Rotations use the same
basis change. Robot coordinates are +X forward, +Y left, +Z up.
Clutching preserves the hand's initial orientation and applies controller
rotation changes, rather than imposing its absolute pose. Headset position is
not needed because controller translation is already measured from engagement.

### Waist-relative reference

Enable and calibrate PICO full-body tracking, then run:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --tracking-reference waist
```

PICO's native tracker calibration is sufficient; no additional calibration file or
SDK modification is required. This mode uses the estimated pelvis (joint 0), not
an individual tracker selected by serial number. It also supports `--with-omnihand`.
The simulated robot's waist remains fixed.

When the operator's waist and controllers move together, robot hand targets stay
unchanged relative to the robot torso. Compensation includes translation, yaw,
pitch, and roll. Moving only the waist beneath room-stationary hands changes
those targets. Head pose is unused.

The first grip engagement sets the pelvis-to-robot control axes for the session.
The right-to-left hip line defines left in the pelvis frame. The calibrated
pelvis joint supplies up, and forward is perpendicular to both. The same
pelvis-relative reach maps to the same robot direction across starting headings
and leans.
Both arms share the mapping through turns, grip releases, HOME, and tracking
recovery. Each grip engagement waits for a new body frame, captures the
pelvis-relative controller pose, and anchors the target at the measured robot
hand pose. Restart to set the axes again. Translation scale applies to relative
displacement. Rotation deltas change hand orientation around the hand center.

Release both grips after startup, then engage independently. Release an
arm's grip to reposition that controller without moving its target. Triggers work
independently during normal tracking. Waist tracking faults hold both arms and
fingers and clear references. Recovery waits for the XR packet and body frame
timestamps to advance, then captures fresh references. Engagement and recovery
frames hold existing arm commands. Change modes by restarting.

The adapter brackets pose reads with the XR packet timestamp and the body
timestamps, retrying up to three times. The packet timestamp and a body frame
timestamp must each advance within 250 ms. If the aggregate timestamp is
missing or zero, the timestamp on joint 15 of the body skeleton is used.
That timestamp stayed frozen when the headset repeated a cached skeleton while
continuing to stream controllers. Repeated body frames hold the last
pelvis-relative targets until a new body frame arrives or the timeout expires.
Rollback clears references.
Per-sample jumps above 0.5 m or 1 rad in raw or pelvis-relative poses also hold
control. A gap between valid
samples longer than the timeout also clears the clutch references.

Record tracking data without commanding a robot:

```bash
python scripts/misc/diagnose_x2_waist.py --duration 30 --rate 50 > /tmp/x2-waist-recording.jsonl
```

The diagnostic records poses, timestamps, observed update rates,
relative transforms, and hold reasons. Compare the separately read
`pelvis_timestamp_ns` and `pelvis_observed_update_hz` with the aggregate
`sample.body_timestamp_ns` and the first entry of `observed_update_hz`.
The capture also reports `sample.body_head_timestamp_ns`. The hold reason
identifies an XR packet or body-frame timeout. Non-finite values become JSON null,
and SDK initialization may also print status lines. Record standing, head-only motion,
walking, turning, bending, and independent hand motion to assess residual motion
and pelvis/head coupling.

The SDK can retain old fields while the packet timestamp advances. The checks
detect a frozen full-body frame. Bracketed reads cannot independently verify
each joint or controller's freshness, guarantee synchronized poses, or detect
every tracking-origin reset. Local observation times are not packet arrival times.
Read-only headset captures validated the timestamp checks. Physical robot
qualification remains pending.

### Coordinate conventions and simulation limits

The upstream PC Service README inconsistently labels headset and controller
handedness. The [Unity sender](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/main/Assets/Scripts/TrackingData.cs)
serializes the native predicted controller poses, and the
[Python binding](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind/blob/main/bindings/py_bindings.cpp)
parses them without an axis conversion. This verifies the source-level mapping;
the installed headset app's behavior still requires a live direction check.

This is simulation only, without grasping or locomotion. Collision avoidance
uses conservative primitives and sampled joint-space checks; it does not guarantee
collision-free physics motion, check the environment, or resolve contacts
between fingers. MuJoCo self-contact forces remain disabled.
See [model provenance, collision pairs, and timing](../assets/agibot/x2/README.md).
