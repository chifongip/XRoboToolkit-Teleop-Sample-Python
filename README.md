# XRoboToolkit-Teleop-Sample-Python

Pico teleoperation demo written in python for both mujoco simulation and robot hardware.

## Overview

This project provides a framework for controlling robots in robot hardware and MuJoCo simulation through XR (VR/AR) input devices. It allows users to manipulate robot arms using natural hand movements captured through XR controllers.

## Installation

### X2-only installation

For the fixed-base X2 simulation, activate a Python 3.10+ environment and install
the `x2` extra from this repository's root:

```bash
python -m pip install -e ".[x2]"
```

A fresh environment is recommended for dependency isolation, but an existing
compatible environment also works. With Conda installed, create a dedicated
Python 3.11 environment and run these commands from this repository's root:

```bash
conda create -n robojudo-xr python=3.11 pip -y
conda activate robojudo-xr
python -m pip install -e ".[x2]"
```

If reusing an existing Python 3.10+ environment, activate it and run only the pip
command. RoboJuDo-Plus's installer requires Python 3.11+, so use Python 3.11 for
an environment shared with that integration. In later terminals, activate
`robojudo-xr` again before running the sample. These commands install only the
X2 Python dependencies; PC Service remains a separate installation. The X2-only
pip installation does not create or switch Conda environments.

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

### RoboJuDo-Plus X2 bridge

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
arm outside its command speed window. The posture preference remains at the configured
default pose, so repeated near-straight-arm movements do not accumulate redundant-joint drift.
Nonlinear collision rejection tries bounded shorter steps, each checked against
both measured and acknowledged configurations. Explicit teleop disable requests
a checked return to defaults, which can run without XR tracking. If the bridge is offline the robot retains
that request and holds until it recovers. XR `Y` held also requests a return;
loss of XR freshness ends that button-driven return. Walking, runtime modes,
and recording controls remain in RoboJuDo. OmniHand integration is deferred.

The reusable `common.x2_arm_retargeter.X2ArmRetargeter` accepts measured joint
positions, acknowledged arm targets, defaults, limits, intent, and optional
`X2XRFrame` input. It returns a complete float32 arm proposal or `None` to hold.
It restores body DOFs in the bundled model for observation, masks them during
IK, and checks the final measured-to-target and previous-to-target paths without
advancing physics. Robot-owned session, generation, feedback, and command IDs
prevent old proposals from executing after mode changes or restarts.

For timing diagnostics, run the bridge with `--xr-diagnostics`. It reports SDK
observations and the latest policy, stepping, retargeting, and acknowledgement durations.
The measurement-age lower bound excludes transport and queueing; RoboJuDo performs
the authoritative deadline check using its own clock.

To measure SDK updates without IK or ZMQ, stop the bridge and other SDK clients,
leave PC Service/headset streaming, and run:

```bash
python -m xrobotoolkit_teleop.integrations.xr_receive_rate --duration 30 --csv /tmp/xr-rate.csv
```

This samples timestamps at 200 Hz; it does not count every network packet.
It reports the observed rate, recent gap p95 (last 4096 updates), lifetime maximum
gaps, and timeout episodes, without comparing headset and host clocks.

See the [RoboJuDo-Plus guide](https://github.com/chifongip/RoboJuDo-Plus/blob/main/docs/xrobotoolkit_teleop.md)
for two-terminal setup, protocol details, and synthetic two-process validation.
Real hardware acceptance is separate from automated MuJoCo checks.

### Full installation and hardware setup

1. Download and install [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service). Run the installed program before running the following demo.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
    cd XRoboToolkit-Teleop-Sample-Python
    ```

3.  **Installation**
    **Note:** The setup scripts are currently only tested on Ubuntu 22.04.
    It is recommended to setup a Conda environment and install the project using the included script.
    ```bash
    bash setup_conda.sh --conda <optional_env_name>
    conda activate <env_name>
    bash setup_conda.sh --install
    ```

    If installing on system python:
    ```bash
    bash setup.sh
    ```

## Usage
Use the following instructions to run example scripts. For a more detailed description, please refer to [`teleop_details.md`](teleop_details.md).

### Running the MuJoCo Simulation Demo

To run the teleoperation demo with a UR5e robot in MuJoCo simulation:

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```
This script initializes the [`MujocoTeleopController`](xrobotoolkit_teleop/simulation/mujoco_teleop_controller.py) with the UR5e model and starts the teleoperation loop.

To run the teleoperation demo with dual Agilex Piper robotic arms in MuJoCo simulation:

```bash
python scripts/simulation/teleop_dual_piper_mujoco.py
```

### Agibot X2 fixed-base dual-arm simulation

The optional `--ik-backend lerobot` selects LeRobot's G1-style nonlinear IK
adapted to X2. Placo remains the default. See the [backend setup and behavior](docs/lerobot_ik.md)
for native dependencies, joint-only tracking limits, and headless comparisons.

Activate the installed environment (for example, `conda activate xr-robotics`),
start XRoboToolkit PC Service, and connect the headset and two VR controllers:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py
```

The bundled X2 v1.3.0 model keeps the pelvis, legs, waist, and head fixed. Hold
the left or right **grip** to control the corresponding hand center's position and
orientation using controller motion in a fixed frame aligned to your forward
heading when the first grip engages.
Moving your head alone leaves the hand targets unchanged. Turning your body
while gripping moves the targets if it changes the controllers' world poses.
Release to hold its last joint command; press again to start from
the current controller pose. Both arms can move simultaneously.

Hold **Y** to return both arms to the default pose; release Y to hold the
current joint commands. This overrides the arm grips while held and respects
a separate joint target speed limit of **1.0 rad/s** (about 57 degrees/s), bounded by
the arm and URDF limits. Set `--return-velocity-limit 0.25` for a slower return.
When a direct step is blocked, a collision-constrained posture solve seeks
clearance around the obstacle. Every step still passes collision checks; motion
pauses if tracking is lost or no safe step is found. Controller motion rebases
when Y is released.
Omnihand fingers remain under trigger control.

To attach both Omnihand Pro hands, use:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --with-omnihand
```

Each **trigger** continuously closes its corresponding hand; release it to open.
Fingers work independently of the arm grips. The hands retain their nonlinear
finger coupling, mass, and inertia, with contacts disabled. This mode demonstrates
hand motion without object grasping. The default arm-only model assigns no trigger
action. Placo visualization includes the articulated hands and follows their
measured finger joints. See [hand asset provenance](assets/omnihand_pro/README.md).

Both modes use `left_hand_center` and `right_hand_center` for IK and target
markers. These fixed frames sit at the bounding-box centers of the rigid hand
meshes in default mode, or the palm meshes with Omnihand Pro enabled. Their axes
remain parallel to the wrist-roll frames. Finger motion does not move the frame
within the hand. The Omnihand option automatically selects its matching
`x2_omnihand_ik.urdf`; custom scenes require a URDF with matching center frames.

Use `--scale-factor 0.5` for smaller translation movements, or
`--visualize-placo` to also open the IK visualization. `--xml-path` and
`--robot-urdf-path` accept matching alternative fixed-body models. Use a custom
`--xml-path` separately from `--with-omnihand`. Alternative models must use the
same 14 arm joints, optionally with both Omnihand Pro hands, and joint position servos
with unit gear. Close the
MuJoCo viewer or press Ctrl-C to exit. A graphical desktop is needed for the
viewer; the headless checks below require neither a headset nor PC Service.

```bash
python -m unittest discover -s tests -v
```

The demo defaults to **50 Hz** control (20 ms per cycle). Set the target rate with
`--control-frequency`, for example:

```bash
python scripts/simulation/teleop_agibot_x2_mujoco.py --with-omnihand --control-frequency 100
```

Loop pacing and Placo's IK integration and command speed limits use the configured
period. Physics steps are at most 1 ms in the bundled scenes; rates such as 60 Hz
use slightly smaller steps so each cycle advances exactly one control period.
Collision checks can overrun the target period.

The following arm speed and collision details apply to the default `placo`
backend. The `lerobot` backend uses the [joint-bounded behavior described here](docs/lerobot_ik.md).

Placo IK reuses the baseline query's exact collision witnesses to build the same
linearized clearance constraints for nearby pairs. This avoids a second full
distance query inside the solver. Joint limits, velocity limits, penetration
recovery, and nonlinear command-path checks remain enabled.

Set the arm IK joint speed cap with `--arm-velocity-limit 2.0` (rad/s; default 8.0). Each arm joint
uses the lower of this value and its URDF limit (4.188 rad/s for wrist pitch/roll, 13.088 rad/s for
shoulder pitch/roll, and 15.077 rad/s for shoulder yaw, elbows, and wrist yaw).
With Omnihand Pro, `--hand-velocity-limit` separately caps actuated finger target
speeds (rad/s; default 2.0), also bounded by each joint's supplied hand URDF limit.
The former `--velocity-limit` option is now `--arm-velocity-limit`.

With Placo, self-collision avoidance is enabled for both X2 models. Placo uses fitted cylinders
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
14 arm and 38 observed finger joints; finger DOFs are masked during arm IK.
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
250 ms, joint commands hold and clutch references reset; tracking recovery starts
from the measured hand-center poses. Omnihand Pro commands also hold during stream outages;
fresh packets resume trigger control. Non-finite trigger values hold only the
affected hand, and finite values are clamped to the trigger's 0–1 range.
The SDK exposes a shared stream timestamp, so
this watchdog detects stream outages, not individual devices dropping out while
other packets continue arriving. Invalid active controller poses also hold
both arms' commands and reset both clutch references, so recovery discards
controller motion made during the hold.

When the first grip engages, the headset's forward direction projected onto the
horizontal plane sets the control heading. Look forward at engagement. This
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
The session heading is removed before the fixed basis conversion
`[robot_x, robot_y, robot_z] = [-vr_z, -vr_x, vr_y]`. Rotations use the same
basis change. Robot coordinates are +X forward, +Y left, +Z up.
Clutching preserves the hand's initial orientation and applies controller
rotation changes, rather than imposing its absolute pose. Headset position is
not needed because controller translation is already measured from engagement.

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
See [model provenance, collision pairs, and timing](assets/agibot/x2/README.md).

### Running the Placo Visualization Demo

To run the teleoperation demo with a UR5e robot in Placo visualization:

```bash
python scripts/simulation/teleop_x7s_placo.py
```
This script initializes the [`PlacoTeleopController`](xrobotoolkit_teleop/simulation/placo_teleop_controller.py) with the X7S robot and starts the teleoperation loop.

### Running Dexterous Hand Teleop Simulation
- Shadow hand simulation in Mujoco
    ```bash
    python scripts/simulation/teleop_shadow_hand_mujoco.py
    ```

- Inspire hand in Placo Visualization
    ```bash
    scripts/simulation/teleop_inspire_hand_placo.py
    ```

### Running the Hardware Demo (Dual UR5 Arms and Dynamixel-based Head)

To run the teleoperation demo with the physical dual UR arms and Dynamixel-based head:

1.  **Normal Operation:**
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py
    ```
    This script initializes the [`DynamixelHeadController`](xrobotoolkit_teleop/hardware/dynamixel.py) and [`DualArmURController`](xrobotoolkit_teleop/hardware/ur.py) and starts the teleoperation loops for both head tracking and arm control.

2.  **Resetting Arm Positions:**
    If you need to reset the UR arms to their initial/home positions and initialize the robotiq grippers, you can run the script with the `--reset` flag:
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py --reset
    ```
    This will execute the reset procedure defined in the [`DualArmURController`](xrobotoolkit_teleop/hardware/ur.py) and then exit.

3.  **Visualizing IK results:**
    To visualize the inverse kinematics solution with placo during teleoperation, run the script with the `--visualize_placo` flag.
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py --visualize_placo
    ```

### Running ARX R5 Hardware Demo

To run the teleoperation demo with dual ARX R5 robotic arms:

```bash
python scripts/hardware/teleop_dual_arx_r5_hardware.py
```

This script initializes the [`ARXR5TeleopController`](xrobotoolkit_teleop/hardware/arx_r5_teleop_controller.py) for dual arm control with built-in grippers.

### Running Galaxea R1 Lite Humanoid Demo

To run the teleoperation demo with the Galaxea R1 Lite humanoid robot:

```bash
python scripts/hardware/teleop_r1lite_hardware.py
```

This script initializes the [`GalaxeaR1LiteTeleopController`](xrobotoolkit_teleop/hardware/galaxea_r1_lite_teleop_controller.py) for mobile manipulator control, the controller communicates with the robot hardware via ROS.

### Running Agilex Piper Hardware Demo

Find CAN modules (follows [piper_sdk](https://github.com/agilexrobotics/piper_sdk)):

```bash
bash find_all_can_port.sh
```

If there are multiple modules, the output will look like this:

```
Both ethtool and can-utils are installed.
Interface can1 is connected to USB port 1-4.4.3:1.0
Interface can0 is connected to USB port 1-4.4.2:1.0
```

Activate the CAN devices:

```bash
bash can_activate.sh can1 1000000 "1-4.4.3:1.0"
bash can_activate.sh can0 1000000 "1-4.4.2:1.0"
```

To run the teleoperation demo with dual Agilex Piper robotic arms:

```bash
python scripts/hardware/teleop_dual_piper_hardware.py
```

This script initializes the [`PiperTeleopController`](xrobotoolkit_teleop/hardware/piper_teleop_controller.py) for dual arm control with built-in grippers.

## Data Collection

### Collecting Teleoperation Data

The framework automatically logs teleoperation sessions when running hardware demos. Data collection includes:

- **Robot joint states** and end effector poses
- **Camera streams** from multiple viewpoints
- **User input data** from XR controllers
- **Timestamp synchronization** across all data streams

#### Starting Data Collection

1. **Run any hardware teleoperation script:**
   ```bash
   python scripts/hardware/teleop_dual_arx_r5_hardware.py
   ```

2. **Press B button** on the VR controller to start/stop logging
   - First press: Start data logging
   - Second press: Stop logging and save data to disk

3. **Emergency stop:** Press right joystick click to discard current session

#### Data Storage

Collected data is saved as `.pkl` files in the `logs/` directory with timestamps:
```
logs/
├── <robot_name>/
│   └── teleop_log_YYYYMMDD_HHMMSS_<session_id>.pkl
└── <another_robot>/
    ├── teleop_log_YYYYMMDD_HHMMSS_<session_id>.pkl
    └── teleop_log_YYYYMMDD_HHMMSS_<session_id>.pkl
```

### Validating Collected Data

Use the provided analysis script to verify data integrity and examine collected datasets:

```bash
python scripts/misc/test_data_log_analysis.py logs/<robot_name>/teleop_log_YYYYMMDD_HHMMSS_1.pkl
```

This script will:
- Display available data fields and their types
- Verify robot states and camera images are properly saved
- Show sample entries and data statistics
- Count total logged entries

### Converting to LeRobot Dataset

For training imitation learning models, convert collected data to [LeRobot](https://github.com/huggingface/lerobot) format using this example conversion script:

**Example:** [ARX Dual Arm Data Converter](https://github.com/zhigenzhao/openpi/blob/dev/finetuning/examples/arx_r5/arx_dual/convert_dual_arm_data_to_lerobot.py)

This conversion enables:
- Standardized dataset format for machine learning
- Integration with LeRobot training pipelines  
- Support for various imitation learning algorithms
- Easy data sharing and reproducibility

## Teleoperation Guide

### Tracking Modes

The teleoperation system supports multiple tracking modes for controlling robot end effectors:

#### 1. Controller Tracking (Default)
- **Description**: Uses VR/AR controller poses to control robot end effectors
- **Use Case**: Primary method for precise manipulation tasks
- **Configuration**: Set `pose_source` to `"left_controller"` or `"right_controller"`
- **Tracking**: Full 6DOF pose (position + orientation) or 3DOF position-only

#### 2. Hand Tracking
- **Description**: Uses hand pose estimation from XR cameras
- **Use Case**: Natural hand gesture control

#### 3. Head Tracking
- **Description**: Uses headset pose for controlling specific robot components
- **Use Case**: Head/neck control for humanoid robots or camera orientation

#### 4. Motion Tracker Tracking
- **Description**: Uses additional motion tracking devices for controlling auxiliary robot links
- **Use Case**: Multi-point control (e.g., elbow position while controlling end effector)
- **Configuration**: Add `motion_tracker` config with device serial and target link
- **Note**: Not recommended for 6DOF arms like UR5e; better suited for redundant arms

### Controller Button Functions

When using VR controllers for teleoperation, the following button mappings apply:

#### **Grip Buttons**
- **Left Grip** (`left_grip`): Activates left arm teleoperation
- **Right Grip** (`right_grip`): Activates right arm teleoperation
- **Function**: Hold to enable arm control, release to deactivate

#### **Trigger Buttons**
- **Left Trigger** (`left_trigger`): Controls left gripper/hand
- **Right Trigger** (`right_trigger`): Controls right gripper/hand
- **Function**: Analog control (0.0 = fully open, 1.0 = fully closed)

#### **System Buttons**
- **A Button**: Reserved for system functions
- **B Button**: Toggle data logging on/off
  - Press once: Start logging
  - Press again: Stop logging and save data

#### **Joysticks/Touchpads**
- **Left Joystick**: Linear velocity commands for mobile robots
- **Right Joystick**: Angular velocity commands for mobile robots
- **Right Axis Click**: stop data logging (discards current data)


## Dependencies
XR Robotics dependencies:
- [`xrobotookit_sdk`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind): Python binding for XRoboToolkit PC Service SDK, MIT License

Robotics Simulation and Solver
- [`mujoco`](https://github.com/google-deepmind/mujoco): robotics simulation, Apache 2.0 License
- [`placo`](https://github.com/rhoban/placo): inverse kinematics, MIT License

Hardware Control
- [`dynamixel_sdk`](https://github.com/ROBOTIS-GIT/DynamixelSDK.git): Dynamixel control functions, Apache-2.0 License
- [`ur_rtde`](https://gitlab.com/sdurobotics/ur_rtde): interface for controlling and receiving data from a UR robot, MIT License
- [`ARX R5 SDK`](https://github.com/zhigenzhao/R5/tree/dev/python_pkg): Interface for controlling ARX R5 robotic arms

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
