# Omnihand Pro source assets

Copied from `/home/ubuntu/o12_hand_description-o12_t3` for the optional X2
MuJoCo demo. `MJCF/`, `meshes/`,
and both `omnihand_pro_*.urdf` files are unmodified supplied assets.
`package.xml` and `README_EN.md` preserve upstream metadata and usage notes.
No Git revision metadata was supplied.

The supplied `package.xml` declares `Apach-2.0` (upstream spelling). No separate
license text or NOTICE file was present. These assets retain that upstream
declaration and are not covered by this project's MIT license. X2 wrist and
body assets retain their separate Mulan PSL v2 license.

The combined scene uses the source MJCF finger dynamics, 12 actuators per hand,
and seven nonlinear coupling constraints per hand. Explicit palm inertia comes
from the matching Omnihand Pro URDFs. The generator disables contacts and visual mesh
mass, replaces standalone palm placement with wrist mounts, and sets joint limit
margin to zero instead of 0.1 rad so fingers can reach the open pose. Other
source joint dynamics and actuator limits/gains are preserved.

The controller moves each hand's actuated targets by a common fraction of the
remaining distance to its trigger pose. This keeps the thumb and fingers on the
same pose interpolation during partial trigger changes and reversals. The
fraction is bounded so no joint changes by more than
`min(hand_velocity_limit, URDF joint velocity) * dt` per control cycle; some
joints move slower to stay coordinated with the limiting joint.
`--hand-velocity-limit` defaults to 2.0 rad/s and is independent of
`--arm-velocity-limit` (default 8.0 rad/s). Stale XR data or invalid triggers
hold commands; recovery resumes the ramp with one control step's allowance.
These are command speed limits, not hard limits on simulated joint velocities.
Passive coupled joints are not separately rate limited.

The default closed target uses thumb MCP -0.05 rad and PIP -0.6 rad on both
hands. The shallower proximal flexion keeps the thumb outside the curled
fingers instead of intersecting them. Thumb roll/abduction remain mirrored
at -0.6/+0.6 rad on the left and +0.6/-0.6 rad on the right. The four finger
targets are unchanged. Collision-mesh regression checks sample both hands
during opening, closing, and the partial-trigger reversal sequence found in
review, at hand speed caps of 0.5, 2.0, and 8.0 rad/s.
The combined X2 IK model now includes these URDF joint chains, with per-link
boxes, cylinders, and spheres following measured finger motion. The generated
model preserves the supplied primitive shapes and their local transforms. Trigger proposals are
checked against the body, opposite arm, and opposite hand. Internal same-hand
pairs remain excluded; the preset thumb clearance checks do not constitute
runtime finger-to-finger avoidance or a grasp controller.

The short v1.4 X2 wrists and existing wrist-to-palm transforms remain in use.
This hand-only package does not supply an X2 adapter transform; the current
mount is an initial simulation assembly, not a physical mounting specification.
The source URDF approximates coupling linearly; the simulation uses MJCF
nonlinear coupling. The generated collision/IK model removes mimic tags and
observes all actual finger joint positions, so passive fingers follow measured
nonlinear coupling rather than the URDF approximation.

Regenerate: `python scripts/simulation/build_x2_omnihand_assets.py`.
Launch: `python scripts/simulation/teleop_agibot_x2_mujoco.py --with-omnihand`.
