# Agibot X2 arm simulation assets

Source: [Agibot X2 URDF repository](https://github.com/AgibotTech/agibot_x2_urdf),
commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, directory `X2_URDF-v1.3.0`,
supplied from `/home/ubuntu/agibot_x2_urdf`. The upstream models and meshes are licensed under
Mulan PSL v2; see [LICENSE](LICENSE). This license applies to these model assets,
independently of the surrounding project's MIT license.

`x2_ultra.urdf` is the unmodified source URDF. `meshes/` contains all 39 meshes
referenced by the source URDF and the adapted MuJoCo model.

Modifications for this demo:

- `x2_fixed.urdf` fixes legs, waist, and head at zero and adds a fixed world-to-base
  transform at z = 0.68 m. Only the seven joints in each arm remain movable.
  Collision elements are removed because collision avoidance is not enabled.
- `x2_fixed.xml` derives from the upstream `x2_ultra.xml`. It removes the floating
  base, non-arm joints, collision geometry, and original torque actuators/sensors.
  Arm position actuators use the source joint and effort limits, damping, and
  implicit integration. Meshes, body transforms, and inertias retain upstream data.
- `scene.xml` adds lighting, a floor, and non-colliding hand-center target markers.
- `left_hand_center` and `right_hand_center` are fixed frames at the bounding-box
  centers of the rigid hand meshes, with axes parallel to the wrist-roll frames.
  They are present in both `x2_fixed.xml` and `x2_fixed.urdf` and are the IK targets.
- The home pose uses shoulder pitch -0.3, shoulder roll +0.2/-0.2, elbows -1.0,
  and other joints zero (radians). Position servos support gravity-loaded arms;
  small steady-state tracking errors are expected.

This demo simulates arm dynamics, without self-contact or collision avoidance.
The upstream collision meshes overlap at multiple joints even in neutral posture;
they must be refined before enabling self-contact. Fingers, grasping, locomotion,
and hardware control are not included.

## Optional Omnihand Pro scene

`scene_omnihand.xml` loads `x2_omnihand.xml`, an adaptation of `x2_fixed.xml`
with both Omnihand Pro palm bodies fixed beneath their matching wrist-roll
bodies. This adds 38 finger joints and 24 position actuators; the 14-joint
`x2_omnihand_ik.urdf` is the matching 14-joint arm IK model with replacement
wrist geometry and hand-center offsets at the palm mesh centers. The frame
axes stay parallel to the wrist-roll frames. The `home` keyframe is rebuilt by joint
and actuator names, because fingers change the order of the right arm's joints.

This scene replaces the v1.3 wrist-roll meshes with v1.4
`left_wrist_roll_link10.stl` and `right_wrist_roll_link2.stl`, and uses their
matching mass, centre of mass, inertia tensor, and visual color. The source link
and palm-mount joint excerpts are
bundled in `omnihand_wrist_links.xml` for reproducible generation. These meshes
and link excerpts come from `/home/ubuntu/agibot_x2_urdf/X2_URDF-v1.4.0/`
and retain the X2 asset license above. The arm-only scene keeps its original
wrist geometry and dynamics.

Mounts come from the local Agibot v1.4.0 `X2-Ultra_omnihand.urdf`:

| Palm | Wrist-frame XYZ (m) | URDF RPY (rad) |
| --- | --- | --- |
| `L_palm` | `0 -0.001 -0.049` | `0 3.14 1.58` |
| `R_palm` | `0 0.001 -0.049` | `0 3.14 -1.58` |

These rotations are converted to quaternions using URDF fixed-axis conventions,
independently of the X2 MJCF compiler's Euler convention. The generator reads
these transforms from `omnihand_wrist_links.xml`, alongside the replacement
wrist data. The mount transforms originate from a newer X2 description than
the bundled v1.3.0 body. Rendered views and headless checks verify the configured
frames; an Omnihand Pro-specific physical adapter transform has not been supplied.

The hands preserve source actuator limits, gains, finger inertias, and all 14
coupling constraints. Palms use the supplied Omnihand Pro URDF inertias explicitly;
visual meshes contribute no mass. All hand contacts are disabled. This scene
adds finger motion, but still has no collision avoidance or object grasping.
The source hand joint limit margin is reduced from 0.1 rad to zero so the open
pose does not fight prematurely active joint limits. Right thumb roll/abduction
are mirrored; thumb MCP/PIP flexion has the same sign on both hands.
The wrist-to-palm transforms above are retained as an initial mount for Omnihand Pro;
the supplied hand-only package does not define an X2 adapter transform.

Rebuild the derived scene from bundled sources after changing the generator:

```bash
python scripts/simulation/build_x2_omnihand_assets.py
```

The generator also refreshes hand-center frames in the default MJCF/URDF and
writes `x2_omnihand_ik.urdf`. Centers come from the supplied mesh bounds and
mounting transforms, so regeneration preserves alignment between physics and IK.

Tests compare regenerated documents with the bundled output, so changing source
data or generator logic requires rebuilding these files. Inertia conversion
preserves URDF inertial-frame rotations before MuJoCo diagonalizes the tensor.

See [Omnihand Pro source and license metadata](../../omnihand_pro/README.md).
