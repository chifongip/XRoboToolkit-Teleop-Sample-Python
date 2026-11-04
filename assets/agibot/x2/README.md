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
- `scene.xml` adds lighting, a floor, and non-colliding wrist target markers.
- The home pose uses shoulder pitch -0.3, shoulder roll +0.2/-0.2, elbows -1.0,
  and other joints zero (radians). Position servos support gravity-loaded arms;
  small steady-state tracking errors are expected.

This demo simulates arm dynamics, without self-contact or collision avoidance.
The upstream collision meshes overlap at multiple joints even in neutral posture;
they must be refined before enabling self-contact. Fingers, grasping, locomotion,
and hardware control are not included.
