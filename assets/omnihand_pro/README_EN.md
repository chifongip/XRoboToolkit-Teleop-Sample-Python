# OmniHandPro ROS 2 Package

This is a ROS 2 package for **OmniHandPro**. It includes the URDF files for both the left and right hands, along with visualization support in RViz. The recommended ROS 2 distribution for using this package is **Humble**.

## Installation

### 1. Clone the Repository
Create a new `src` directory, navigate into it, and clone the repository:

```bash
mkdir -p ~/omnihand_pro_ws/src
cd ~/omnihand_pro_ws/src
git clone <repository_url>
cd ..
```

### 2. Build the Package
Build the workspace using:

```bash
colcon build
```

### 3. Source the Setup File
Before using the package, source the workspace setup file:

```bash
source install/setup.bash
```

## Generate OmniHandPro URDF File

To generate the URDF description from the Xacro files:

```bash
cd ~/omnihand_pro_ws/src/omnihand_pro_description/assets/urdf/xacro
xacro omnihand_pro_right.xacro > omnihand_pro_right.urdf
xacro omnihand_pro_left.xacro > omnihand_pro_left.urdf
```

## Visualization in RViz

### Right Hand
```bash
ros2 launch omnihand_pro_description omnihand_pro_description_launch.py
```

### Left Hand
```bash
ros2 launch omnihand_pro_description omnihand_pro_description_launch.py hand_type:=left
```

## High Precision Collision URDF
This package additionally provides a collision-optimized URDF model, generated using convex optimization over original hand mesh geometry.
Compared to standard primitive collision approximations, this version produces more accurate physical interaction.

The convex collision URDF is located at:
omnihand_pro_description/assets/urdf_mesh_col/

### Right Hand With Accurate Collision
```bash
ros2 launch omnihand_pro_description omnihand_pro_description_col.launch.py
```

### Left Hand With Accurate Collision
```bash
ros2 launch omnihand_pro_description omnihand_pro_description_col.launch.py hand_type:=left
```

> **Note on Joint Coupling Accuracy**\
> URDF only supports **linear coupling** between passive and active
> joints.\
> However, OmniHandPro's mechanical structure uses **nonlinear
> coupling** for most passive joints, making it impossible for the URDF
> model to fully match the real hardware behavior.\
> If you need to simulate the **true nonlinear coupling**, please refer
> to: 
> - The **MuJoCo MJCF model**, which contains accurate nonlinear mappings
> - The **OmniHandPro SDK**, which provides real coupling functions

------------------------------------------------------------------------

## MuJoCo MJCF Files

This package also provides **MuJoCo MJCF models** for simulating OmniHandPro in physics-based environments.  
The MJCF files are compatible with **MuJoCo 3.1.0 and later**.

Model files are located at:

```
omnihand_pro_description/assets/MJCF/
```

This directory includes:

- `scene.xml` – A complete scene file for quick preview  
- `omnihand_pro_left.xml` – Left-hand model  
- `omnihand_pro_right.xml` – Right-hand model  
- `meshes/` – STL mesh assets used by MJCF  

### Quick Preview

To quickly preview the OmniHandPro model:

1. Open MuJoCo’s `simulate` viewer  
2. Drag and drop `assets/MJCF/scene.xml` into the window  
3. The OmniHandPro model will automatically load and display

Example:

![alt text](mujoco_image.png)
