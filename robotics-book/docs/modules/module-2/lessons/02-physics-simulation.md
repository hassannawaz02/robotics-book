---
title: "Gazebo Physics Simulation"
description: "Gravity, collisions, joints, and physics properties in Gazebo"
---

# Gazebo Physics Simulation

## Overview

Gazebo's physics simulation engine provides realistic modeling of physical interactions in digital twin environments. This lesson covers gravity, collisions, joint dynamics, and other fundamental physics properties that enable accurate simulation of humanoid robots and their environments.

## Learning Objectives

- Configure Gazebo physics engines and parameters
- Implement gravity, collision detection, and joint constraints
- Set up realistic physical properties for humanoid robots
- Create custom physics models for specific applications

## Theoretical Foundations

### Physics Engine Architecture

Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default engine, good for general use
- **Bullet**: Better for complex collision scenarios
- **DART**: Advanced dynamics for articulated robots
- **SimBody**: High-fidelity biomechanics simulation

### Core Physics Concepts

1. **Gravity**: 3D vector defining gravitational acceleration (default: [0, 0, -9.8])
2. **Collision Detection**: Algorithms to detect when objects intersect
3. **Contact Response**: How objects react to collisions
4. **Joints**: Constraints between rigid bodies with specific degrees of freedom

## Physics Configuration in SDF

### World File Physics Configuration

```xml
<sdf version="1.7">
  <world name="digital_twin_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Your models and entities here -->
  </world>
</sdf>
```

### Model Physics Properties

```xml
<model name="humanoid_robot">
  <link name="base_link">
    <inertial>
      <mass>10.0</mass>
      <inertia>
        <ixx>0.4</ixx>
        <ixy>0.0</ixy>
        <ixz>0.0</ixz>
        <iyy>0.4</iyy>
        <iyz>0.0</iyz>
        <izz>0.4</izz>
      </inertia>
    </inertial>

    <collision name="collision">
      <geometry>
        <box>
          <size>0.5 0.5 0.5</size>
        </box>
      </geometry>
    </collision>

    <visual name="visual">
      <geometry>
        <box>
          <size>0.5 0.5 0.5</size>
        </box>
      </geometry>
    </visual>
  </link>
</model>
```

## Joint Configuration for Humanoid Robots

### Joint Types and Properties

```xml
<joint name="hip_joint" type="revolute">
  <parent>base_link</parent>
  <child>thigh_link</child>
  <axis>
    <xyz>0 0 1</xyz>
    <limit>
      <lower>-1.57</lower>
      <upper>1.57</upper>
      <effort>100.0</effort>
      <velocity>1.0</velocity>
    </limit>
    <dynamics>
      <damping>0.1</damping>
      <friction>0.0</friction>
    </dynamics>
  </axis>
</joint>
```

### Common Joint Types for Humanoid Robots:
- **Revolute**: Rotational joint with 1 DOF
- **Prismatic**: Linear joint with 1 DOF
- **Ball**: Ball-and-socket joint with 3 DOF
- **Fixed**: No movement (0 DOF)
- **Continuous**: Revolute joint without limits

## Hands-On Lab: Create a Simple Physics World

### Step 1: Create World File

Create `simple_physics.world`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_physics">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Box with physics properties -->
    <model name="falling_box">
      <pose>0 0 2 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 2: Launch the World

```bash
# Launch Gazebo with the world file
gzserver simple_physics.world

# Or run in GUI mode
gazebo simple_physics.world
```

## Advanced Physics Configuration

### Contact Parameters

```xml
<collision name="collision">
  <surface>
    <contact>
      <ode>
        <kp>1e+6</kp>  <!-- Spring stiffness -->
        <kd>100</kd>   <!-- Damping coefficient -->
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
    <friction>
      <ode>
        <mu>1.0</mu>    <!-- Static friction -->
        <mu2>1.0</mu2>  <!-- Secondary friction -->
        <fdir1>0 0 1</fdir1>  <!-- Friction direction -->
      </ode>
    </friction>
  </surface>
</collision>
```

### Custom Physics Plugins

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

class CustomPhysicsPlugin : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr _world, sdf::ElementPtr _sdf)
  {
    // Custom physics logic here
    gzdbg << "Custom physics plugin loaded\n";
  }
};

GZ_REGISTER_WORLD_PLUGIN(CustomPhysicsPlugin)
```

## Troubleshooting Common Physics Issues

### 1. Objects Falling Through Each Other
- Check collision geometry overlaps
- Adjust contact parameters (kp, kd values)
- Verify mass and inertia properties

### 2. Unstable Simulation
- Reduce max_step_size
- Increase solver iterations
- Check for degenerate inertia matrices

### 3. Joint Limit Issues
- Verify joint limits are properly defined
- Check for conflicting constraints
- Adjust dynamics parameters (damping, friction)

## Next Steps

In the next lesson, we'll explore sensor simulation including LiDAR, depth cameras, and IMUs that provide the sensory data for our digital twin system.