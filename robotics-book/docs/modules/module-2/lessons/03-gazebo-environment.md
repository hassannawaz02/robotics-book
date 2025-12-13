---
title: "Gazebo Environment Building"
description: "Creating complex simulation environments with models, lighting, and physics"
---

# Gazebo Environment Building

## Overview

Creating realistic simulation environments is crucial for effective digital twin applications. This lesson covers the complete process of building sophisticated Gazebo environments with detailed models, realistic lighting, and accurate physics configurations that mirror real-world scenarios for humanoid robotics applications.

## Learning Objectives

- Design and create comprehensive SDF world files
- Build and integrate complex 3D models with proper physics properties
- Configure realistic lighting and atmospheric conditions
- Optimize environments for performance and accuracy
- Create humanoid robot-friendly environments with obstacles and interaction points

## World File Structure and Configuration

### Basic World File Template

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="digital_twin_environment">
    <!-- Physics Engine Configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom environment elements will go here -->

    <!-- Plugins for additional functionality -->
    <plugin name="world_plugin" filename="libWorldPlugin.so">
      <!-- Plugin-specific parameters -->
    </plugin>
  </world>
</sdf>
```

### Advanced Physics Configuration

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
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
```

## Creating Complex Indoor Environments

### Multi-Room Environment Example

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="multi_room_lab">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room 1: Office -->
    <model name="office_walls">
      <pose>0 0 0 0 0 0</pose>
      <!-- Wall 1 -->
      <link name="wall_1">
        <pose>5 0 2.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <!-- Wall 2 -->
      <link name="wall_2">
        <pose>-5 0 2.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <!-- Wall 3 -->
      <link name="wall_3">
        <pose>0 5 2.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <!-- Wall 4 with door opening -->
      <link name="wall_4">
        <pose>0 -3 2.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.2 4 5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 4 5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_4b">
        <pose>0 7 2.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.2 4 5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 4 5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Room 2: Laboratory -->
    <model name="lab_walls">
      <pose>10 0 0 0 0 0</pose>
      <!-- Similar wall structure offset by 10m in x direction -->
      <link name="wall_1_lab">
        <pose>5 0 2.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 10 5</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.9 1</ambient>
            <diffuse>0.7 0.7 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <!-- Additional walls for lab room -->
    </model>

    <!-- Indoor lighting -->
    <model name="ceiling_light_1">
      <pose>0 0 4.5 0 0 0</pose>
      <link name="light_link">
        <visual name="light_visual">
          <geometry>
            <cylinder><radius>0.1</radius><length>0.05</length></cylinder>
          </geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
          </material>
        </visual>
        <light name="light_1" type="point">
          <pose>0 0 0 0 0 0</pose>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
          <attenuation>
            <range>10</range>
            <constant>0.9</constant>
            <linear>0.01</linear>
            <quadratic>0.001</quadratic>
          </attenuation>
          <cast_shadows>false</cast_shadows>
        </light>
      </link>
    </model>

    <!-- Furniture and obstacles -->
    <include>
      <uri>model://table</uri>
      <pose>2 2 0 0 0 0</pose>
    </include>
    <include>
      <uri>model://chair</uri>
      <pose>2.5 1.5 0 0 0 1.57</pose>
    </include>
    <include>
      <uri>model://cylinder</uri>
      <pose>-3 -2 0.5 0 0 0</pose>
      <name>obstacle_1</name>
    </include>
  </world>
</sdf>
```

## Model Creation and Integration

### Custom Model SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <pose>0 0 0.75 0 0 0</pose> <!-- Initial pose with z=0.75 for standing height -->

    <!-- Links for each body part -->
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
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
          <box><size>0.3 0.3 0.3</size></box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box><size>0.3 0.3 0.3</size></box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.8 1</ambient>
          <diffuse>0.1 0.1 0.8 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Torso -->
    <link name="torso">
      <pose>0 0 0.3 0 0 0</pose>
      <inertial>
        <mass>8.0</mass>
        <inertia>
          <ixx>0.2</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.2</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box><size>0.25 0.25 0.5</size></box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box><size>0.25 0.25 0.5</size></box>
        </geometry>
        <material>
          <ambient>0.8 0.1 0.1 1</ambient>
          <diffuse>0.8 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Joint connecting base to torso -->
    <joint name="base_torso_joint" type="revolute">
      <parent>base_link</parent>
      <child>torso</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>100.0</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
    </joint>

    <!-- Additional links and joints for full humanoid model -->
    <!-- ... more links for arms, legs, head ... -->

    <!-- Sensors -->
    <include>
      <uri>model://depth_camera</uri>
      <pose>0.15 0 0.1 0 0 0</pose> <!-- On head/upper body -->
    </include>

    <include>
      <uri>model://imu_sensor</uri>
      <pose>0 0 0.1 0 0 0</pose> <!-- On torso -->
    </include>

    <!-- Plugins -->
    <plugin name="humanoid_controller" filename="libHumanoidController.so">
      <!-- Controller parameters -->
    </plugin>
  </model>
</sdf>
```

## Environment Optimization Techniques

### Level of Detail (LOD) for Models

```xml
<model name="detailed_environment_object">
  <!-- High detail visual for close viewing -->
  <link name="visual_lod_0">
    <visual name="high_detail">
      <geometry>
        <mesh><uri>model://object/meshes/high_detail.dae</uri></mesh>
      </geometry>
    </visual>
  </link>

  <!-- Simplified visual for distance viewing -->
  <link name="visual_lod_1">
    <visual name="low_detail">
      <geometry>
        <mesh><uri>model://object/meshes/low_detail.dae</uri></mesh>
      </geometry>
    </visual>
  </link>

  <!-- Simple collision geometry -->
  <link name="collision">
    <collision name="simple_collision">
      <geometry>
        <box><size>1 1 1</size></box>
      </geometry>
    </collision>
  </link>
</model>
```

### Performance Optimization Parameters

```xml
<!-- In world file, add performance optimization settings -->
<world name="optimized_environment">
  <!-- Physics optimization -->
  <physics type="ode">
    <max_step_size>0.002</max_step_size> <!-- Increase slightly for performance -->
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>500.0</real_time_update_rate> <!-- Reduced for performance -->
    <gravity>0 0 -9.8</gravity>
    <ode>
      <solver>
        <type>quick</type>
        <iters>50</iters> <!-- Reduced iterations for performance -->
        <sor>1.3</sor>
      </solver>
    </ode>
  </physics>

  <!-- Visual optimization -->
  <scene>
    <ambient>0.3 0.3 0.3 1</ambient>
    <background>0.6 0.7 0.9 1</background>
    <shadows>false</shadows> <!-- Disable shadows for performance -->
  </scene>
</world>
```

## Hands-On Lab: Create a Humanoid-Friendly Environment

### Step 1: Create Basic World File

Create `humanoid_lab.world`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_robotics_lab">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Indoor environment with obstacles -->
    <model name="lab_room">
      <pose>0 0 0 0 0 0</pose>
      <!-- Perimeter walls -->
      <link name="north_wall">
        <pose>0 5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="south_wall">
        <pose>0 -5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="east_wall">
        <pose>5 0 1.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="west_wall">
        <pose>-5 0 1.5 0 0 1.5707</pose>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacles for navigation testing -->
    <model name="obstacle_1">
      <pose>-2 2 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.1</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.4 0.2 1</ambient>
            <diffuse>0.8 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>2 -2 0.3 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>2.0</mass>
          <inertia>
            <ixx>0.2</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.2</iyy>
            <iyz>0</iyz>
            <izz>0.2</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box><size>0.5 0.5 0.6</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.5 0.5 0.6</size></box>
          </geometry>
          <material>
            <ambient>0.2 0.6 0.8 1</ambient>
            <diffuse>0.2 0.6 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Navigation markers -->
    <model name="waypoint_1">
      <pose>0 0 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.1</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Lighting fixtures -->
    <model name="ceiling_light">
      <pose>0 0 2.8 0 0 0</pose>
      <link name="light_link">
        <visual name="light_visual">
          <geometry>
            <box><size>0.8 0.8 0.1</size></box>
          </geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
          </material>
        </visual>
        <light name="main_light" type="point">
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
          <attenuation>
            <range>15</range>
            <constant>0.2</constant>
            <linear>0.01</linear>
            <quadratic>0.001</quadratic>
          </attenuation>
          <cast_shadows>false</cast_shadows>
        </light>
      </link>
    </model>
  </world>
</sdf>
```

### Step 2: Launch the Environment

```bash
# Launch Gazebo with the custom world
gazebo humanoid_lab.world

# Or use gz command if using newer version
gz sim -r humanoid_lab.world
```

## Troubleshooting Common Issues

### 1. Model Spawning Problems
- Check pose coordinates are within world bounds
- Verify model files exist in Gazebo model path
- Ensure proper permissions on model files

### 2. Physics Instability
- Reduce max_step_size for better stability
- Increase solver iterations
- Check for intersecting collision geometries

### 3. Performance Issues
- Simplify collision geometries
- Reduce model complexity at a distance
- Optimize lighting calculations

## Next Steps

In the next lesson, we'll explore the integration between ROS 2, Gazebo, and Unity to create a complete digital twin system where all components work together seamlessly.