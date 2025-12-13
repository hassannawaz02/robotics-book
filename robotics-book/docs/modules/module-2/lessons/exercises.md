---
title: "Lab: Build a Simple World and Spawn a Humanoid Robot"
description: "Complete hands-on lab to build a world and spawn a humanoid robot"
---

# Lab: Build a Simple World and Spawn a Humanoid Robot

## Overview

This hands-on lab combines all the concepts learned in previous lessons to create a complete digital twin environment. You will build a simple world from scratch, create a humanoid robot model, spawn it in the environment, and test the integration with ROS 2.

## Learning Objectives

- Design and implement a complete simulation environment
- Create a humanoid robot model with proper physics and sensors
- Spawn and control the robot in the simulated world
- Validate the complete digital twin integration

## Prerequisites

Before starting this lab, ensure you have:

- ROS 2 Humble Hawksbill installed
- Gazebo Garden or compatible version
- Unity 2022.3 LTS (for visualization components)
- Basic understanding of URDF/SDF
- Completed previous lessons in this module

## Step 1: Create the Environment World

### 1.1 Create the World File

Create `simple_world.world` in your worlds package:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_humanoid_world">
    <!-- Physics Engine Configuration -->
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

    <!-- Environment lighting -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Simple room environment -->
    <model name="room_walls">
      <pose>0 0 0 0 0 0</pose>
      <!-- North wall -->
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
      <!-- South wall -->
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
      <!-- East wall -->
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
      <!-- West wall -->
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

    <!-- Obstacles for navigation -->
    <model name="obstacle_1">
      <pose>-2 2 0.5 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.4</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.4</iyy>
            <iyz>0</iyz>
            <izz>0.4</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.4</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.4</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>3 -1 0.3 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>3.0</mass>
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
            <box><size>0.6 0.6 0.6</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.6 0.6 0.6</size></box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.6 1</ambient>
            <diffuse>0.2 0.8 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Navigation target -->
    <model name="target">
      <pose>4 4 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>0.2</length>
            </cylinder>
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

### 1.2 Set Up the World Directory Structure

Create the proper directory structure:

```bash
mkdir -p ~/ros2_ws/src/my_worlds_package/models
mkdir -p ~/ros2_ws/src/my_worlds_package/worlds
mkdir -p ~/ros2_ws/src/my_worlds_package/launch
```

Move your world file to the worlds directory and create a package.xml:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_worlds_package</name>
  <version>0.1.0</version>
  <description>Worlds for digital twin simulation</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>gazebo_ros_pkgs</exec_depend>
  <exec_depend>gazebo_ros</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Step 2: Create the Humanoid Robot Model

### 2.1 Create the URDF Model

Create `humanoid_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.4" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.3 0.3 0.3" />
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.3 0.3 0.3" />
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link" />
    <child link="torso" />
    <origin xyz="0 0 0.3" rpy="0 0 0" />
  </joint>

  <link name="torso">
    <inertial>
      <mass value="8.0" />
      <origin xyz="0 0 0.25" />
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.1" />
    </inertial>

    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0" />
      <geometry>
        <box size="0.25 0.25 0.5" />
      </geometry>
      <material name="red">
        <color rgba="0.8 0.1 0.1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0" />
      <geometry>
        <box size="0.25 0.25 0.5" />
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="fixed">
    <parent link="torso" />
    <child link="head" />
    <origin xyz="0 0 0.5" rpy="0 0 0" />
  </joint>

  <link name="head">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.15" />
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <sphere radius="0.15" />
      </geometry>
    </collision>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso" />
    <child link="left_upper_arm" />
    <origin xyz="0.15 0.15 0.25" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.15" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
      <material name="green">
        <color rgba="0.1 0.8 0.1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm" />
    <child link="left_lower_arm" />
    <origin xyz="0 0 -0.3" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="0.5" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0025" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.04" length="0.2" />
      </geometry>
      <material name="green">
        <color rgba="0.1 0.8 0.1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.04" length="0.2" />
      </geometry>
    </collision>
  </link>

  <!-- Right Arm (similar to left, mirrored) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso" />
    <child link="right_upper_arm" />
    <origin xyz="0.15 -0.15 0.25" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.15" />
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
      <material name="green">
        <color rgba="0.1 0.8 0.1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm" />
    <child link="right_lower_arm" />
    <origin xyz="0 0 -0.3" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="0.5" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0025" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.04" length="0.2" />
      </geometry>
      <material name="green">
        <color rgba="0.1 0.8 0.1 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.04" length="0.2" />
      </geometry>
    </collision>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link" />
    <child link="left_upper_leg" />
    <origin xyz="-0.1 0.1 -0.15" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="left_upper_leg">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.06" length="0.4" />
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.06" length="0.4" />
      </geometry>
    </collision>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg" />
    <child link="left_lower_leg" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="left_lower_leg">
    <inertial>
      <mass value="1.5" />
      <origin xyz="0 0 -0.15" />
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.0075" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
    </collision>
  </link>

  <!-- Right Leg (similar to left, mirrored) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link" />
    <child link="right_upper_leg" />
    <origin xyz="-0.1 -0.1 -0.15" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="right_upper_leg">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 -0.2" />
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.06" length="0.4" />
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.06" length="0.4" />
      </geometry>
    </collision>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg" />
    <child link="right_lower_leg" />
    <origin xyz="0 0 -0.4" rpy="0 0 0" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0" />
  </joint>

  <link name="right_lower_leg">
    <inertial>
      <mass value="1.5" />
      <origin xyz="0 0 -0.15" />
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.0075" />
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1" />
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0" />
      <geometry>
        <cylinder radius="0.05" length="0.3" />
      </geometry>
    </collision>
  </link>

  <!-- Sensors -->
  <!-- Depth camera on head -->
  <joint name="camera_joint" type="fixed">
    <parent link="head" />
    <child link="camera_link" />
    <origin xyz="0.1 0 0" rpy="0 0 0" />
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001" />
    </inertial>
  </link>

  <!-- IMU in torso -->
  <joint name="imu_joint" type="fixed">
    <parent link="torso" />
    <child link="imu_link" />
    <origin xyz="0 0 0.1" rpy="0 0 0" />
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001" />
    </inertial>
  </link>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="camera_link">
    <sensor name="depth_camera" type="depth">
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
        <cameraName>depth_camera</cameraName>
        <imageTopicName>/camera/rgb/image_raw</imageTopicName>
        <depthImageTopicName>/camera/depth/image_raw</depthImageTopicName>
        <pointCloudTopicName>/camera/depth/points</pointCloudTopicName>
        <cameraInfoTopicName>/camera/rgb/camera_info</cameraInfoTopicName>
        <depthImageCameraInfoTopicName>/camera/depth/camera_info</depthImageCameraInfoTopicName>
        <frameName>camera_link</frameName>
      </plugin>
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.00174533</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.00174533</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.00174533</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>
  </gazebo>

  <!-- Gazebo plugins for control -->
  <gazebo>
    <plugin name="humanoid_controller" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>odom</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>world</frameName>
    </plugin>
  </gazebo>
</robot>
```

### 2.2 Create the Robot Description Package

Create `robot_description/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_robot_description</name>
  <version>0.1.0</version>
  <description>Humanoid robot description for digital twin</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>xacro</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create `robot_description/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(humanoid_robot_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  meshes
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Step 3: Create Launch Files and Integration

### 3.1 Create Launch File

Create `robot_description/launch/humanoid_spawn.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')
    world_file = LaunchConfiguration('world', default='simple_world.world')

    # Package directories
    robot_description_dir = get_package_share_directory('humanoid_robot_description')
    worlds_package_dir = get_package_share_directory('my_worlds_package')
    gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')

    # Robot description
    robot_description_path = os.path.join(
        robot_description_dir, 'urdf', 'humanoid_robot.urdf.xacro'
    )

    # World file path
    world_path = PathJoinSubstitution([
        FindPackageShare('my_worlds_package'),
        'worlds',
        world_file
    ])

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_package_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'gui': 'true',
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(robot_description_path).read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_name,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.75'  # Height to place robot standing
        ],
        output='screen'
    )

    return LaunchDescription([
        # Launch Gazebo
        gazebo,

        # Launch robot state publisher
        robot_state_publisher,

        # Launch joint state publisher
        joint_state_publisher,

        # Spawn the robot
        spawn_entity,
    ])
```

### 3.2 Create Control Launch File

Create `robot_description/launch/humanoid_control.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Robot controller node
    robot_controller = Node(
        package='humanoid_robot_control',
        executable='robot_controller',
        name='robot_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Navigation stack (if available)
    navigation = Node(
        package='nav2_bringup',
        executable='nav2_bringup',
        name='navigation',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(
            get_package_share_directory('humanoid_robot_description'),
            'rviz',
            'humanoid_config.rviz'
        )],
        output='screen'
    )

    return LaunchDescription([
        robot_controller,
        navigation,
        rviz
    ])
```

## Step 4: Execute the Lab

### 4.1 Build and Source Your Workspace

```bash
cd ~/ros2_ws
colcon build --packages-select my_worlds_package humanoid_robot_description
source install/setup.bash
```

### 4.2 Launch the World and Robot

```bash
# Launch the world with the humanoid robot
ros2 launch humanoid_robot_description humanoid_spawn.launch.py
```

### 4.3 Test Robot Control

In a new terminal, test basic movement:

```bash
# Send a velocity command to move the robot forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' --once

# Check sensor topics
ros2 topic list | grep -E "(scan|imu|camera)"
```

### 4.4 Verify Integration

Check that all systems are working:

```bash
# Check TF tree
ros2 run tf2_tools view_frames

# Monitor sensor data
ros2 topic echo /camera/rgb/image_raw --field data --field height --field width

# Monitor IMU data
ros2 topic echo /imu/data --field orientation --field angular_velocity
```

## Step 5: Advanced Integration with Unity (Optional)

If you have Unity set up with the ROS-TCP connector:

### 5.1 Create Unity Scene

1. Create a new Unity scene
2. Import the humanoid robot model (or create a simple representation)
3. Add the Unity-ROS bridge components
4. Configure the bridge to connect to the same ROS network

### 5.2 Synchronize Data

Ensure that Unity visualizes the same robot state that's simulated in Gazebo by subscribing to the same topics and publishing to the same control topics.

## Troubleshooting

### Common Issues:

1. **Robot not spawning**: Check that the URDF is valid and the spawn topic is correct
2. **Sensors not publishing**: Verify Gazebo plugins are properly configured
3. **TF issues**: Ensure robot_state_publisher is running and joint states are published
4. **Control problems**: Check topic names and message types match expectations

### Validation Commands:

```bash
# Check all topics
ros2 topic list

# Check robot state
ros2 run tf2_tools view_frames

# Monitor sensor data
ros2 topic echo /scan --field ranges --field angle_min --field angle_max
```

## Lab Completion Checklist

- [ ] World file created with room and obstacles
- [ ] Humanoid robot model created with proper kinematics
- [ ] Robot successfully spawned in Gazebo
- [ ] Sensors (camera, IMU) publishing data
- [ ] TF tree showing proper robot structure
- [ ] Robot responds to control commands
- [ ] Integration validated with ROS 2 tools

## Next Steps

In the next lesson, we'll explore debugging techniques for common issues in digital twin systems.