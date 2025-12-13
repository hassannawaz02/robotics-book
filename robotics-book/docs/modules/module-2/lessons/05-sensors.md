---
title: "Sensor Simulation"
description: "LiDAR, Depth Cameras, and IMU simulation in digital twin environments"
---

# Sensor Simulation

## Overview

Sensor simulation is critical for creating realistic digital twin environments that can provide the sensory data needed for robot perception, navigation, and control. This lesson covers the implementation of LiDAR, depth cameras, and IMUs in Gazebo and Unity, with focus on realistic noise models and data accuracy.

## Learning Objectives

- Configure realistic sensor models in Gazebo and Unity
- Implement LiDAR, depth camera, and IMU simulation
- Apply noise models and calibration parameters
- Validate sensor data accuracy against real-world specifications

## Theoretical Foundations

### Sensor Data Pipeline

```
Physical Environment → Sensor Physics → Raw Data → Noise Model → Processed Data
         ↓                   ↓               ↓            ↓              ↓
   Ground Truth      Ray Casting/Physics  Clean Data  Realistic Noise  Usable Data
```

### Sensor Characteristics

Each sensor type has specific characteristics that must be accurately modeled:

1. **LiDAR**: Range, resolution, field of view, scan pattern, accuracy
2. **Depth Camera**: Resolution, field of view, depth range, noise characteristics
3. **IMU**: Accelerometer/gyroscope/magnetometer noise, bias, drift, update rate

## LiDAR Simulation in Gazebo

### SDF Configuration

```xml
<sensor name="lidar_3d" type="ray">
  <pose>0.0 0.0 0.2 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>1081</samples>
        <resolution>1</resolution>
        <min_angle>-2.35619</min_angle>
        <max_angle>2.35619</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>
        <max_angle>0.261799</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.08</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_3d_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>lidar_3d</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>lidar_3d_frame</frame_name>
  </plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### Noise Models for LiDAR

```xml
<sensor name="lidar_3d" type="ray">
  <!-- ... previous configuration ... -->
  <noise type="gaussian">
    <mean>0.0</mean>
    <stddev>0.01</stddev>
  </noise>
</sensor>
```

## Depth Camera Simulation in Gazebo

### RGB-D Camera Configuration

```xml
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
  <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <baseline>0.2</baseline>
    <alwaysOn>true</alwaysOn>
    <updateRate>10.0</updateRate>
    <cameraName>depth_camera</cameraName>
    <imageTopicName>/rgb/image_raw</imageTopicName>
    <depthImageTopicName>/depth/image_raw</depthImageTopicName>
    <pointCloudTopicName>/depth/points</pointCloudTopicName>
    <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
    <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
    <frameName>depth_camera_frame</frameName>
    <pointCloudCutoff>0.1</pointCloudCutoff>
    <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <CxPrime>0.0</CxPrime>
    <Cx>320.0</Cx>
    <Cy>240.0</Cy>
    <focalLength>320.0</focalLength>
    <hackBaseline>0.0</hackBaseline>
  </plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

## IMU Simulation in Gazebo

### IMU Sensor Configuration

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00174533</stddev> <!-- ~0.1 deg/s -->
          <bias_mean>0.000174533</bias_mean> <!-- ~0.01 deg/s -->
          <bias_stddev>1.74533e-05</bias_stddev> <!-- ~0.001 deg/s -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00174533</stddev>
          <bias_mean>0.000174533</bias_mean>
          <bias_stddev>1.74533e-05</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.00174533</stddev>
          <bias_mean>0.000174533</bias_mean>
          <bias_stddev>1.74533e-05</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev> <!-- 17 mg -->
          <bias_mean>0.1</bias_mean> <!-- 100 mg -->
          <bias_stddev>0.001</bias_stddev> <!-- 1 mg -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>imu</namespace>
      <remapping>~/out:=data</remapping>
    </ros>
    <frame_name>imu_frame</frame_name>
    <body_name>imu_link</body_name>
    <update_rate>100</update_rate>
    <gaussian_noise>0.00174533</gaussian_noise>
  </plugin>
</sensor>
```

## Hands-On Lab: Complete Sensor Setup

### Step 1: Create a Robot Model with Sensors

Create `sensor_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="sensor_robot">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.4" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.5 0.5" />
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <box size="0.5 0.5 0.5" />
      </geometry>
    </collision>
  </link>

  <!-- IMU Link -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link" />
    <child link="imu_link" />
    <origin xyz="0.1 0 0.2" rpy="0 0 0" />
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001" />
    </inertial>
  </link>

  <!-- LiDAR Link -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link" />
    <child link="lidar_link" />
    <origin xyz="0.2 0 0.3" rpy="0 0 0" />
  </joint>

  <link name="lidar_link">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001" />
    </inertial>
  </link>

  <!-- Depth Camera Link -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link" />
    <child link="camera_link" />
    <origin xyz="0.2 0 0.2" rpy="0 0 0" />
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.01" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001" />
    </inertial>
  </link>

  <!-- Gazebo plugins for sensors -->
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

  <gazebo reference="lidar_link">
    <sensor name="lidar_3d" type="ray">
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-1.570796</min_angle>
            <max_angle>1.570796</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_3d_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>lidar_3d</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_link</frame_name>
      </plugin>
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

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
      <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
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
</robot>
```

### Step 2: Launch File for Sensor Robot

Create `sensor_robot.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Get URDF file path
    urdf_path = os.path.join(
        get_package_share_directory('your_robot_description'),
        'urdf',
        'sensor_robot.urdf.xacro'
    )

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(urdf_path).read()
        }]
    )

    # Joint State Publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    # TF2 Static Transform Publisher for base_link to imu_link
    tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu_tf',
        arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'imu_link']
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        tf_publisher
    ])
```

## Unity Sensor Simulation

Unity doesn't natively support the same sensor simulation as Gazebo, but we can create custom sensors using raycasting and physics:

### LiDAR Simulation in Unity

```csharp
using UnityEngine;
using System.Collections.Generic;

public class UnityLidar : MonoBehaviour
{
    [Header("Lidar Configuration")]
    public int horizontalSamples = 360;
    public int verticalSamples = 16;
    public float horizontalFOV = 360f;
    public float verticalFOV = 30f;
    public float minRange = 0.1f;
    public float maxRange = 30f;
    public LayerMask detectionMask = -1;

    [Header("Noise Parameters")]
    public float noiseStdDev = 0.01f;

    private float[] ranges;
    private Vector3[] directions;

    void Start()
    {
        InitializeLidar();
    }

    void InitializeLidar()
    {
        int totalSamples = horizontalSamples * verticalSamples;
        ranges = new float[totalSamples];
        directions = new Vector3[totalSamples];

        int index = 0;
        for (int v = 0; v < verticalSamples; v++)
        {
            float vAngle = (v - verticalSamples / 2) * (verticalFOV / verticalSamples) * Mathf.Deg2Rad;
            for (int h = 0; h < horizontalSamples; h++)
            {
                float hAngle = h * (horizontalFOV / horizontalSamples) * Mathf.Deg2Rad;

                // Calculate direction vector
                Vector3 direction = new Vector3(
                    Mathf.Cos(vAngle) * Mathf.Sin(hAngle),
                    Mathf.Cos(vAngle) * Mathf.Cos(hAngle),
                    Mathf.Sin(vAngle)
                ).normalized;

                directions[index] = transform.TransformDirection(direction);
                ranges[index] = maxRange;
                index++;
            }
        }
    }

    void Update()
    {
        SimulateLidarScan();
    }

    void SimulateLidarScan()
    {
        int index = 0;
        for (int v = 0; v < verticalSamples; v++)
        {
            for (int h = 0; h < horizontalSamples; h++)
            {
                RaycastHit hit;
                if (Physics.Raycast(transform.position, directions[index], out hit, maxRange, detectionMask))
                {
                    float distance = hit.distance;
                    // Add noise to the measurement
                    distance += RandomGaussian() * noiseStdDev;
                    ranges[index] = Mathf.Clamp(distance, minRange, maxRange);
                }
                else
                {
                    ranges[index] = maxRange;
                }
                index++;
            }
        }
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian noise
        float u1 = Random.value;
        float u2 = Random.value;
        if (u1 < 0.00001f) u1 = 0.00001f;
        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }

    // Publish ranges data via ROS or other communication method
    public float[] GetRanges()
    {
        return ranges;
    }
}
```

## Sensor Data Validation

### Validation Checklist

1. **Range Validation**: Verify sensor readings are within expected min/max ranges
2. **Noise Characteristics**: Confirm noise follows expected statistical distribution
3. **Update Rate**: Ensure sensors publish at expected frequency
4. **Coordinate Frames**: Validate TF transforms between sensor frames
5. **Calibration**: Check sensor parameters match real-world specifications

### Testing Script

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
import numpy as np
from scipy import stats

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Subscriptions for each sensor
        self.lidar_sub = self.create_subscription(
            LaserScan, '/lidar_3d/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.camera_callback, 10)

        self.get_logger().info('Sensor validator node started')

    def lidar_callback(self, msg):
        # Validate LiDAR data
        ranges = np.array(msg.ranges)

        # Check for invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]
        if len(valid_ranges) == 0:
            self.get_logger().warn('All LiDAR ranges are invalid')
            return

        # Check range bounds
        if np.any(valid_ranges < msg.range_min) or np.any(valid_ranges > msg.range_max):
            self.get_logger().warn('LiDAR ranges outside bounds')

        # Validate update rate
        expected_rate = 1.0 / msg.time_increment
        self.get_logger().info(f'LiDAR rate: {expected_rate:.2f} Hz')

    def imu_callback(self, msg):
        # Validate IMU data
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # Check for NaN or infinity
        if not (np.all(np.isfinite(acc)) and np.all(np.isfinite(gyro))):
            self.get_logger().warn('IMU data contains NaN or infinity')

        # Check magnitude bounds (reasonable for Earth gravity)
        if np.linalg.norm(acc) > 20.0:  # 2x Earth gravity
            self.get_logger().warn('IMU acceleration magnitude too high')

    def camera_callback(self, msg):
        # Validate camera data
        width = msg.width
        height = msg.height
        step = msg.step

        # Check image dimensions
        expected_size = width * height * (step // width)
        if len(msg.data) != expected_size:
            self.get_logger().warn('Camera image size mismatch')

def main(args=None):
    rclpy.init(args=args)
    validator = SensorValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Issues

### 1. LiDAR Point Cloud Sparsity
- Increase horizontal/vertical samples in SDF
- Check update rate settings
- Verify collision geometry on objects

### 2. IMU Drift and Noise
- Adjust noise parameters in SDF
- Verify update rate (higher for less drift)
- Check for integration errors in processing

### 3. Depth Camera Artifacts
- Verify camera clipping planes
- Check for rendering artifacts
- Validate point cloud generation parameters

## Next Steps

In the next lesson, we'll explore Unity's high-fidelity rendering capabilities for creating realistic human-robot interaction environments that complement our Gazebo physics and sensor simulation.