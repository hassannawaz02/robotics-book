---
title: "ROS 2 ↔ Gazebo ↔ Unity Integration"
description: "Connecting ROS 2, Gazebo, and Unity for complete digital twin systems"
---

# ROS 2 ↔ Gazebo ↔ Unity Integration

## Overview

Creating a complete digital twin system requires seamless integration between ROS 2 for robotics middleware, Gazebo for physics simulation, and Unity for high-fidelity visualization. This lesson covers the complete integration pipeline that enables bidirectional communication between all three components.

## Learning Objectives

- Configure ROS 2 bridges for Gazebo and Unity communication
- Implement bidirectional data flow between simulation environments
- Create unified control interfaces for digital twin systems
- Validate synchronization between physics and visualization

## The Digital Twin Integration Architecture

### System Architecture Overview

```
Physical Robot ──────────────────────────────────────────────→ Digital Twin System
       │                                                           │
       │ (Optional)                                                │
       ↓                                                           ↓
Real Sensors ─→ [ROS 2 Middleware] ←─ Simulated Sensors ←── Unity Visualization
       │              │                         │                    │
       │              │                         │                    │
       │              ↓                         ↓                    │
       ──────────→ Gazebo Physics ←─────────── Unity-ROS Bridge ←────
                  │      │
                  │      └─── Robot State (TF, Joint States)
                  │
                  └─── Sensor Data (LIDAR, IMU, Cameras, etc.)
```

### Core Integration Components

1. **ROS 2 Middleware**: Provides message passing and service communication
2. **Gazebo-ROS Bridge**: Connects physics simulation to ROS 2 topics
3. **Unity-ROS Bridge**: Links Unity visualization to ROS 2 ecosystem
4. **TF2 System**: Maintains coordinate frame relationships
5. **Robot State Publisher**: Synchronizes robot joint states

## ROS 2 Bridge Configuration

### Gazebo-ROS 2 Bridge Setup

```xml
<!-- In robot model SDF -->
<plugin name="ros_gz_bridge" filename="libros_gz_bridge.so">
  <ros>
    <namespace>/robot1</namespace>
    <remapping>cmd_vel:=cmd_vel</remapping>
    <remapping>odom:=odom</remapping>
    <remapping>imu:=imu/data</remapping>
    <remapping>scan:=scan</remapping>
  </ros>
  <parameters>
    <!-- Bridge configuration parameters -->
    <config_file>/path/to/bridge_config.yaml</config_file>
  </parameters>
</plugin>
```

### Bridge Configuration File

Create `bridge_config.yaml`:

```yaml
# Bridge configuration for Gazebo-ROS integration
- ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Velocity"
  ros_topic_name: "/cmd_vel"
  gz_topic_name: "/model/robot/cmd_vel"
  direction: "BOTH"

- ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  ros_topic_name: "/scan"
  gz_topic_name: "/lidar_3d/scan"
  direction: "GZ_TO_ROS"

- ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  ros_topic_name: "/imu/data"
  gz_topic_name: "/imu_3d"
  direction: "GZ_TO_ROS"

- ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  ros_topic_name: "/odom"
  gz_topic_name: "/model/robot/odometry"
  direction: "GZ_TO_ROS"
```

## Unity-ROS Bridge Integration

### Unity ROS-TCP Connector Setup

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using RosMessageTypes.Sensor;

public class UnityRobotController : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Configuration")]
    public string robotNamespace = "/robot1";
    public Transform robotTransform;

    private ROSConnection ros;
    private string cmdVelTopic;
    private string odomTopic;
    private string imuTopic;

    void Start()
    {
        SetupROSConnection();
        SetupTopicNames();
        SubscribeToTopics();
    }

    void SetupROSConnection()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIPAddress, rosPort);
    }

    void SetupTopicNames()
    {
        cmdVelTopic = robotNamespace + "/cmd_vel";
        odomTopic = robotNamespace + "/odom";
        imuTopic = robotNamespace + "/imu/data";
    }

    void SubscribeToTopics()
    {
        ros.Subscribe<OdometryMsg>(odomTopic, OnOdometryReceived);
        ros.Subscribe<ImuMsg>(imuTopic, OnImuReceived);
    }

    void OnOdometryReceived(OdometryMsg odom)
    {
        // Update Unity robot position based on ROS odometry
        Vector3 position = new Vector3((float)odom.pose.pose.position.x,
                                      (float)odom.pose.pose.position.z,  // Unity Y is ROS Z
                                      (float)odom.pose.pose.position.y); // Unity Z is ROS Y

        Quaternion rotation = new Quaternion((float)odom.pose.pose.orientation.x,
                                            (float)odom.pose.pose.orientation.z,
                                            (float)odom.pose.pose.orientation.y,
                                            (float)odom.pose.pose.orientation.w);

        robotTransform.position = position;
        robotTransform.rotation = rotation;
    }

    void OnImuReceived(ImuMsg imu)
    {
        // Process IMU data for Unity visualization
        Vector3 angularVelocity = new Vector3((float)imu.angular_velocity.x,
                                             (float)imu.angular_velocity.z,
                                             (float)imu.angular_velocity.y);

        Vector3 linearAcceleration = new Vector3((float)imu.linear_acceleration.x,
                                                (float)imu.linear_acceleration.z,
                                                (float)imu.linear_acceleration.y);

        // Use data for Unity physics or visualization
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        // Send velocity command to ROS
        TwistMsg cmd = new TwistMsg();
        cmd.linear = new Vector3Msg(linearX, 0, 0);
        cmd.angular = new Vector3Msg(0, 0, angularZ);

        ros.Publish(cmdVelTopic, cmd);
    }
}
```

## Launch File Configuration

### Complete Digital Twin Launch File

Create `digital_twin.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default='false')
    server_mode = LaunchConfiguration('server_mode', default='false')
    gui = LaunchConfiguration('gui', default='true')

    # Package directories
    gazebo_ros_package_dir = get_package_share_directory('gazebo_ros')
    robot_description_package_dir = get_package_share_directory('your_robot_description')

    # World file
    world_file = PathJoinSubstitution([
        get_package_share_directory('your_worlds_package'),
        'worlds',
        'humanoid_lab.world'
    ])

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_package_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'headless': headless,
            'server_mode': server_mode,
            'gui': gui,
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
            'robot_description': open(
                os.path.join(robot_description_package_dir, 'urdf', 'humanoid_robot.urdf.xacro')
            ).read()
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Gazebo-ROS bridge
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        parameters=[{
            'config_file': os.path.join(robot_description_package_dir, 'config', 'bridge_config.yaml'),
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    # TF2 static transform publisher for Unity coordinate system
    unity_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='unity_coordinate_transform',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'unity_world']
    )

    # Unity bridge node (if using custom Unity bridge)
    unity_bridge = Node(
        package='unity_ros_bridge',
        executable='unity_bridge_node',
        name='unity_bridge',
        parameters=[{
            'use_sim_time': use_sim_time,
            'unity_ip': '127.0.0.1',
            'unity_port': 5005
        }],
        output='screen'
    )

    return LaunchDescription([
        # Launch Gazebo
        gazebo,

        # Launch robot state publisher
        robot_state_publisher,

        # Launch joint state publisher
        joint_state_publisher,

        # Launch ROS-Gazebo bridge
        ros_gz_bridge,

        # Launch Unity bridge
        unity_bridge,

        # Launch TF publisher
        unity_tf_publisher,
    ])
```

## Synchronization and Validation

### Time Synchronization

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import time

class TimeSynchronizer(Node):
    def __init__(self):
        super().__init__('time_synchronizer')

        # Publishers for synchronized time
        self.time_pub = self.create_publisher(Time, '/sync_time', 10)

        # Timer for publishing synchronized time
        self.timer = self.create_timer(0.01, self.publish_time)  # 100Hz

        self.get_logger().info('Time synchronizer node started')

    def publish_time(self):
        # Get current time
        current_time = self.get_clock().now().to_msg()

        # Publish to synchronize all systems
        self.time_pub.publish(current_time)

def main(args=None):
    rclpy.init(args=args)
    synchronizer = TimeSynchronizer()

    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        pass
    finally:
        synchronizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Validation Tools

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import numpy as np

class IntegrationValidator(Node):
    def __init__(self):
        super().__init__('integration_validator')

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for validation
        self.timer = self.create_timer(1.0, self.validate_sync)

        self.get_logger().info('Integration validator node started')

    def validate_sync(self):
        try:
            # Check transform between Gazebo and Unity coordinate frames
            transform = self.tf_buffer.lookup_transform(
                'gazebo_world', 'unity_world', rclpy.time.Time()
            )

            # Calculate transform error
            translation_error = np.sqrt(
                transform.transform.translation.x**2 +
                transform.transform.translation.y**2 +
                transform.transform.translation.z**2
            )

            if translation_error > 0.01:  # 1cm threshold
                self.get_logger().warn(
                    f'Coordinate frame drift detected: {translation_error:.3f}m'
                )
            else:
                self.get_logger().info(
                    f'Systems synchronized: {translation_error:.3f}m error'
                )

        except Exception as e:
            self.get_logger().error(f'Transform lookup failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    validator = IntegrationValidator()

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

## Troubleshooting Integration Issues

### 1. Topic Connection Problems
- Verify ROS 2 network configuration
- Check topic names and message types
- Confirm bridge configuration files
- Validate IP addresses and ports

### 2. Timing Synchronization
- Ensure all systems use the same clock source
- Check for time drift between environments
- Verify simulation time settings

### 3. Coordinate Frame Issues
- Validate TF tree consistency
- Check frame naming conventions
- Verify transform orientations

## Next Steps

In the next lesson, we'll create a complete hands-on lab where we build a simple world, spawn a humanoid robot, and test the full integration pipeline.