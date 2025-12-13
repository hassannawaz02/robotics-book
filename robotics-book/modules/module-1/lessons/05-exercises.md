---
title: Module 1 Exercises - ROS 2 Fundamentals
sidebar_position: 5
---

# Module 1 Exercises - ROS 2 Fundamentals

## Overview

This lesson contains practical exercises to reinforce your understanding of ROS 2 fundamentals: nodes, topics, services, and URDF. These exercises build on the concepts covered in previous lessons and provide hands-on experience with the core components of the robotic nervous system.

## Exercise 1: Create a Custom Publisher Node

### Objective
Create a ROS 2 publisher node that publishes temperature sensor data.

### Requirements
- Create a node named `temperature_publisher`
- Publish temperature values to the topic `temperature_data`
- Use the `std_msgs.msg.Float32` message type
- Publish data every 2 seconds
- Include random variation in the temperature readings (simulate sensor noise)

### Solution Template
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class TemperaturePublisher(Node):
    def __init__(self):
        super().__init__('temperature_publisher')
        # TODO: Create publisher
        # TODO: Create timer
        # TODO: Initialize temperature base value

    def publish_temperature(self):
        # TODO: Generate random temperature with noise
        # TODO: Publish the temperature message
        pass

def main(args=None):
    # TODO: Initialize ROS 2, create node, and spin
    pass

if __name__ == '__main__':
    main()
```

### Solution
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class TemperaturePublisher(Node):
    def __init__(self):
        super().__init__('temperature_publisher')
        self.publisher_ = self.create_publisher(Float32, 'temperature_data', 10)
        self.timer = self.create_timer(2.0, self.publish_temperature)
        self.base_temp = 20.0  # Base temperature in Celsius

    def publish_temperature(self):
        msg = Float32()
        # Add random noise between -2 and 2 degrees
        msg.data = self.base_temp + random.uniform(-2.0, 2.0)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing temperature: {msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = TemperaturePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down temperature publisher...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 2: Create a Subscriber Node with Processing

### Objective
Create a ROS 2 subscriber node that receives temperature data and performs basic analysis.

### Requirements
- Subscribe to the `temperature_data` topic
- Calculate and log statistics (min, max, average)
- Alert when temperature goes above 25°C
- Store the last 10 readings for statistical analysis

### Solution Template
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from collections import deque

class TemperatureAnalyzer(Node):
    def __init__(self):
        super().__init__('temperature_analyzer')
        # TODO: Create subscription
        # TODO: Initialize data storage
        # TODO: Initialize statistics tracking

    def temperature_callback(self, msg):
        # TODO: Store the reading
        # TODO: Update statistics
        # TODO: Check for alerts
        # TODO: Log statistics periodically
        pass

def main(args=None):
    # TODO: Initialize ROS 2, create node, and spin
    pass

if __name__ == '__main__':
    main()
```

### Solution
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from collections import deque

class TemperatureAnalyzer(Node):
    def __init__(self):
        super().__init__('temperature_analyzer')
        self.subscription = self.create_subscription(
            Float32,
            'temperature_data',
            self.temperature_callback,
            10)

        self.readings = deque(maxlen=10)  # Store last 10 readings
        self.total_readings = 0
        self.high_temp_alert_count = 0

    def temperature_callback(self, msg):
        temp = msg.data
        self.readings.append(temp)
        self.total_readings += 1

        # Check for high temperature
        if temp > 25.0:
            self.high_temp_alert_count += 1
            self.get_logger().warn(f'HIGH TEMPERATURE ALERT: {temp:.2f}°C')

        # Log statistics every 5 readings
        if self.total_readings % 5 == 0:
            if len(self.readings) > 0:
                min_temp = min(self.readings)
                max_temp = max(self.readings)
                avg_temp = sum(self.readings) / len(self.readings)

                self.get_logger().info(
                    f'Temperature Stats - Min: {min_temp:.2f}°C, '
                    f'Max: {max_temp:.2f}°C, Avg: {avg_temp:.2f}°C, '
                    f'High Temp Alerts: {self.high_temp_alert_count}'
                )

def main(args=None):
    rclpy.init(args=args)
    node = TemperatureAnalyzer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down temperature analyzer...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 3: Create a Custom Service

### Objective
Create a ROS 2 service that calculates the distance between two 2D points.

### Requirements
- Create a service named `calculate_distance`
- Service type should accept two points (x1, y1) and (x2, y2)
- Return the Euclidean distance between the points
- Use custom service definition or built-in types

### Solution
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import TwoInts  # Using TwoInts as a simple example
import math

class DistanceCalculatorServer(Node):
    def __init__(self):
        super().__init__('distance_calculator_server')
        # For this example, we'll use TwoInts and adapt it conceptually
        # In practice, you'd define a custom service type
        self.srv = self.create_service(
            TwoInts,
            'calculate_distance',
            self.calculate_distance_callback
        )
        self.get_logger().info('Distance Calculator Service Started')

    def calculate_distance_callback(self, request, response):
        # Conceptually: treat request.a, request.b as point 1 coordinates
        # and we'd need a custom message for full implementation
        # This is a simplified example
        x1, y1 = request.a, request.b
        # For a full implementation, you'd need a custom service with more fields

        # Simplified: return the sum as an example
        response.sum = abs(x1) + abs(y1)  # Placeholder
        self.get_logger().info(f'Distance calculation: result = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = DistanceCalculatorServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down distance calculator server...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

For a complete custom service implementation, you would define a `.srv` file like:

`CalculateDistance.srv`:
```
float64 x1
float64 y1
float64 x2
float64 y2
---
float64 distance
```

## Exercise 4: URDF Modification

### Objective
Modify the simple humanoid URDF to add an arm.

### Requirements
- Add a left arm to the simple humanoid model
- The arm should have upper arm, lower arm, and hand links
- Use appropriate joint types to allow realistic movement
- Define proper visual and collision properties

### Solution Template
```xml
<robot name="simple_humanoid_with_arm">
  <!-- Base link (torso) -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint connecting head to torso -->
  <joint name="neck_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <!-- TODO: Add left arm with upper arm, lower arm, and hand -->
  <!-- Upper Arm -->
  <link name="upper_arm">
    <!-- TODO: Define inertial, visual, and collision properties -->
  </link>

  <!-- Joint connecting upper arm to torso -->
  <joint name="shoulder_joint" type="revolute">
    <!-- TODO: Define parent, child, origin, axis, and limits -->
  </joint>

  <!-- TODO: Add lower arm and hand links and joints -->
</robot>
```

### Solution
```xml
<robot name="simple_humanoid_with_arm">
  <!-- Base link (torso) -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint connecting head to torso -->
  <joint name="neck_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <!-- Upper Arm -->
  <link name="upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting upper arm to torso -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0.3 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Lower Arm -->
  <link name="lower_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.12"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.008"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.12"/>
      <geometry>
        <cylinder radius="0.04" length="0.24"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.12"/>
      <geometry>
        <cylinder radius="0.04" length="0.24"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting lower arm to upper arm -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm"/>
    <child link="lower_arm"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Hand -->
  <link name="hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting hand to lower arm -->
  <joint name="wrist_joint" type="fixed">
    <parent link="lower_arm"/>
    <child link="hand"/>
    <origin xyz="0 0 -0.24"/>
  </joint>
</robot>
```

## Exercise 5: Python-ROS Bridge Enhancement

### Objective
Enhance the Python-ROS bridge example to handle multiple types of messages and perform more complex processing.

### Requirements
- Modify the bridge to handle both sensor data and command messages
- Implement a simple decision-making algorithm in the Python agent
- Add parameter configuration for the agent behavior
- Include error handling and graceful degradation

### Solution
```python
#!/usr/bin/env python3
"""
Enhanced ROS 2 Python Bridge Example
This script demonstrates an enhanced bridge with multiple message types and decision making.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy


class EnhancedPythonROSBridge(Node):
    """
    Enhanced bridge with multiple message types and decision making.
    """

    def __init__(self):
        super().__init__('enhanced_python_ros_bridge')

        # Declare parameters
        self.declare_parameter('safety_threshold', 30.0)
        self.declare_parameter('agent_response_time', 0.5)
        self.declare_parameter('log_level', 'info')

        # Get parameter values
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.response_time = self.get_parameter('agent_response_time').value
        self.log_level = self.get_parameter('log_level').value

        # Create publishers for different message types
        self.status_publisher = self.create_publisher(String, 'agent_status', 10)
        self.command_publisher = self.create_publisher(String, 'robot_command', 10)
        self.alert_publisher = self.create_publisher(String, 'safety_alert', 10)

        # Create subscriptions for different message types
        self.sensor_subscription = self.create_subscription(
            Float32,
            'sensor_data',
            self.sensor_callback,
            10)

        self.command_subscription = self.create_subscription(
            String,
            'external_command',
            self.command_callback,
            10)

        # Create timer for periodic tasks
        self.timer = self.create_timer(self.response_time, self.timer_callback)

        # Internal state
        self.last_sensor_value = 0.0
        self.agent_state = "IDLE"
        self.command_queue = []

        self.get_logger().info(f'Enhanced Python-ROS Bridge Started with threshold: {self.safety_threshold}')

    def sensor_callback(self, msg):
        """Callback for sensor data"""
        self.last_sensor_value = msg.data
        self.get_logger().info(f'Received sensor data: {msg.data}')

        # Decision making based on sensor value
        if msg.data > self.safety_threshold:
            self.agent_state = "ALERT"
            alert_msg = String()
            alert_msg.data = f'SAFETY_ALERT: Sensor value {msg.data} exceeds threshold {self.safety_threshold}'
            self.alert_publisher.publish(alert_msg)
        else:
            self.agent_state = "NORMAL"

        # Publish agent status
        status_msg = String()
        status_msg.data = f'Sensor: {msg.data}, State: {self.agent_state}'
        self.status_publisher.publish(status_msg)

    def command_callback(self, msg):
        """Callback for external commands"""
        self.get_logger().info(f'Received external command: {msg.data}')
        self.command_queue.append(msg.data)

        # Process command if appropriate
        if self.agent_state == "NORMAL":
            response_msg = String()
            response_msg.data = f'EXECUTING: {msg.data}'
            self.command_publisher.publish(response_msg)
        else:
            response_msg = String()
            response_msg.data = f'DEFERRED: {msg.data} (Safety alert active)'
            self.status_publisher.publish(response_msg)

    def timer_callback(self):
        """Periodic tasks"""
        # Process any queued commands
        if self.command_queue and self.agent_state == "NORMAL":
            cmd = self.command_queue.pop(0)
            self.get_logger().info(f'Processing queued command: {cmd}')

            response_msg = String()
            response_msg.data = f'PROCESSED: {cmd}'
            self.command_publisher.publish(response_msg)

        # Periodic status update
        status_msg = String()
        status_msg.data = f'Agent Status - Sensor: {self.last_sensor_value}, State: {self.agent_state}, Queue: {len(self.command_queue)}'
        self.status_publisher.publish(status_msg)

    def on_shutdown(self):
        """Cleanup on shutdown"""
        self.get_logger().info('Enhanced Python-ROS Bridge shutting down...')


def main(args=None):
    """Main function to run the enhanced bridge"""
    rclpy.init(args=args)

    try:
        bridge_node = EnhancedPythonROSBridge()
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        print("\nShutting down Enhanced Python-ROS Bridge...")
    finally:
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Exercise 6: Integration Challenge

### Objective
Create a complete ROS 2 system that combines nodes, topics, services, and demonstrates the concepts learned in this module.

### Requirements
- Create a "robot health monitor" system
- Include at least 3 nodes: sensor simulator, health monitor, and alert system
- Use both topics and services for communication
- Implement proper error handling
- Create a launch file to start the entire system

### System Architecture
1. **Sensor Simulator Node**: Simulates various robot sensors (temperature, battery, joint angles)
2. **Health Monitor Node**: Analyzes sensor data and detects anomalies
3. **Alert System Node**: Responds to health alerts and takes appropriate actions

This exercise combines all the concepts from this module into a practical application.

## Summary

These exercises have reinforced your understanding of:
- Creating and managing ROS 2 nodes
- Implementing publisher-subscriber communication patterns
- Developing service-based request-response systems
- Working with URDF for robot modeling
- Building Python-ROS bridges for intelligent agents

Completing these exercises provides hands-on experience with the core concepts of ROS 2 and prepares you for more advanced robotics development.

## Additional Challenges

1. **Advanced URDF**: Create a complete humanoid robot URDF with both arms and legs
2. **Action Servers**: Implement ROS 2 actions for long-running tasks
3. **Parameter Server**: Use ROS 2 parameter server for runtime configuration
4. **Launch Files**: Create launch files to start multiple nodes simultaneously
5. **Testing**: Write unit tests for your ROS 2 nodes using `launch_testing`

---

**Congratulations! You have completed Module 1: The Robotic Nervous System (ROS 2). Continue to the next module to learn about simulation and digital twins.**