---
title: ROS 2 Nodes - The Building Blocks
sidebar_position: 2
---

# ROS 2 Nodes - The Building Blocks

## Overview

In ROS 2, a **node** is a fundamental building block that represents a single process performing computation. Nodes are the basic execution units of a ROS 2 program and are designed to be modular and reusable. Understanding nodes is crucial for developing any ROS 2 application, especially for humanoid robots where multiple nodes might control different subsystems like walking, vision, speech, and manipulation.

## What is a Node?

A node in ROS 2 is:
- A process that performs computation
- The basic unit of a ROS 2 program
- Responsible for a specific task within the robot system
- Able to communicate with other nodes through topics, services, and actions

Nodes are designed to be lightweight and focused on a single responsibility, following the Unix philosophy of "do one thing and do it well."

## Node Structure in Python

To create a node in Python using ROS 2, you'll typically:

1. Import the necessary ROS 2 modules
2. Create a class that inherits from `rclpy.node.Node`
3. Initialize the node in the constructor
4. Implement the node's functionality
5. Create a main function to run the node

Here's the basic structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize node components here

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Your First Node

Let's examine the simple node example we created in the code examples:

```python
#!/usr/bin/env python3
"""
Simple ROS 2 Node Example
This script demonstrates the basic structure of a ROS 2 node.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleNode(Node):
    """
    A simple ROS 2 node that publishes messages to a topic.
    """

    def __init__(self):
        super().__init__('simple_node')

        # Create a publisher
        self.publisher_ = self.create_publisher(String, 'chatter', 10)

        # Create a timer to publish messages periodically
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Counter for messages
        self.counter = 0

        self.get_logger().info('Simple Node Started')

    def timer_callback(self):
        """Callback function for the timer"""
        msg = String()
        msg.data = f'Hello ROS 2! Message #{self.counter}'

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        self.counter += 1


def main(args=None):
    """Main function to run the simple node"""
    rclpy.init(args=args)

    # Create the node
    simple_node = SimpleNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(simple_node)
    except KeyboardInterrupt:
        print("\nShutting down simple node...")
    finally:
        # Clean up
        simple_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Key Node Components

### 1. Initialization
```python
super().__init__('node_name')
```
The node name must be unique within your ROS 2 system. It helps identify the node in logs and debugging.

### 2. Publishers and Subscribers
```python
self.publisher_ = self.create_publisher(String, 'chatter', 10)
```
Publishers allow nodes to send messages to topics. The third parameter (10) is the queue size for outgoing messages.

### 3. Timers
```python
self.timer = self.create_timer(0.5, self.timer_callback)
```
Timers allow nodes to perform actions at regular intervals. The first parameter is the period in seconds.

### 4. Logging
```python
self.get_logger().info('Simple Node Started')
```
ROS 2 provides built-in logging capabilities that are essential for debugging distributed systems.

## Running a Node

To run a ROS 2 node:

1. Make sure your ROS 2 environment is sourced:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Navigate to your workspace and source the setup file:
   ```bash
   cd ~/ros2_ws
   source install/setup.bash
   ```

3. Run the node:
   ```bash
   ros2 run package_name node_script_name
   ```

## Node Lifecycle

ROS 2 nodes follow a specific lifecycle:
- **Unconfigured**: Node is created but not yet configured
- **Inactive**: Node is configured but not yet activated
- **Active**: Node is running and processing callbacks
- **Finalized**: Node is shutting down

## Nodes in Humanoid Robotics

In humanoid robotics, different aspects of the robot are typically controlled by separate nodes:
- **Walking controller node**: Handles locomotion and balance
- **Vision processing node**: Processes camera data for object detection
- **Speech recognition node**: Handles voice commands
- **Motion planning node**: Plans arm movements and manipulations
- **Sensor fusion node**: Combines data from multiple sensors

This modular approach allows for better maintainability and testing of individual components.

## Best Practices

1. **Single Responsibility**: Each node should focus on one specific task
2. **Error Handling**: Implement proper error handling and cleanup
3. **Logging**: Use ROS 2's logging system for debugging
4. **Parameter Management**: Use ROS 2 parameters for configuration
5. **Resource Management**: Properly clean up resources when the node shuts down

## Exercises

1. Modify the simple node example to publish a different type of message (e.g., integer or custom message)
2. Create a node that subscribes to a topic and logs the received messages
3. Research and explain how nodes in a humanoid robot like Boston Dynamics' Atlas might be organized

---

**Continue to [Topics and Services](./03-topics-services.md) to learn about communication between nodes.**