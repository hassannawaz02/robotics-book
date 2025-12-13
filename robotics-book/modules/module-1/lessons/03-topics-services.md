---
title: Topics and Services - Communication Patterns
sidebar_position: 3
---

# Topics and Services - Communication Patterns

## Overview

Communication is the backbone of any distributed robotic system. In ROS 2, nodes communicate with each other through various mechanisms, with **topics** and **services** being the two most fundamental patterns. Understanding these communication patterns is essential for developing coordinated robotic systems, particularly for humanoid robots where multiple subsystems need to exchange information seamlessly.

## Topics - Asynchronous Communication

Topics enable **asynchronous, one-way communication** between nodes. They follow a **publish-subscribe pattern** where:
- **Publishers** send messages to a topic
- **Subscribers** receive messages from a topic
- Multiple publishers and subscribers can exist for the same topic
- Communication is decoupled in time (publishers and subscribers don't need to be active simultaneously)

### Topic Characteristics

- **Unidirectional**: Data flows from publisher to subscriber
- **Asynchronous**: Publishers and subscribers operate independently
- **Many-to-many**: Multiple publishers can send to the same topic; multiple subscribers can receive from the same topic
- **Data-driven**: Communication is triggered by data availability

### Topic Example - Publisher

Here's the publisher example from our code:

```python
#!/usr/bin/env python3
"""
ROS 2 Publisher Example
This script demonstrates how to create a publisher node that sends messages to a topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PublisherNode(Node):
    """
    A publisher node that sends messages to a topic.
    """

    def __init__(self):
        super().__init__('publisher_node')

        # Create a publisher for the 'topic_name' topic
        self.publisher_ = self.create_publisher(String, 'topic_name', 10)

        # Create a timer to publish messages periodically
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Counter for messages
        self.counter = 0

        self.get_logger().info('Publisher Node Started')

    def timer_callback(self):
        """Callback function for the timer"""
        msg = String()
        msg.data = f'Hello from publisher! Message #{self.counter}'

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

        self.counter += 1


def main(args=None):
    """Main function to run the publisher node"""
    rclpy.init(args=args)

    # Create the publisher node
    publisher_node = PublisherNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        print("\nShutting down publisher node...")
    finally:
        # Clean up
        publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Topic Example - Subscriber

Here's the subscriber example:

```python
#!/usr/bin/env python3
"""
ROS 2 Subscriber Example
This script demonstrates how to create a subscriber node that receives messages from a topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SubscriberNode(Node):
    """
    A subscriber node that receives messages from a topic.
    """

    def __init__(self):
        super().__init__('subscriber_node')

        # Create a subscription to the 'topic_name' topic
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)

        # Make sure the subscription is properly configured
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Subscriber Node Started')

    def listener_callback(self, msg):
        """Callback function to process incoming messages"""
        self.get_logger().info(f'Received: {msg.data}')


def main(args=None):
    """Main function to run the subscriber node"""
    rclpy.init(args=args)

    # Create the subscriber node
    subscriber_node = SubscriberNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        print("\nShutting down subscriber node...")
    finally:
        # Clean up
        subscriber_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Services - Synchronous Communication

Services enable **synchronous, request-response communication** between nodes. They follow a **client-server pattern** where:
- **Service servers** provide a specific functionality
- **Service clients** request that functionality
- Communication is synchronous (client waits for response)
- Request-response pattern ensures completion

### Service Characteristics

- **Bidirectional**: Request goes from client to server, response goes from server to client
- **Synchronous**: Client waits for server response
- **One-to-one**: One client talks to one server at a time
- **Action-oriented**: Communication is triggered by specific actions or requests

### Service Example - Server

Here's the service server example:

```python
#!/usr/bin/env python3
"""
ROS 2 Service Server Example
This script demonstrates how to create a service server that responds to requests.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class ServiceServerNode(Node):
    """
    A service server node that responds to requests to add two integers.
    """

    def __init__(self):
        super().__init__('service_server_node')

        # Create a service server for the 'add_two_ints' service
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

        self.get_logger().info('Service Server Node Started')

    def add_two_ints_callback(self, request, response):
        """Callback function to handle service requests"""
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response


def main(args=None):
    """Main function to run the service server node"""
    rclpy.init(args=args)

    # Create the service server node
    service_server_node = ServiceServerNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(service_server_node)
    except KeyboardInterrupt:
        print("\nShutting down service server node...")
    finally:
        # Clean up
        service_server_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Example - Client

Here's the service client example:

```python
#!/usr/bin/env python3
"""
ROS 2 Service Client Example
This script demonstrates how to create a service client that sends requests to a service.
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class ServiceClientNode(Node):
    """
    A service client node that sends requests to add two integers.
    """

    def __init__(self):
        super().__init__('service_client_node')

        # Create a client for the 'add_two_ints' service
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        """Send a request to the service"""
        self.request.a = a
        self.request.b = b

        # Call the service asynchronously
        self.future = self.client.call_async(self.request)
        self.get_logger().info(f'Sent request: {a} + {b}')


def main(args=None):
    """Main function to run the service client node"""
    rclpy.init(args=args)

    # Create the service client node
    service_client_node = ServiceClientNode()

    # Check if command line arguments are provided
    if len(sys.argv) != 3:
        print('Usage: python3 service_client.py <int1> <int2>')
        sys.exit(1)

    try:
        # Parse command line arguments
        a = int(sys.argv[1])
        b = int(argv[2])

        # Send the request
        service_client_node.send_request(a, b)

        # Spin until the response is received
        while rclpy.ok():
            rclpy.spin_once(service_client_node)
            if service_client_node.future.done():
                try:
                    response = service_client_node.future.result()
                    print(f'Result of {a} + {b} = {response.sum}')
                except Exception as e:
                    service_client_node.get_logger().info(f'Service call failed: {e}')
                break

    except ValueError:
        print('Please provide two integers as arguments')
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down service client node...")
    finally:
        # Clean up
        service_client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## When to Use Topics vs Services

### Use Topics when:
- You need continuous data flow (sensor data, robot state)
- Publishers and subscribers don't need to be synchronized
- Multiple nodes need to receive the same information
- Communication doesn't require acknowledgment
- You're implementing a streaming or broadcasting pattern

### Use Services when:
- You need a specific action to be performed
- You require a response to a request
- Communication is transactional in nature
- You need guaranteed delivery and processing
- You're implementing a request-response pattern

## Quality of Service (QoS) Settings

ROS 2 provides Quality of Service settings to control communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Example QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,  # or BEST_EFFORT
    history=HistoryPolicy.KEEP_LAST  # or KEEP_ALL
)
```

## Communication in Humanoid Robotics

In humanoid robotics, different communication patterns serve specific purposes:

### Topics for:
- **Sensor data streaming**: Camera feeds, IMU data, joint positions
- **Robot state broadcasting**: Current joint angles, robot pose, battery status
- **Command streaming**: Continuous control commands for walking or manipulation

### Services for:
- **Action execution**: "Walk to location", "Grasp object", "Perform action"
- **Configuration changes**: Update parameters, change modes
- **One-time queries**: Request robot status, calibration data

## Exercises

1. Create a topic-based system where one node publishes sensor data and multiple nodes subscribe to process it differently.
2. Implement a service that calculates the distance between two points in 3D space.
3. Research how topics and services are used in a real humanoid robot system (e.g., ROS-enabled robots) and describe the communication architecture.

---

**Continue to [URDF - Unified Robot Description Format](./04-urdf.md) to learn about robot modeling.**