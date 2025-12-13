#!/usr/bin/env python3
"""
ROS 2 Python Bridge Example
This script demonstrates how to bridge Python agents to ROS 2 controllers using rclpy.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32


class PythonROSBridge(Node):
    """
    Example bridge between Python application logic and ROS 2 controllers.
    Demonstrates how Python agents can interact with ROS 2 nodes.
    """

    def __init__(self):
        super().__init__('python_ros_bridge')

        # Create a publisher for sending messages from Python agent to ROS
        self.publisher_ = self.create_publisher(String, 'python_agent_output', 10)

        # Create a subscriber for receiving messages from ROS controllers
        self.subscription_ = self.create_subscription(
            String,
            'ros_controller_input',
            self.listener_callback,
            10)

        # Create a timer to periodically send messages
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Counter for demonstration
        self.counter = 0

        self.get_logger().info('Python-ROS Bridge Node Started')

    def listener_callback(self, msg):
        """Callback for receiving messages from ROS controllers"""
        self.get_logger().info(f'Received from ROS controller: {msg.data}')

        # Process the message as part of the Python agent logic
        processed_data = self.process_ros_data(msg.data)

        # Send processed data back to ROS
        response_msg = String()
        response_msg.data = f'Processed: {processed_data}'
        self.publisher_.publish(response_msg)
        self.get_logger().info(f'Sent to ROS: {response_msg.data}')

    def timer_callback(self):
        """Timer callback to periodically send messages"""
        msg = String()
        msg.data = f'Python agent message {self.counter}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.counter += 1

    def process_ros_data(self, data):
        """Process data received from ROS controllers"""
        # In a real application, this would contain the Python agent logic
        return f"[PYTHON_AGENT_PROCESSED] {data}"


def main(args=None):
    """Main function to run the Python-ROS bridge"""
    rclpy.init(args=args)

    # Create the bridge node
    bridge_node = PythonROSBridge()

    try:
        # Spin the node to process callbacks
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        print("\nShutting down Python-ROS Bridge...")
    finally:
        # Clean up
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()