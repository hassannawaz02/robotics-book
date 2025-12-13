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