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