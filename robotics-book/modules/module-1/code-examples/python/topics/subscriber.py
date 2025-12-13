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