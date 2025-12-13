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