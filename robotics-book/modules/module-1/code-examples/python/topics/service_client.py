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
        b = int(sys.argv[2])

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