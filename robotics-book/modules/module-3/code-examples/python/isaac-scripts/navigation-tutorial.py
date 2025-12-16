#!/usr/bin/env python3
"""
Isaac ROS Navigation Tutorial
This script demonstrates navigation concepts using Isaac ROS
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
import math
import numpy as np


class IsaacNavigatorNode(Node):
    def __init__(self):
        super().__init__('isaac_navigator_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.path_pub = self.create_publisher(Path, '/current_path', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Navigation parameters
        self.current_pose = None
        self.goal_pose = None
        self.path = []
        self.linear_vel = 0.5
        self.angular_vel = 0.5
        self.safe_distance = 0.5  # meters

        # Timer for navigation control
        self.timer = self.create_timer(0.1, self.navigation_callback)

        self.get_logger().info('Isaac Navigator Node initialized')

    def odom_callback(self, msg):
        """Update current robot pose from odometry"""
        self.current_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

        # Extract yaw from quaternion (simplified)
        # In a real implementation, proper quaternion to euler conversion would be used
        self.current_yaw = 0.0

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Check for obstacles in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 50 : len(msg.ranges)//2 + 50]
        min_distance = min([r for r in front_scan if not math.isnan(r)], default=float('inf'))

        if min_distance < self.safe_distance:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def set_goal(self, x, y, z=0.0):
        """Set navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z

        self.goal_pose = [x, y, z]
        self.goal_pub.publish(goal_msg)

    def navigation_callback(self):
        """Main navigation control loop"""
        if self.current_pose is None or self.goal_pose is None:
            return

        # Calculate distance to goal
        dx = self.goal_pose[0] - self.current_pose[0]
        dy = self.goal_pose[1] - self.current_pose[1]
        distance = math.sqrt(dx**2 + dy**2)

        # Create Twist message for robot movement
        cmd_vel = Twist()

        if distance > 0.2:  # If not close to goal
            # Calculate angle to goal
            angle_to_goal = math.atan2(dy, dx)
            angle_diff = angle_to_goal - self.current_yaw

            # Normalize angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Simple proportional controller
            if abs(angle_diff) > 0.1:  # Need to turn
                cmd_vel.angular.z = self.angular_vel * angle_diff
            else:  # Can move forward
                cmd_vel.linear.x = min(self.linear_vel, distance)

            # Check for obstacles before moving forward
            # (In a real implementation, this would use scan data)
            cmd_vel.linear.x = 0.0 if cmd_vel.linear.x > 0.1 and self.is_obstacle_ahead() else cmd_vel.linear.x

        self.cmd_vel_pub.publish(cmd_vel)

    def is_obstacle_ahead(self):
        """Check if there's an obstacle ahead (simplified)"""
        # In a real implementation, this would check laser scan data
        return False

    def follow_path(self, waypoints):
        """Follow a predefined path of waypoints"""
        self.path = waypoints
        if len(self.path) > 0:
            next_waypoint = self.path[0]
            self.set_goal(next_waypoint[0], next_waypoint[1])


def main(args=None):
    rclpy.init(args=args)

    navigator = IsaacNavigatorNode()

    # Example: Set a goal at (2.0, 2.0)
    navigator.set_goal(2.0, 2.0)

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot on shutdown
        cmd_vel = Twist()
        navigator.cmd_vel_pub.publish(cmd_vel)
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()