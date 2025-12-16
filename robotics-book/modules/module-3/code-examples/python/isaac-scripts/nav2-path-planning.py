#!/usr/bin/env python3
"""
Nav2 Path Planning Example
This script demonstrates basic path planning using Nav2 for robotics navigation
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import math


class Nav2PathPlanner(Node):
    def __init__(self):
        super().__init__('nav2_path_planner')

        # Action client for Nav2 navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.marker_pub = self.create_publisher(Marker, 'path_markers', 10)

        # Parameters for path planning
        self.planning_frequency = 1.0  # Hz
        self.safe_distance = 0.5  # meters
        self.max_planning_distance = 10.0  # meters

        self.get_logger().info('Nav2 Path Planner initialized')

    def create_goal_pose(self, x, y, z=0.0, yaw=0.0):
        """Create a PoseStamped message for navigation goal"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = z

        # Convert yaw to quaternion (simplified)
        # In a real implementation, proper conversion would be used
        goal_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_pose.pose.orientation.w = math.cos(yaw / 2.0)

        return goal_pose

    def send_goal_pose(self, pose_stamped):
        """Send a goal pose to Nav2 navigation system"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped

        # Wait for action server
        self.nav_to_pose_client.wait_for_server()

        # Send goal
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        return future

    def plan_path(self, start_pose, goal_pose):
        """Plan a path from start to goal using simple interpolation"""
        # This is a simplified path planner
        # In a real implementation, this would use proper path planning algorithms

        path_points = []
        steps = 20  # Number of intermediate points

        start_x = start_pose.pose.position.x
        start_y = start_pose.pose.position.y
        goal_x = goal_pose.pose.position.x
        goal_y = goal_pose.pose.position.y

        for i in range(steps + 1):
            t = i / steps
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)

            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            path_points.append(point)

        return path_points

    def visualize_path(self, path_points):
        """Visualize the planned path using markers"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'planned_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = path_points

        self.marker_pub.publish(marker)

    def execute_navigation(self, goal_x, goal_y):
        """Execute navigation to specified goal coordinates"""
        goal_pose = self.create_goal_pose(goal_x, goal_y)
        future = self.send_goal_pose(goal_pose)

        # Plan path for visualization
        # Note: In real Nav2, path planning is done internally
        # This is just for demonstration
        current_pose = self.create_goal_pose(0.0, 0.0)  # Starting at origin
        path_points = self.plan_path(current_pose, goal_pose)
        self.visualize_path(path_points)

        self.get_logger().info(f'Navigating to goal: ({goal_x}, {goal_y})')

        return future


def main(args=None):
    rclpy.init(args=args)

    path_planner = Nav2PathPlanner()

    # Example: Navigate to a specific point
    future = path_planner.execute_navigation(5.0, 3.0)

    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()