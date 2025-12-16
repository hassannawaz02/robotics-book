#!/usr/bin/env python3
"""
Bipedal Path Planning Example
This script demonstrates path planning specifically for bipedal humanoid movement
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math
import numpy as np


class BipedalPathPlanner(Node):
    def __init__(self):
        super().__init__('bipedal_path_planner')

        # Publishers for bipedal control
        self.footstep_pub = self.create_publisher(Float64MultiArray, 'footstep_plan', 10)
        self.com_pub = self.create_publisher(PoseStamped, 'center_of_mass', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)

        # Parameters for bipedal walking
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters (distance between feet)
        self.step_height = 0.1  # meters (foot lift height)
        self.step_duration = 1.0  # seconds per step
        self.zmp_margin = 0.05  # Zero Moment Point safety margin

        # Robot parameters
        self.leg_length = 0.8  # meters
        self.com_height = 0.7  # center of mass height
        self.robot_width = 0.3  # robot width

        self.get_logger().info('Bipedal Path Planner initialized')

    def calculate_footsteps(self, start_pos, goal_pos):
        """Calculate footsteps for bipedal movement from start to goal"""
        footsteps = []

        # Calculate distance and direction
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        direction = math.atan2(dy, dx)

        # Calculate number of steps needed
        num_steps = int(distance / self.step_length) + 1

        # Generate footsteps
        for i in range(num_steps):
            step_progress = (i + 1) / num_steps
            step_x = start_pos[0] + step_progress * dx
            step_y = start_pos[1] + step_progress * dy

            # Alternate between left and right foot
            foot_offset = self.step_width / 2 if i % 2 == 0 else -self.step_width / 2
            step_y += foot_offset * math.cos(direction)
            step_x += foot_offset * math.sin(direction)

            # Create footstep
            footstep = {
                'x': step_x,
                'y': step_y,
                'z': 0.0,  # Ground level
                'yaw': direction,
                'foot': 'left' if i % 2 == 0 else 'right',
                'step_number': i + 1
            }

            footsteps.append(footstep)

        return footsteps

    def generate_zmp_trajectory(self, footsteps):
        """Generate Zero Moment Point trajectory for stable walking"""
        zmp_points = []

        for i, footstep in enumerate(footsteps):
            # Simplified ZMP planning - in real implementation, this would be more complex
            zmp_x = footstep['x']
            zmp_y = footstep['y']

            # Add some stability margin
            zmp_point = {
                'x': zmp_x,
                'y': zmp_y,
                'time': i * self.step_duration
            }

            zmp_points.append(zmp_point)

        return zmp_points

    def plan_com_trajectory(self, footsteps):
        """Plan Center of Mass trajectory for stable walking"""
        com_trajectory = []

        # Use inverted pendulum model for CoM planning
        omega = math.sqrt(9.81 / self.com_height)  # Natural frequency

        for i, footstep in enumerate(footsteps):
            t = i * self.step_duration

            # Simplified CoM trajectory planning
            # In real implementation, this would use more sophisticated methods like DCM (Divergent Component of Motion)
            com_x = footstep['x']
            com_y = footstep['y']

            # Add some dynamic adjustment for balance
            com_point = {
                'x': com_x,
                'y': com_y,
                'z': self.com_height,
                'time': t
            }

            com_trajectory.append(com_point)

        return com_trajectory

    def execute_bipedal_walk(self, goal_pos):
        """Execute bipedal walking to goal position"""
        start_pos = [0.0, 0.0]  # Starting at origin
        footsteps = self.calculate_footsteps(start_pos, goal_pos)

        self.get_logger().info(f'Calculated {len(footsteps)} footsteps to reach goal: {goal_pos}')

        # Generate ZMP trajectory
        zmp_trajectory = self.generate_zmp_trajectory(footsteps)

        # Generate CoM trajectory
        com_trajectory = self.plan_com_trajectory(footsteps)

        # Publish footstep plan
        footstep_msg = Float64MultiArray()
        for footstep in footsteps:
            footstep_msg.data.extend([footstep['x'], footstep['y'], footstep['z'], footstep['yaw']])
        self.footstep_pub.publish(footstep_msg)

        # Publish CoM trajectory points
        for com_point in com_trajectory:
            com_pose = PoseStamped()
            com_pose.header.stamp = self.get_clock().now().to_msg()
            com_pose.header.frame_id = 'map'
            com_pose.pose.position.x = com_point['x']
            com_pose.pose.position.y = com_point['y']
            com_pose.pose.position.z = com_point['z']
            com_pose.pose.orientation.w = 1.0
            self.com_pub.publish(com_pose)

        return footsteps, zmp_trajectory, com_trajectory

    def generate_joint_trajectories(self, footsteps):
        """Generate joint angle trajectories for walking motion"""
        joint_state = JointState()
        joint_state.name = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]

        # Simplified joint angles - in real implementation, inverse kinematics would be used
        joint_angles = [0.0] * len(joint_state.name)
        joint_state.position = joint_angles

        # Add walking pattern
        for i, footstep in enumerate(footsteps):
            # This is a simplified walking gait
            # Real implementation would use proper inverse kinematics and gait patterns
            phase = (i % 4) / 4.0  # Normalize to 0-1 range

            # Hip and knee movements for walking
            left_hip_angle = math.sin(phase * 2 * math.pi) * 0.2
            right_hip_angle = math.sin(phase * 2 * math.pi + math.pi) * 0.2

            joint_state.position[0] = left_hip_angle  # left_hip_pitch
            joint_state.position[7] = right_hip_angle  # right_hip_pitch

            self.joint_cmd_pub.publish(joint_state)


def main(args=None):
    rclpy.init(args=args)

    bipedal_planner = BipedalPathPlanner()

    # Example: Walk to a specific point
    goal_position = [3.0, 2.0]
    footsteps, zmp_trajectory, com_trajectory = bipedal_planner.execute_bipedal_walk(goal_position)

    bipedal_planner.get_logger().info(f'Footsteps: {len(footsteps)}, ZMP points: {len(zmp_trajectory)}, CoM points: {len(com_trajectory)}')

    try:
        rclpy.spin(bipedal_planner)
    except KeyboardInterrupt:
        pass
    finally:
        bipedal_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()