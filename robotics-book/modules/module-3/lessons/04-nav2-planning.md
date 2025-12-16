---
title: Nav2 for Bipedal Robots and Humanoid Navigation
sidebar_label: Nav2 Bipedal
---

# Nav2 for Bipedal Robots and Humanoid Navigation

This lesson provides comprehensive coverage of Nav2 (Navigation 2) for bipedal humanoid robots with detailed humanoid-specific navigation strategies and advanced path planning algorithms. Nav2 is the navigation stack for ROS 2 with specific enhancements for humanoid robotics, offering sophisticated algorithms for finding optimal paths while considering the unique balance and movement constraints of two-legged robots.

## Learning Objectives

By the end of this lesson, you will understand:
- How to implement Nav2 navigation specifically for bipedal humanoid robots with balance constraints
- How to plan sophisticated paths that account for humanoid gait patterns and stability
- How to configure advanced Nav2 behavior trees for complex bipedal navigation tasks
- How to implement humanoid-specific local and global path planners
- How to integrate Nav2 with Isaac Sim for testing and validation of bipedal navigation
- How to optimize navigation performance for real-time humanoid locomotion

## Prerequisites

- Isaac Sim environment configured from Lesson 2
- Isaac ROS VSLAM pipeline from Lesson 3
- Nav2 installed (2.0+ with humanoid extensions)
- Python 3.11+ environment with Isaac-compatible libraries
- Basic knowledge of ROS 2 navigation and humanoid robotics concepts
- Understanding of balance control and gait planning for bipedal robots

## Comprehensive Nav2 Architecture for Humanoid Robots

### Nav2 Overview and Humanoid Extensions

Navigation2 (Nav2) is the next-generation navigation stack for ROS 2, specifically enhanced for humanoid robotics applications. Unlike traditional wheeled robot navigation, humanoid navigation must account for:

- **Balance Constraints**: Maintaining center of mass within support polygon
- **Gait Patterns**: Coordinated leg movement for stable locomotion
- **Footstep Planning**: Precise placement of feet for stability
- **Dynamic Stability**: Continuous balance adjustment during movement

The Nav2 architecture for humanoid robots includes specialized components:

1. **Navigation Server**: Coordinates the entire navigation system
2. **Behavior Trees**: Define complex navigation behaviors with humanoid-specific actions
3. **Humanoid-Specific Planners**: Global and local planners adapted for bipedal movement
4. **Gait Controllers**: Execute coordinated leg movements for locomotion
5. **Balance Controllers**: Maintain stability during navigation
6. **Footstep Planners**: Plan precise foot placements for stable walking

### Behavior Tree Architecture for Humanoid Navigation

Nav2 uses behavior trees to define complex navigation behaviors, with humanoid-specific nodes:

```xml
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="global_plan">
      <ComputePathToPose goal="{goal}" path="{path}"/>
      <SmoothPath input_path="{path}" output_path="{smoothed_path}"/>
    </PipelineSequence>
    <PipelineSequence name="local_plan">
      <FollowPath path="{smoothed_path}" controller_frequency="20"/>
    </PipelineSequence>
    <ReactiveSequence>
      <Spin spin_dist="1.57"/>
      <Wait wait_duration="5"/>
    </ReactiveSequence>
  </BehaviorTree>
</root>
```

For humanoid robots, these trees include additional stability and balance nodes:

```xml
<root main_tree_to_execute="HumanoidMainTree">
  <BehaviorTree ID="HumanoidMainTree">
    <PipelineSequence name="humanoid_global_plan">
      <ComputePathToPose goal="{goal}" path="{path}"/>
      <HumanoidPathValidation input_path="{path}" output_path="{validated_path}"/>
      <FootstepPlanning path="{validated_path}" footsteps="{footsteps}"/>
    </PipelineSequence>
    <PipelineSequence name="humanoid_local_plan">
      <HumanoidFollowPath footsteps="{footsteps}" controller_frequency="20"/>
      <BalanceControl target_com="{target_com}"/>
    </PipelineSequence>
    <ReactiveFallback name="safety_check">
      <IsStable/>
      <RecoverBalance/>
    </ReactiveFallback>
  </BehaviorTree>
</root>
```

## Advanced Path Planning for Bipedal Robots

### Humanoid-Specific Global Planners

Traditional global planners like A* and Dijkstra need modification for humanoid robots. The path must consider:

- **Support Polygon**: Area where feet can be placed for stability
- **Gait Constraints**: Limitations on step size and direction
- **Balance Requirements**: Maintaining center of mass during transitions
- **Terrain Traversability**: Surface stability for bipedal locomotion

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidGlobalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_global_planner')

        # Create subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            '/humanoid_global_plan',
            10
        )

        self.footstep_pub = self.create_publisher(
            MarkerArray,
            '/footsteps',
            10
        )

        # Initialize planner parameters
        self.map_data = None
        self.map_resolution = 0.05  # 5cm resolution
        self.robot_radius = 0.3     # Humanoid robot radius
        self.step_length = 0.4      # Maximum step length for humanoid
        self.step_width = 0.2       # Maximum step width for humanoid
        self.min_step_length = 0.1  # Minimum step length

        # Initialize A* planner with humanoid constraints
        self.initialize_planner()

    def initialize_planner(self):
        """Initialize A* planner with humanoid-specific constraints"""
        self.planner = {
            'start': None,
            'goal': None,
            'open_set': set(),
            'closed_set': set(),
            'g_score': {},  # Cost from start
            'f_score': {},  # Estimated total cost
            'came_from': {}  # Parent nodes for path reconstruction
        }

    def map_callback(self, msg):
        """Receive map data"""
        self.map_data = msg
        self.map_resolution = msg.info.resolution
        self.get_logger().info(f'Received map: {msg.info.width}x{msg.info.height}')

    def plan_path(self, start_pose, goal_pose):
        """Plan path with humanoid-specific constraints"""
        if self.map_data is None:
            self.get_logger().error('Map not received yet')
            return None

        # Convert poses to map coordinates
        start_map = self.world_to_map(start_pose.pose.position)
        goal_map = self.world_to_map(goal_pose.pose.position)

        # Initialize planner
        self.planner['start'] = start_map
        self.planner['goal'] = goal_map
        self.planner['open_set'] = {start_map}
        self.planner['closed_set'] = set()
        self.planner['g_score'] = {start_map: 0}
        self.planner['f_score'] = {start_map: self.heuristic(start_map, goal_map)}
        self.planner['came_from'] = {}

        # A* algorithm with humanoid constraints
        while self.planner['open_set']:
            # Find node with minimum f_score
            current = min(self.planner['open_set'],
                         key=lambda x: self.planner['f_score'].get(x, float('inf')))

            if current == goal_map:
                # Path found, reconstruct it
                return self.reconstruct_path(current)

            self.planner['open_set'].remove(current)
            self.planner['closed_set'].add(current)

            # Get neighbors with humanoid constraints
            for neighbor in self.get_humanoid_neighbors(current):
                if neighbor in self.planner['closed_set']:
                    continue

                # Check if neighbor is traversable for humanoid
                if not self.is_traversable_for_humanoid(neighbor):
                    continue

                # Calculate tentative g_score
                tentative_g_score = self.planner['g_score'][current] + \
                                   self.humanoid_cost(current, neighbor)

                if neighbor not in self.planner['g_score'] or \
                   tentative_g_score < self.planner['g_score'][neighbor]:
                    # This path to neighbor is better
                    self.planner['came_from'][neighbor] = current
                    self.planner['g_score'][neighbor] = tentative_g_score
                    self.planner['f_score'][neighbor] = tentative_g_score + \
                                                      self.heuristic(neighbor, goal_map)

                    if neighbor not in self.planner['open_set']:
                        self.planner['open_set'].add(neighbor)

        # No path found
        return None

    def get_humanoid_neighbors(self, pos):
        """Get neighbors considering humanoid step constraints"""
        neighbors = []

        # Define possible step directions and distances for humanoid
        step_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions
        step_distances = [self.min_step_length, self.step_length * 0.7, self.step_length]

        for angle in step_angles:
            for dist in step_distances:
                # Convert angle to radians
                rad = math.radians(angle)
                dx = int(round(dist * math.cos(rad) / self.map_resolution))
                dy = int(round(dist * math.sin(rad) / self.map_resolution))

                new_pos = (pos[0] + dx, pos[1] + dy)

                # Check bounds
                if (0 <= new_pos[0] < self.map_data.info.width and
                    0 <= new_pos[1] < self.map_data.info.height):
                    neighbors.append(new_pos)

        return neighbors

    def is_traversable_for_humanoid(self, pos):
        """Check if position is traversable for humanoid robot"""
        # Check if position is in obstacle space
        map_index = pos[1] * self.map_data.info.width + pos[0]

        if map_index >= len(self.map_data.data):
            return False

        if self.map_data.data[map_index] > 50:  # Occupancy threshold
            return False

        # Check surrounding area for humanoid robot radius
        for dx in range(-int(self.robot_radius / self.map_resolution),
                       int(self.robot_radius / self.map_resolution) + 1):
            for dy in range(-int(self.robot_radius / self.map_resolution),
                           int(self.robot_radius / self.map_resolution) + 1):
                check_pos = (pos[0] + dx, pos[1] + dy)

                if (0 <= check_pos[0] < self.map_data.info.width and
                    0 <= check_pos[1] < self.map_data.info.height):
                    check_index = check_pos[1] * self.map_data.info.width + check_pos[0]

                    if check_index < len(self.map_data.data) and self.map_data.data[check_index] > 50:
                        return False

        return True

    def humanoid_cost(self, pos1, pos2):
        """Calculate cost between two positions with humanoid constraints"""
        # Euclidean distance
        dx = (pos2[0] - pos1[0]) * self.map_resolution
        dy = (pos2[1] - pos1[1]) * self.map_resolution
        distance = math.sqrt(dx*dx + dy*dy)

        # Add penalty for steps that are too long or too short
        if distance > self.step_length:
            return float('inf')  # Not traversable
        elif distance < self.min_step_length:
            return distance * 1.5  # Penalty for very short steps
        else:
            return distance

    def heuristic(self, pos1, pos2):
        """Heuristic function for A* (Euclidean distance)"""
        dx = (pos2[0] - pos1[0]) * self.map_resolution
        dy = (pos2[1] - pos1[1]) * self.map_resolution
        return math.sqrt(dx*dx + dy*dy)

    def reconstruct_path(self, current):
        """Reconstruct path from goal to start"""
        path = [current]
        while current in self.planner['came_from']:
            current = self.planner['came_from'][current]
            path.append(current)

        path.reverse()
        return path

    def world_to_map(self, point):
        """Convert world coordinates to map coordinates"""
        x = int((point.x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        y = int((point.y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return (x, y)

    def map_to_world(self, x, y):
        """Convert map coordinates to world coordinates"""
        world_x = x * self.map_data.info.resolution + self.map_data.info.origin.position.x
        world_y = y * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return (world_x, world_y)

def main(args=None):
    rclpy.init(args=args)
    planner = HumanoidGlobalPlanner()
    rclpy.spin(planner)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Footstep Planning for Bipedal Locomotion

### Advanced Footstep Planning Algorithm

Footstep planning is critical for humanoid navigation, ensuring stable and efficient locomotion:

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

class FootstepPlanner:
    def __init__(self):
        self.support_polygon = self.define_support_polygon()
        self.step_length = 0.4  # meters
        self.step_width = 0.2   # meters
        self.max_step_angle = 30  # degrees

    def define_support_polygon(self):
        """Define the support polygon for humanoid stability"""
        # Simplified support polygon (convex hull of feet when standing)
        # This represents the area where the center of mass must be kept
        polygon = np.array([
            [-0.1, -0.1],  # front left
            [0.2, -0.1],   # front right
            [0.2, 0.1],    # back right
            [-0.1, 0.1]    # back left
        ])
        return polygon

    def plan_footsteps(self, path, start_pose, robot_state):
        """Plan footsteps along the global path"""
        footsteps = []

        # Start with current position
        current_left_foot = robot_state['left_foot_pose']
        current_right_foot = robot_state['right_foot_pose']
        support_foot = 'left'  # Start with left foot as support

        # Plan footsteps along the path
        for i in range(1, len(path)):
            target_pos = path[i]

            # Determine next foot position based on path direction
            if i > 0:
                direction = np.array([target_pos[0] - path[i-1][0],
                                     target_pos[1] - path[i-1][1]])
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1, 0])  # Default direction

            # Calculate next step position
            step_pos = self.calculate_next_step(
                current_left_foot,
                current_right_foot,
                direction,
                support_foot
            )

            # Verify step is stable
            if self.is_step_stable(step_pos, current_left_foot, current_right_foot, support_foot):
                # Add footstep to plan
                footstep = {
                    'position': step_pos,
                    'support_foot': support_foot,
                    'timestamp': i * 0.5,  # 0.5 second per step
                    'gait_phase': 'swing' if support_foot == 'left' else 'stance'
                }
                footsteps.append(footstep)

                # Update support foot
                if support_foot == 'left':
                    current_left_foot = step_pos
                    support_foot = 'right'
                else:
                    current_right_foot = step_pos
                    support_foot = 'left'

        return footsteps

    def calculate_next_step(self, left_foot, right_foot, direction, support_foot):
        """Calculate next step position based on current state and direction"""
        # Calculate desired step direction
        if support_foot == 'left':
            # Right foot should step forward
            step_direction = direction
        else:
            # Left foot should step forward
            step_direction = direction

        # Calculate step position
        if support_foot == 'left':
            # Step from right foot
            base_pos = np.array(right_foot)
        else:
            # Step from left foot
            base_pos = np.array(left_foot)

        # Apply step offset
        step_offset = self.step_length * step_direction
        next_pos = base_pos + step_offset

        # Adjust for step width
        if support_foot == 'left':
            # Right foot should be slightly to the right
            lateral_offset = np.array([-step_direction[1], step_direction[0]]) * self.step_width/2
        else:
            # Left foot should be slightly to the left
            lateral_offset = np.array([step_direction[1], -step_direction[0]]) * self.step_width/2

        next_pos += lateral_offset

        return next_pos

    def is_step_stable(self, step_pos, left_foot, right_foot, support_foot):
        """Check if the step maintains stability"""
        # Calculate center of mass position after step
        if support_foot == 'left':
            # After step, right foot becomes support
            support_positions = [left_foot, step_pos]
        else:
            # After step, left foot becomes support
            support_positions = [step_pos, right_foot]

        # Calculate support polygon
        support_polygon = self.calculate_support_polygon(support_positions)

        # Calculate center of mass (simplified as midpoint between feet)
        com = np.mean(support_positions, axis=0)

        # Check if COM is within support polygon
        return self.point_in_polygon(com, support_polygon)

    def calculate_support_polygon(self, support_positions):
        """Calculate support polygon from support foot positions"""
        # For bipedal robot, support polygon is convex hull of support feet
        if len(support_positions) == 2:
            # Simple case: line between feet + buffer
            p1, p2 = support_positions
            center = (np.array(p1) + np.array(p2)) / 2
            direction = np.array(p2) - np.array(p1)
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)

            # Create rectangular support polygon
            width = 0.3  # Support width
            polygon = [
                p1 + perpendicular * width/2,
                p2 + perpendicular * width/2,
                p2 - perpendicular * width/2,
                p1 - perpendicular * width/2
            ]
            return polygon
        else:
            return support_positions

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def optimize_footsteps(self, footsteps):
        """Optimize footsteps for energy efficiency and stability"""
        # This is a simplified optimization - in practice, this would involve
        # complex optimization algorithms considering dynamics, energy, and stability

        optimized_steps = []

        for i, step in enumerate(footsteps):
            # Smooth step positions to reduce energy consumption
            if i > 0 and i < len(footsteps) - 1:
                prev_step = footsteps[i-1]['position']
                next_step = footsteps[i+1]['position']

                # Apply smoothing
                smoothed_pos = 0.2 * np.array(prev_step) + \
                              0.6 * np.array(step['position']) + \
                              0.2 * np.array(next_step)

                step['position'] = smoothed_pos
            elif i == 0:
                # First step - keep original
                pass
            else:
                # Last step - keep original
                pass

            optimized_steps.append(step)

        return optimized_steps

# Usage example
footstep_planner = FootstepPlanner()
path = [(0, 0), (1, 0), (2, 0), (3, 0)]  # Example path
robot_state = {
    'left_foot_pose': [0, 0.1],
    'right_foot_pose': [0, -0.1]
}
footsteps = footstep_planner.plan_footsteps(path, (0, 0, 0), robot_state)
optimized_footsteps = footstep_planner.optimize_footsteps(footsteps)
```

## Humanoid-Specific Local Planners

### Dynamic Balance and Local Path Following

Implement local planners that maintain balance while following the global path:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformListener
import tf2_geometry_msgs
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidLocalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_local_planner')

        # Create subscribers and publishers
        self.path_sub = self.create_subscription(
            Path,
            '/humanoid_global_plan',
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize local planner
        self.current_path = None
        self.current_pose = None
        self.current_twist = None
        self.imu_data = None
        self.joint_states = None
        self.current_waypoint = 0
        self.lookahead_distance = 0.5  # meters
        self.max_linear_vel = 0.3      # m/s
        self.max_angular_vel = 0.5     # rad/s
        self.balance_threshold = 0.1   # meters from support polygon

        # Initialize balance controller
        self.balance_controller = BalanceController()

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def path_callback(self, msg):
        """Receive global path"""
        self.current_path = msg.poses
        self.current_waypoint = 0
        self.get_logger().info(f'Received path with {len(self.current_path)} waypoints')

    def odom_callback(self, msg):
        """Receive odometry data"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Receive IMU data for balance control"""
        self.imu_data = msg

    def joint_state_callback(self, msg):
        """Receive joint state data"""
        self.joint_states = msg

    def control_loop(self):
        """Main control loop for local planning"""
        if (self.current_path is None or
            self.current_pose is None or
            len(self.current_path) == 0):
            return

        # Find next waypoint based on lookahead distance
        next_waypoint = self.find_next_waypoint()

        if next_waypoint is None:
            # Reached goal
            self.stop_robot()
            return

        # Calculate desired velocity to reach next waypoint
        cmd_vel = self.calculate_velocity_to_waypoint(next_waypoint)

        # Apply balance constraints
        cmd_vel = self.apply_balance_constraints(cmd_vel)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Update waypoint if reached
        if self.is_waypoint_reached(next_waypoint):
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.current_path):
                self.stop_robot()

    def find_next_waypoint(self):
        """Find next waypoint based on lookahead distance"""
        if self.current_pose is None or self.current_waypoint >= len(self.current_path):
            return None

        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])

        # Look ahead to find waypoint at appropriate distance
        for i in range(self.current_waypoint, len(self.current_path)):
            waypoint_pos = np.array([
                self.current_path[i].pose.position.x,
                self.current_path[i].pose.position.y
            ])

            distance = np.linalg.norm(waypoint_pos - current_pos)

            if distance >= self.lookahead_distance:
                return self.current_path[i]

        # If no waypoint is far enough, return the last one
        if len(self.current_path) > 0:
            return self.current_path[-1]

        return None

    def calculate_velocity_to_waypoint(self, waypoint):
        """Calculate velocity command to reach waypoint"""
        if self.current_pose is None:
            return Twist()

        # Calculate desired direction
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])

        waypoint_pos = np.array([
            waypoint.pose.position.x,
            waypoint.pose.position.y
        ])

        direction = waypoint_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Very close to waypoint
            return Twist()

        direction = direction / distance  # Normalize

        # Calculate desired linear velocity
        linear_vel = min(self.max_linear_vel, distance * 0.5)  # Proportional to distance

        # Calculate desired angular velocity to face direction
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        desired_yaw = math.atan2(direction[1], direction[0])

        angle_diff = desired_yaw - current_yaw
        # Normalize angle difference to [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, angle_diff * 1.0))

        # Create velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        return cmd_vel

    def apply_balance_constraints(self, cmd_vel):
        """Apply balance constraints to velocity command"""
        if self.imu_data is None:
            return cmd_vel

        # Get roll and pitch from IMU
        orientation = self.imu_data.orientation
        roll, pitch, _ = self.get_euler_from_quaternion(orientation)

        # Check if robot is tilting too much
        max_tilt = 0.2  # radians

        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            # Reduce velocity to maintain balance
            cmd_vel.linear.x *= 0.5
            cmd_vel.angular.z *= 0.5

        # Apply additional balance control if needed
        if self.joint_states is not None:
            cmd_vel = self.balance_controller.adjust_for_balance(cmd_vel, self.joint_states)

        return cmd_vel

    def is_waypoint_reached(self, waypoint):
        """Check if current waypoint is reached"""
        if self.current_pose is None:
            return False

        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])

        waypoint_pos = np.array([
            waypoint.pose.position.x,
            waypoint.pose.position.y
        ])

        distance = np.linalg.norm(waypoint_pos - current_pos)
        return distance < 0.2  # 20cm tolerance

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_euler_from_quaternion(self, quat):
        """Extract Euler angles from quaternion"""
        # Convert quaternion to rotation matrix first
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler('xyz')

class BalanceController:
    def __init__(self):
        self.kp_balance = 1.0  # Proportional gain for balance
        self.kd_balance = 0.1  # Derivative gain for balance
        self.previous_tilt = 0.0

    def adjust_for_balance(self, cmd_vel, joint_states):
        """Adjust velocity command based on balance state"""
        # This is a simplified balance controller
        # In practice, this would use more sophisticated control algorithms

        # Example: adjust velocity based on joint angles
        # Look for hip or ankle joint angles that indicate balance issues
        try:
            # Find relevant joint positions
            left_hip_idx = joint_states.name.index('left_hip_joint')
            right_hip_idx = joint_states.name.index('right_hip_joint')

            left_hip_angle = joint_states.position[left_hip_idx]
            right_hip_angle = joint_states.position[right_hip_idx]

            # Calculate average hip angle (indication of balance)
            avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

            # If robot is tilting, reduce velocity
            if abs(avg_hip_angle) > 0.1:  # 0.1 rad tolerance
                reduction_factor = max(0.1, 1.0 - abs(avg_hip_angle) * self.kp_balance)
                cmd_vel.linear.x *= reduction_factor
                cmd_vel.angular.z *= reduction_factor

        except ValueError:
            # Joint names not found, skip balance adjustment
            pass

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    local_planner = HumanoidLocalPlanner()
    rclpy.spin(local_planner)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Humanoid Navigation Strategies

### Terrain Adaptation and Dynamic Obstacle Avoidance

Implement navigation strategies that adapt to terrain and avoid dynamic obstacles:

```python
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import math

class AdvancedHumanoidNavigator:
    def __init__(self):
        self.terrain_classifier = TerrainClassifier()
        self.obstacle_predictor = ObstaclePredictor()
        self.adaptive_planner = AdaptivePlanner()

    def plan_adaptive_path(self, start, goal, dynamic_obstacles, terrain_map):
        """Plan path that adapts to terrain and avoids dynamic obstacles"""
        # Classify terrain along potential paths
        terrain_info = self.terrain_classifier.analyze_terrain(terrain_map, start, goal)

        # Predict future positions of dynamic obstacles
        predicted_obstacles = self.obstacle_predictor.predict_obstacles(dynamic_obstacles)

        # Plan path considering both terrain and obstacles
        path = self.adaptive_planner.plan_path(
            start, goal, terrain_info, predicted_obstacles
        )

        return path

    def adjust_gait_for_terrain(self, current_pose, terrain_type):
        """Adjust gait parameters based on terrain type"""
        gait_params = {
            'step_length': 0.4,      # Default step length
            'step_height': 0.05,     # Default step height
            'step_duration': 0.8,    # Default step duration
            'com_height': 0.8        # Default center of mass height
        }

        if terrain_type == 'rough':
            gait_params.update({
                'step_length': 0.3,      # Shorter steps on rough terrain
                'step_height': 0.1,      # Higher steps to clear obstacles
                'step_duration': 1.0,    # Slower steps for stability
                'com_height': 0.75       # Lower COM for stability
            })
        elif terrain_type == 'slippery':
            gait_params.update({
                'step_length': 0.2,      # Very short steps
                'step_height': 0.02,     # Low steps to maintain contact
                'step_duration': 1.2,    # Slow and careful
                'com_height': 0.7        # Lower for better stability
            })
        elif terrain_type == 'stairs':
            gait_params.update({
                'step_length': 0.3,      # Adjusted for stair climbing
                'step_height': 0.15,     # Higher to clear steps
                'step_duration': 1.0,    # Careful timing
                'com_height': 0.85       # Slightly higher for stair navigation
            })

        return gait_params

class TerrainClassifier:
    def __init__(self):
        self.terrain_types = {
            'flat': {'friction': 0.8, 'roughness': 0.01, 'traversability': 1.0},
            'rough': {'friction': 0.7, 'roughness': 0.1, 'traversability': 0.6},
            'slippery': {'friction': 0.3, 'roughness': 0.02, 'traversability': 0.4},
            'stairs': {'friction': 0.7, 'roughness': 0.05, 'traversability': 0.5}
        }

    def analyze_terrain(self, terrain_map, start, goal):
        """Analyze terrain characteristics along path"""
        # Sample terrain along the path
        path_samples = self.sample_path(terrain_map, start, goal)

        terrain_analysis = {
            'segments': [],
            'overall_type': 'flat',
            'risk_factors': {
                'slip': 0.0,
                'stability': 1.0,
                'obstacle_density': 0.0
            }
        }

        for i, point in enumerate(path_samples):
            terrain_type = self.classify_point(terrain_map, point)
            terrain_analysis['segments'].append({
                'point': point,
                'type': terrain_type,
                'properties': self.terrain_types[terrain_type]
            })

        # Determine overall terrain type based on majority
        types = [seg['type'] for seg in terrain_analysis['segments']]
        terrain_analysis['overall_type'] = max(set(types), key=types.count)

        return terrain_analysis

    def sample_path(self, terrain_map, start, goal, num_samples=20):
        """Sample points along a path"""
        # Linear interpolation between start and goal
        path = []
        for i in range(num_samples + 1):
            t = i / num_samples
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append((x, y))

        return path

    def classify_point(self, terrain_map, point):
        """Classify terrain type at a specific point"""
        # This would involve analyzing the terrain_map data structure
        # For now, return a simple classification based on heuristics
        x, y = point

        # Example: classify based on height variation in neighborhood
        neighborhood_variance = self.calculate_height_variance(terrain_map, x, y)

        if neighborhood_variance < 0.01:
            return 'flat'
        elif neighborhood_variance < 0.05:
            return 'rough'
        else:
            return 'rough'  # Very rough terrain

    def calculate_height_variance(self, terrain_map, x, y, radius=1.0):
        """Calculate height variance in neighborhood of point"""
        # This is a simplified implementation
        # In practice, this would analyze the actual terrain data
        return 0.02  # Placeholder value

class ObstaclePredictor:
    def __init__(self):
        self.prediction_horizon = 5.0  # seconds
        self.time_steps = 10

    def predict_obstacles(self, dynamic_obstacles):
        """Predict future positions of dynamic obstacles"""
        predicted_obstacles = []

        for obstacle in dynamic_obstacles:
            predicted_trajectory = []

            # Predict trajectory using constant velocity model
            dt = self.prediction_horizon / self.time_steps
            for t in range(self.time_steps + 1):
                future_time = t * dt
                future_pos = (
                    obstacle['position'][0] + obstacle['velocity'][0] * future_time,
                    obstacle['position'][1] + obstacle['velocity'][1] * future_time
                )
                predicted_trajectory.append({
                    'time': future_time,
                    'position': future_pos,
                    'confidence': 0.9 - (0.1 * t / self.time_steps)  # Confidence decreases over time
                })

            predicted_obstacles.append({
                'id': obstacle['id'],
                'trajectory': predicted_trajectory,
                'radius': obstacle['radius']
            })

        return predicted_obstacles

class AdaptivePlanner:
    def __init__(self):
        self.replan_threshold = 0.5  # meters
        self.smoothing_factor = 0.8

    def plan_path(self, start, goal, terrain_info, predicted_obstacles):
        """Plan adaptive path considering terrain and obstacles"""
        # Use a modified A* algorithm that considers terrain and dynamic obstacles

        # Create a cost map that includes terrain costs and predicted obstacle costs
        cost_map = self.create_adaptive_cost_map(start, goal, terrain_info, predicted_obstacles)

        # Plan path using the cost map
        path = self.plan_with_cost_map(start, goal, cost_map)

        # Smooth the path for humanoid locomotion
        smoothed_path = self.smooth_path(path)

        return smoothed_path

    def create_adaptive_cost_map(self, start, goal, terrain_info, predicted_obstacles):
        """Create cost map that adapts to terrain and obstacles"""
        # This would create a grid-based cost map
        # For simplicity, return a placeholder
        return np.ones((100, 100))  # Placeholder cost map

    def plan_with_cost_map(self, start, goal, cost_map):
        """Plan path using cost map"""
        # Simplified path planning - in practice, this would use A* or D* Lite
        path = [start, goal]  # Placeholder
        return path

    def smooth_path(self, path):
        """Smooth path for humanoid locomotion"""
        if len(path) < 3:
            return path

        # Convert to numpy arrays for processing
        points = np.array(path)

        # Apply smoothing using interpolation
        if len(points) >= 4:
            # Use cubic spline for smoother path
            t = np.arange(len(points))
            fx = interp1d(t, points[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
            fy = interp1d(t, points[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')

            # Generate more intermediate points
            t_new = np.linspace(0, len(points)-1, len(points)*3)
            smooth_x = fx(t_new)
            smooth_y = fy(t_new)

            smoothed_path = [(x, y) for x, y in zip(smooth_x, smooth_y)]
        else:
            # For shorter paths, use simple linear interpolation
            smoothed_path = []
            for i in range(len(path)-1):
                start_point = path[i]
                end_point = path[i+1]

                # Add intermediate points
                for j in range(3):
                    t = j / 3.0
                    x = start_point[0] + t * (end_point[0] - start_point[0])
                    y = start_point[1] + t * (end_point[1] - start_point[1])
                    smoothed_path.append((x, y))

            smoothed_path.append(path[-1])  # Add final point

        return smoothed_path

# Usage example
navigator = AdvancedHumanoidNavigator()
start = (0, 0)
goal = (10, 10)
dynamic_obstacles = [
    {'id': 1, 'position': (5, 5), 'velocity': (0.1, 0.1), 'radius': 0.5}
]
terrain_map = {}  # Placeholder for terrain map

path = navigator.plan_adaptive_path(start, goal, dynamic_obstacles, terrain_map)
gait_params = navigator.adjust_gait_for_terrain((0, 0, 0), 'flat')
```

## Practical Bipedal Navigation Exercises

### Exercise 1: Indoor Navigation with Balance Control

Implement a complete indoor navigation system for a bipedal robot:

1. Set up Nav2 with humanoid-specific planners in Isaac Sim
2. Configure behavior trees for complex navigation tasks
3. Implement footstep planning for stable locomotion
4. Test navigation performance with balance control in various indoor scenarios

### Exercise 2: Terrain Adaptation

Create a navigation system that adapts to different terrains:

1. Implement terrain classification algorithms
2. Adjust gait parameters based on terrain type
3. Test navigation on flat, rough, and slippery surfaces
4. Evaluate stability and efficiency metrics

### Exercise 3: Dynamic Obstacle Avoidance

Extend navigation for dynamic environments:

1. Implement obstacle prediction algorithms
2. Plan adaptive paths around moving obstacles
3. Test collision avoidance in crowded environments
4. Evaluate navigation success rates and safety metrics

## Integration with Isaac Sim and Isaac ROS

### Complete Navigation Pipeline

Integrate Nav2 with Isaac Sim and Isaac ROS for comprehensive humanoid navigation:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, Imu, JointState
from visualization_msgs.msg import MarkerArray

class IntegratedHumanoidNavigator(Node):
    def __init__(self):
        super().__init__('integrated_humanoid_navigator')

        # Initialize Isaac Sim interface
        self.isaac_sim_interface = self.initialize_isaac_sim()

        # Initialize Isaac ROS perception
        self.perception_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.perception_callback, 10
        )

        # Initialize IMU and joint state subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Initialize Nav2 action client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize visualization publishers
        self.footstep_pub = self.create_publisher(MarkerArray, '/planned_footsteps', 10)

        # Initialize navigation state
        self.current_goal = None
        self.navigation_active = False

        self.get_logger().info('Integrated Humanoid Navigator initialized')

    def initialize_isaac_sim(self):
        """Initialize interface to Isaac Sim"""
        # Set up Isaac Sim connection and configuration
        # This would involve setting up the Isaac Sim environment and sim interface
        return {'connected': True, 'environment': 'default'}

    def perception_callback(self, msg):
        """Process perception data from Isaac ROS"""
        # Process camera data for obstacle detection and mapping
        # This integrates with the VSLAM system from previous lessons
        pass

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Use IMU data for real-time balance adjustment
        pass

    def joint_state_callback(self, msg):
        """Process joint state data for gait control"""
        # Use joint state data for coordinated locomotion
        pass

    def send_navigation_goal(self, pose):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.nav_to_pose_client.wait_for_server()
        self._send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Navigation feedback: {feedback.current_pose}')

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        self.navigation_active = False

def main(args=None):
    rclpy.init(args=args)
    navigator = IntegratedHumanoidNavigator()
    rclpy.spin(navigator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Humanoid Navigation

### Common Issues and Solutions

**Balance Loss During Navigation:**
- **Problem**: Robot loses balance during path following
- **Solution**: Adjust balance controller gains and implement proactive balance adjustments

**Footstep Planning Failures:**
- **Problem**: Robot cannot find stable footsteps in complex terrain
- **Solution**: Implement fallback strategies and terrain adaptation algorithms

**Path Planning Inefficiency:**
- **Problem**: Long computation times for path planning
- **Solution**: Use hierarchical planning and caching of computed paths

**Dynamic Obstacle Conflicts:**
- **Problem**: Robot collides with moving obstacles
- **Solution**: Improve obstacle prediction and implement reactive avoidance

### Performance Validation

Validate navigation performance using standardized metrics:

```python
def validate_humanoid_navigation_performance(traj_data):
    """Validate humanoid navigation performance"""
    metrics = {
        'path_efficiency': 0.0,      # Ratio of optimal path to actual path
        'balance_stability': 0.0,    # Average balance margin
        'success_rate': 0.0,         # Percentage of successful navigations
        'time_efficiency': 0.0,      # Time to goal vs optimal time
        'energy_efficiency': 0.0     # Energy consumption per meter
    }

    # Calculate path efficiency
    if traj_data['optimal_path_length'] > 0:
        metrics['path_efficiency'] = traj_data['optimal_path_length'] / traj_data['actual_path_length']

    # Calculate balance stability (average distance from support polygon)
    if 'balance_data' in traj_data:
        balance_distances = [abs(d) for d in traj_data['balance_data']]
        metrics['balance_stability'] = 1.0 - (sum(balance_distances) / len(balance_distances))

    # Calculate success rate
    if 'navigation_attempts' in traj_data and traj_data['navigation_attempts'] > 0:
        metrics['success_rate'] = traj_data['successful_navigations'] / traj_data['navigation_attempts']

    # Calculate time efficiency
    if traj_data['optimal_time'] > 0:
        metrics['time_efficiency'] = traj_data['optimal_time'] / traj_data['actual_time']

    return metrics

# Example usage
validation_metrics = validate_humanoid_navigation_performance({
    'optimal_path_length': 10.0,
    'actual_path_length': 12.0,
    'balance_data': [0.02, 0.05, 0.03, 0.04, 0.01],
    'navigation_attempts': 10,
    'successful_navigations': 9,
    'optimal_time': 50.0,
    'actual_time': 60.0
})

print(f"Navigation Performance Metrics: {validation_metrics}")
```

## Summary

This lesson covered comprehensive Nav2 navigation for bipedal humanoid robots with detailed humanoid-specific navigation strategies and advanced path planning algorithms. You learned how to implement specialized planners for humanoid robots, plan footsteps for stable locomotion, maintain balance during navigation, and adapt to various terrains and dynamic environments.

The Nav2 system implemented here provides sophisticated path planning capabilities specifically designed for the unique challenges of bipedal robot navigation, including balance constraints, gait patterns, and stability requirements. The integration with Isaac Sim and Isaac ROS creates a complete navigation pipeline for humanoid robots that can be tested and validated in simulation before deployment to real robots.

The next lesson will cover integrated lab exercises that combine all components learned in this module, providing hands-on experience with the complete AI-robot brain system.