---
sidebar_position: 3
title: "LLM Planning Pipeline"
---

# LLM Planning Pipeline

## Introduction to LLM-Based Task Planning

Large Language Models (LLMs) have revolutionized how we approach task planning in robotics. Instead of hard-coded behavior trees or finite state machines, LLMs can interpret natural language commands and generate executable action sequences. This chapter explores how to build an LLM planning pipeline that converts natural language commands like "Clean the room" into ROS 2 action sequences.

## Architecture of the LLM Planning System

The LLM planning pipeline consists of several components:

1. **Input Processing**: Receiving and preprocessing voice commands from the Whisper system
2. **Context Understanding**: Interpreting the command in the context of the robot's capabilities and environment
3. **Task Decomposition**: Breaking down high-level commands into executable subtasks
4. **Action Generation**: Converting subtasks into ROS 2 action calls
5. **Validation and Safety**: Ensuring generated actions are safe and executable
6. **Execution Monitoring**: Tracking task progress and adapting to changes

## Setting Up the LLM Interface

### Installation and Dependencies

```bash
pip install openai  # For OpenAI API
# OR
pip install anthropic  # For Anthropic API
# OR
pip install transformers torch  # For local models
```

### Basic LLM Interface

```python
import openai
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class RobotAction:
    """Represents a single robot action."""
    action_type: str  # "navigation", "manipulation", "perception", etc.
    parameters: Dict[str, Any]
    description: str
    estimated_duration: float  # in seconds

@dataclass
class PlanningResult:
    """Result of the planning process."""
    success: bool
    actions: List[RobotAction]
    reasoning: str
    error_message: Optional[str] = None

class LLMPlanner:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """
        Initialize the LLM Planner.

        Args:
            api_key: API key for the LLM service
            model: Model to use for planning
        """
        openai.api_key = api_key
        self.model = model
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """
You are an expert robotic task planner. Your job is to convert natural language commands into sequences of robot actions.

Robot capabilities:
- Navigation: Move to locations (kitchen, bedroom, office, etc.)
- Manipulation: Pick up, place, grasp objects
- Perception: Look for objects, scan environment
- Interaction: Open doors, press buttons

Action format:
{
    "actions": [
        {
            "action_type": "navigation|manipulation|perception|interaction",
            "parameters": {
                "target_location": "kitchen",
                "target_object": "cup",
                "description": "Move to kitchen"
            },
            "description": "Action description for humans",
            "estimated_duration": 10.0
        }
    ],
    "reasoning": "Brief explanation of the planning process"
}

Rules:
1. Always return valid JSON
2. Ensure actions are executable by the robot
3. Include safety checks in your planning
4. Break complex tasks into simple, sequential actions
5. Consider the robot's current state and environment
"""

    def plan_task(self, command: str, robot_state: Dict[str, Any]) -> PlanningResult:
        """
        Plan a task based on a natural language command.

        Args:
            command: Natural language command
            robot_state: Current state of the robot

        Returns:
            PlanningResult with action sequence
        """
        user_prompt = f"""
Command: {command}

Current robot state: {json.dumps(robot_state, indent=2)}

Generate a sequence of actions to complete this task.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent planning
                max_tokens=1000,
                response_format={"type": "json_object"}  # Request JSON output
            )

            response_text = response.choices[0].message.content
            response_json = json.loads(response_text)

            actions = []
            for action_data in response_json["actions"]:
                action = RobotAction(
                    action_type=action_data["action_type"],
                    parameters=action_data["parameters"],
                    description=action_data["description"],
                    estimated_duration=action_data.get("estimated_duration", 5.0)
                )
                actions.append(action)

            return PlanningResult(
                success=True,
                actions=actions,
                reasoning=response_json["reasoning"]
            )

        except json.JSONDecodeError:
            return PlanningResult(
                success=False,
                actions=[],
                reasoning="",
                error_message="Failed to parse LLM response as JSON"
            )
        except Exception as e:
            return PlanningResult(
                success=False,
                actions=[],
                reasoning="",
                error_message=f"Error during planning: {str(e)}"
            )
```

## ROS 2 Action Generation

The LLM planning system needs to convert abstract actions into concrete ROS 2 action calls:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
from typing import List
from .llm_planner import LLMPlanner, RobotAction

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize LLM planner
        self.llm_planner = LLMPlanner(api_key=self.get_parameter('openai_api_key').value)

        # Subscribers
        self.voice_command_sub = self.create_subscription(
            String,
            'voice_commands',
            self.voice_command_callback,
            10
        )

        # Publishers
        self.action_sequence_pub = self.create_publisher(
            String,
            'action_sequence',
            10
        )

        # Action clients
        self.navigation_client = ActionClient(self, MoveBaseAction, 'move_base')

        self.get_logger().info('LLM Planning Node initialized')

    def voice_command_callback(self, msg: String):
        """Process incoming voice commands."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Get current robot state (simplified)
        robot_state = {
            "current_location": "office",
            "battery_level": 0.85,
            "gripper_status": "open",
            "camera_status": "active",
            "available_actions": ["navigation", "manipulation", "perception"]
        }

        # Plan the task using LLM
        planning_result = self.llm_planner.plan_task(command, robot_state)

        if planning_result.success:
            self.get_logger().info(f'Planning successful: {len(planning_result.actions)} actions generated')

            # Convert to ROS 2 message and publish
            action_sequence_msg = String()
            action_sequence_msg.data = json.dumps({
                "command": command,
                "actions": [
                    {
                        "action_type": action.action_type,
                        "parameters": action.parameters,
                        "description": action.description,
                        "estimated_duration": action.estimated_duration
                    } for action in planning_result.actions
                ],
                "reasoning": planning_result.reasoning
            })

            self.action_sequence_pub.publish(action_sequence_msg)
        else:
            self.get_logger().error(f'Planning failed: {planning_result.error_message}')

    def execute_action_sequence(self, action_sequence: List[RobotAction]):
        """Execute a sequence of actions."""
        for i, action in enumerate(action_sequence):
            self.get_logger().info(f'Executing action {i+1}/{len(action_sequence)}: {action.description}')

            success = self.execute_single_action(action)
            if not success:
                self.get_logger().error(f'Action failed: {action.description}')
                break

    def execute_single_action(self, action: RobotAction) -> bool:
        """Execute a single robot action."""
        if action.action_type == "navigation":
            return self.execute_navigation_action(action.parameters)
        elif action.action_type == "manipulation":
            return self.execute_manipulation_action(action.parameters)
        elif action.action_type == "perception":
            return self.execute_perception_action(action.parameters)
        else:
            self.get_logger().warn(f'Unknown action type: {action.action_type}')
            return False

    def execute_navigation_action(self, params: Dict) -> bool:
        """Execute navigation action."""
        try:
            # Create navigation goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = self.get_clock().now().to_msg()

            # Set target position based on location name
            location = params.get("target_location", "unknown")
            position = self.get_location_position(location)

            if position:
                goal.target_pose.pose.position.x = position[0]
                goal.target_pose.pose.position.y = position[1]
                goal.target_pose.pose.orientation.w = 1.0  # Simple orientation

                # Send goal to navigation system
                self.navigation_client.wait_for_server()
                future = self.navigation_client.send_goal_async(goal)

                # Wait for result (simplified)
                rclpy.spin_until_future_complete(self, future)

                return future.result().status == 3  # Goal reached
            else:
                self.get_logger().error(f'Unknown location: {location}')
                return False

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def get_location_position(self, location: str) -> tuple:
        """Get position for a named location."""
        # In a real system, this would come from a map or localization system
        locations = {
            "kitchen": (1.0, 2.0),
            "bedroom": (3.0, 1.0),
            "office": (0.0, 0.0),
            "living_room": (2.0, 3.0)
        }
        return locations.get(location, None)
```

## Advanced Planning Features

### Context-Aware Planning

```python
class ContextAwarePlanner(LLMPlanner):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        super().__init__(api_key, model)
        self.environment_context = {}
        self.robot_capabilities = {}
        self.task_history = []

    def update_environment_context(self, sensor_data: Dict[str, Any]):
        """Update the environment context with current sensor data."""
        self.environment_context.update(sensor_data)

    def update_robot_capabilities(self, capabilities: Dict[str, Any]):
        """Update robot capabilities."""
        self.robot_capabilities.update(capabilities)

    def plan_task_with_context(self, command: str, additional_context: Dict[str, Any] = None) -> PlanningResult:
        """Plan a task with additional context information."""
        # Combine robot state with environment context
        full_state = {
            "robot_state": {
                "current_location": "office",
                "battery_level": 0.85,
                "gripper_status": "open",
                "camera_status": "active",
                "available_actions": ["navigation", "manipulation", "perception"]
            },
            "environment_context": self.environment_context,
            "robot_capabilities": self.robot_capabilities,
            "task_history": self.task_history[-5:]  # Last 5 tasks
        }

        if additional_context:
            full_state.update(additional_context)

        return self.plan_task(command, full_state)

    def validate_plan(self, plan: PlanningResult) -> PlanningResult:
        """Validate the generated plan for safety and feasibility."""
        if not plan.success:
            return plan

        validated_actions = []
        for action in plan.actions:
            # Check if action is safe
            if not self.is_action_safe(action):
                plan.success = False
                plan.error_message = f"Unsafe action detected: {action.description}"
                return plan

            # Check if action is feasible
            if not self.is_action_feasible(action):
                plan.success = False
                plan.error_message = f"Infeasible action: {action.description}"
                return plan

            validated_actions.append(action)

        plan.actions = validated_actions
        return plan

    def is_action_safe(self, action: RobotAction) -> bool:
        """Check if an action is safe to execute."""
        # Safety checks based on action type
        if action.action_type == "navigation":
            target_location = action.parameters.get("target_location")
            # Check if location is accessible
            if target_location in ["restricted_area", "danger_zone"]:
                return False
        elif action.action_type == "manipulation":
            target_object = action.parameters.get("target_object")
            # Check if object is safe to manipulate
            if target_object in ["fragile", "dangerous"]:
                return False

        return True

    def is_action_feasible(self, action: RobotAction) -> bool:
        """Check if an action is feasible with current robot state."""
        # Check if robot has required capabilities
        required_capability = f"{action.action_type}_capability"
        if required_capability not in self.robot_capabilities:
            return False

        # Check resource constraints
        if action.action_type == "navigation":
            battery_needed = action.estimated_duration * 0.01  # Simplified calculation
            if self.robot_capabilities.get("battery_level", 1.0) < battery_needed:
                return False

        return True
```

### Multi-Step Task Decomposition

```python
class HierarchicalPlanner(LLMPlanner):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        super().__init__(api_key, model)
        self.subtask_templates = self._load_subtask_templates()

    def _load_subtask_templates(self) -> Dict[str, List[RobotAction]]:
        """Load templates for common subtasks."""
        return {
            "clean_room": [
                RobotAction("perception", {"task": "scan_room"}, "Scan room for objects", 5.0),
                RobotAction("navigation", {"target_location": "kitchen"}, "Move to kitchen", 10.0),
                RobotAction("manipulation", {"task": "collect_items"}, "Collect items", 30.0),
                RobotAction("navigation", {"target_location": "office"}, "Return to office", 10.0)
            ],
            "fetch_item": [
                RobotAction("perception", {"target_object": "item"}, "Locate item", 10.0),
                RobotAction("navigation", {"target_location": "item_location"}, "Navigate to item", 15.0),
                RobotAction("manipulation", {"action": "grasp", "target_object": "item"}, "Grasp item", 5.0),
                RobotAction("navigation", {"target_location": "delivery_location"}, "Return to delivery location", 15.0)
            ]
        }

    def decompose_complex_task(self, command: str, robot_state: Dict[str, Any]) -> PlanningResult:
        """Decompose complex tasks into subtasks."""
        # First, determine the high-level task
        high_level_task = self._identify_high_level_task(command)

        if high_level_task in self.subtask_templates:
            # Use template for known task
            template_actions = self.subtask_templates[high_level_task]
            return PlanningResult(
                success=True,
                actions=template_actions,
                reasoning=f"Using template for {high_level_task} task"
            )
        else:
            # Use LLM for novel tasks
            return self.plan_task(command, robot_state)

    def _identify_high_level_task(self, command: str) -> str:
        """Identify the high-level task from the command."""
        command_lower = command.lower()

        if "clean" in command_lower:
            return "clean_room"
        elif "fetch" in command_lower or "bring" in command_lower or "get" in command_lower:
            return "fetch_item"
        else:
            # For other commands, return a generic identifier
            return "custom_task"
```

## Safety and Validation

### Action Validation Pipeline

```python
class ActionValidator:
    def __init__(self):
        self.safety_rules = [
            self._check_battery_level,
            self._check_obstacle_avoidance,
            self._check_manipulation_limits,
            self._check_navigation_bounds
        ]

    def validate_action_sequence(self, actions: List[RobotAction], robot_state: Dict[str, Any]) -> tuple:
        """
        Validate an action sequence for safety and feasibility.

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        for i, action in enumerate(actions):
            for rule in self.safety_rules:
                rule_violations = rule(action, robot_state)
                violations.extend([f"Action {i}: {v}" for v in rule_violations])

        return len(violations) == 0, violations

    def _check_battery_level(self, action: RobotAction, robot_state: Dict[str, Any]) -> List[str]:
        """Check if battery level is sufficient for the action."""
        battery_level = robot_state.get("battery_level", 1.0)
        required_battery = action.estimated_duration * 0.01  # Simplified calculation

        if battery_level < required_battery:
            return [f"Insufficient battery for action (need {required_battery:.2f}, have {battery_level:.2f})"]
        return []

    def _check_obstacle_avoidance(self, action: RobotAction, robot_state: Dict[str, Any]) -> List[str]:
        """Check if navigation action avoids known obstacles."""
        if action.action_type != "navigation":
            return []

        target_location = action.parameters.get("target_location")
        # In a real system, check against obstacle map
        # For now, just return empty list
        return []

    def _check_manipulation_limits(self, action: RobotAction, robot_state: Dict[str, Any]) -> List[str]:
        """Check if manipulation action is within robot capabilities."""
        if action.action_type != "manipulation":
            return []

        # Check if gripper is available
        gripper_status = robot_state.get("gripper_status", "closed")
        if gripper_status not in ["open", "ready"] and "grasp" in action.description.lower():
            return ["Gripper not in proper state for grasping"]

        return []

    def _check_navigation_bounds(self, action: RobotAction, robot_state: Dict[str, Any]) -> List[str]:
        """Check if navigation is within operational bounds."""
        if action.action_type != "navigation":
            return []

        target_location = action.parameters.get("target_location")
        # Check if location is in known map
        valid_locations = ["kitchen", "bedroom", "office", "living_room"]
        if target_location not in valid_locations:
            return [f"Unknown or unreachable location: {target_location}"]

        return []
```

## Integration with ROS 2 Action Servers

```python
class ROS2ActionExecutor:
    def __init__(self, node: Node):
        self.node = node
        self.action_clients = {}
        self.setup_action_clients()

    def setup_action_clients(self):
        """Setup ROS 2 action clients for different robot capabilities."""
        # Navigation action client
        from move_base_msgs.action import MoveBase
        self.action_clients['navigation'] = ActionClient(self.node, MoveBase, 'navigate_to_pose')

        # Manipulation action client
        from control_msgs.action import FollowJointTrajectory
        self.action_clients['manipulation'] = ActionClient(self.node, FollowJointTrajectory, 'manipulator_controller/follow_joint_trajectory')

        # Perception action client
        from sensor_msgs.msg import Image
        self.perception_sub = self.node.create_subscription(Image, 'camera/image_raw', self.perception_callback, 10)

    def execute_action(self, action: RobotAction) -> bool:
        """Execute a robot action using ROS 2 action servers."""
        action_type = action.action_type

        if action_type not in self.action_clients:
            self.node.get_logger().error(f'No action client for type: {action_type}')
            return False

        client = self.action_clients[action_type]

        if action_type == 'navigation':
            return self.execute_navigation_action(client, action.parameters)
        elif action_type == 'manipulation':
            return self.execute_manipulation_action(client, action.parameters)
        else:
            self.node.get_logger().warn(f'Action type {action_type} not implemented')
            return False

    def execute_navigation_action(self, client, params: Dict) -> bool:
        """Execute navigation action."""
        from move_base_msgs.action import MoveBase
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Header

        goal = MoveBase.Goal()
        goal.target_pose.header = Header()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = self.node.get_clock().now().to_msg()

        # Convert location name to coordinates
        location_coords = self.get_coordinates_for_location(params.get("target_location"))
        if location_coords:
            goal.target_pose.pose.position.x = location_coords[0]
            goal.target_pose.pose.position.y = location_coords[1]
            goal.target_pose.pose.orientation.w = 1.0

            # Send goal
            client.wait_for_server()
            future = client.send_goal_async(goal)

            # Wait for result with timeout
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=action.estimated_duration)

            result = future.result()
            return result is not None and result.status == 3  # SUCCEEDED
        else:
            return False

    def get_coordinates_for_location(self, location: str) -> tuple:
        """Get coordinates for a named location."""
        # This would typically come from a map service
        locations = {
            "kitchen": (2.0, 1.0),
            "bedroom": (0.5, 2.5),
            "office": (0.0, 0.0),
            "living_room": (1.5, 2.0)
        }
        return locations.get(location, (0.0, 0.0))

    def execute_manipulation_action(self, client, params: Dict) -> bool:
        """Execute manipulation action."""
        # Implementation would depend on specific manipulator
        # This is a simplified example
        target_object = params.get("target_object")
        action = params.get("action", "move")

        if action == "grasp":
            # Execute grasp action
            return self.perform_grasp(target_object)
        elif action == "place":
            # Execute place action
            return self.perform_place(params.get("target_location"))
        else:
            return False

    def perform_grasp(self, target_object: str) -> bool:
        """Perform grasping action."""
        # Simplified implementation
        self.node.get_logger().info(f'Attempting to grasp {target_object}')
        # In a real system, this would involve complex manipulation planning
        return True

    def perform_place(self, target_location: str) -> bool:
        """Perform placing action."""
        # Simplified implementation
        self.node.get_logger().info(f'Attempting to place object at {target_location}')
        return True

    def perception_callback(self, msg):
        """Handle perception data."""
        # Process perception data for action validation
        pass
```

## Example: "Clean the Room" Task

Let's see how the complete system would handle a "Clean the room" command:

```python
def example_clean_room_task():
    """Example of processing a 'Clean the room' command."""

    # Initialize components
    llm_planner = LLMPlanner(api_key="your-api-key")
    validator = ActionValidator()
    executor = ROS2ActionExecutor(node=None)  # Would be initialized with actual ROS2 node

    # Simulate voice command
    voice_command = "Clean the room"

    # Robot state
    robot_state = {
        "current_location": "office",
        "battery_level": 0.85,
        "gripper_status": "open",
        "camera_status": "active",
        "available_actions": ["navigation", "manipulation", "perception"]
    }

    # Plan the task
    planning_result = llm_planner.plan_task(voice_command, robot_state)

    if planning_result.success:
        print(f"Generated {len(planning_result.actions)} actions:")
        for i, action in enumerate(planning_result.actions):
            print(f"  {i+1}. {action.action_type}: {action.description}")

        # Validate the plan
        is_valid, violations = validator.validate_action_sequence(planning_result.actions, robot_state)

        if is_valid:
            print("Plan is valid and safe to execute")
            # Execute the plan (in a real system)
            # executor.execute_action_sequence(planning_result.actions)
        else:
            print("Plan has violations:")
            for violation in violations:
                print(f"  - {violation}")
    else:
        print(f"Planning failed: {planning_result.error_message}")

# Example output might be:
# Generated 4 actions:
#   1. perception: Scan room for objects
#   2. navigation: Move to kitchen
#   3. manipulation: Collect items
#   4. navigation: Return to office
# Plan is valid and safe to execute
```

## Summary

In this chapter, we've explored how to build an LLM planning pipeline that converts natural language commands into executable ROS 2 actions. We've covered:

- Setting up the LLM interface for task planning
- Creating ROS 2 action generators
- Implementing context-aware planning
- Adding safety and validation checks
- Building hierarchical task decomposition
- Integrating with ROS 2 action servers

The LLM planning pipeline serves as the brain of our VLA system, interpreting human commands and generating executable action sequences. In the next chapter, we'll explore the perception module that provides the visual understanding needed for the robot to interact with its environment.