---
sidebar_position: 6
title: "Capstone Lab: Autonomous Humanoid Performing a Task"
---

# Capstone Lab: Autonomous Humanoid Performing a Task

## Overview

In this capstone lab, we'll implement a complete Vision-Language-Action (VLA) system that allows a humanoid robot to understand and execute natural language commands. This lab will integrate all components we've developed in previous chapters to create a functional autonomous system.

## Lab Objectives

By the end of this lab, you will be able to:

1. Deploy the complete VLA system on a humanoid robot platform
2. Execute end-to-end tasks from voice command to robot action
3. Troubleshoot common integration issues
4. Evaluate system performance and reliability
5. Extend the system with custom behaviors

## Prerequisites

Before starting this lab, ensure you have:

- Completed all previous chapters in this module
- A humanoid robot platform (simulated or physical) with ROS 2 support
- A camera system for perception
- A microphone for voice input
- Access to an LLM API (OpenAI, Anthropic, or local model)
- All required Python packages installed

## Lab Setup

### Robot Platform Configuration

For this lab, we'll use a simulated or physical humanoid robot. The example uses a simplified robot model, but you can adapt it to your specific platform.

```python
# robot_config.py
class RobotConfiguration:
    """Configuration for the humanoid robot platform."""

    # Robot specifications
    name = "VLA_Humanoid"
    base_model = "turtlebot3_waffle"  # or your specific robot
    has_manipulator = True
    manipulator_joints = 6  # number of DOF
    camera_topic = "/camera/image_raw"
    audio_topic = "/audio_input"
    navigation_action = "move_base"

    # Physical constraints
    max_velocity = 0.5  # m/s
    max_angular_velocity = 1.0  # rad/s
    manipulator_speed = 0.1  # m/s

    # Operational parameters
    battery_capacity = 100.0  # percentage
    operational_time = 2.0  # hours
    safe_zones = ["kitchen", "living_room", "office", "bedroom"]
    restricted_zones = ["bathroom", "closet"]

# Environment configuration
class EnvironmentConfiguration:
    """Configuration for the environment."""

    # Known locations
    locations = {
        "kitchen": {"x": 2.0, "y": 1.0, "description": "Kitchen with counter and sink"},
        "bedroom": {"x": 0.5, "y": 2.5, "description": "Bedroom with bed and dresser"},
        "office": {"x": 0.0, "y": 0.0, "description": "Office with desk and chair"},
        "living_room": {"x": 1.5, "y": 2.0, "description": "Living room with couch and table"},
        "dining_room": {"x": 2.5, "y": 0.0, "description": "Dining room with table and chairs"}
    }

    # Common objects
    common_objects = [
        "cup", "bottle", "book", "phone", "keys", "wallet",
        "glasses", "plate", "fork", "spoon", "knife"
    ]

    # Object locations (for simulation)
    object_spawn_points = {
        "kitchen": ["counter", "table", "sink_area"],
        "bedroom": ["dresser", "nightstand", "bed"],
        "office": ["desk", "chair", "bookshelf"],
        "living_room": ["coffee_table", "couch_side", "tv_stand"]
    }
```

### Complete System Integration

```python
# capstone_system.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from move_base_msgs.action import MoveBase
from control_msgs.action import FollowJointTrajectory
import json
import time
import threading
from typing import Dict, List, Any, Optional

from .robot_config import RobotConfiguration, EnvironmentConfiguration
from .voice_processor import VoiceProcessorNode
from .llm_planner import LLMPlannerNode
from .perception_node import PerceptionNode
from .action_executor import ActionExecutorNode

class CapstoneVLASystem(Node):
    def __init__(self):
        super().__init__('capstone_vla_system')

        # Initialize robot configuration
        self.robot_config = RobotConfiguration()
        self.env_config = EnvironmentConfiguration()

        # System state
        self.system_state = "IDLE"
        self.current_task = None
        self.robot_battery_level = 100.0
        self.is_safe_to_operate = True

        # Publishers
        self.system_status_pub = self.create_publisher(String, 'system_status', 10)
        self.safety_alert_pub = self.create_publisher(String, 'safety_alerts', 10)
        self.task_log_pub = self.create_publisher(String, 'task_log', 10)

        # Subscribers
        self.task_request_sub = self.create_subscription(
            String, 'task_requests', self.task_request_callback, 10
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10
        )

        # Initialize subsystems
        self.voice_processor = VoiceProcessorNode()
        self.llm_planner = LLMPlannerNode()
        self.perception_system = PerceptionNode()
        self.action_executor = ActionExecutorNode()

        # Task execution thread
        self.task_execution_thread = None
        self.task_execution_lock = threading.Lock()

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.system_monitor)

        self.get_logger().info('Capstone VLA System initialized')
        self.log_task("System initialized and ready for tasks")

    def task_request_callback(self, msg: String):
        """Handle incoming task requests."""
        try:
            task_data = json.loads(msg.data)
            command = task_data.get('command', '')
            priority = task_data.get('priority', 'normal')

            self.get_logger().info(f'Received task request: {command} (Priority: {priority})')

            # Validate system state before accepting task
            if not self.is_safe_to_operate:
                self.get_logger().warn('System not safe to operate, rejecting task')
                return

            if self.robot_battery_level < 10.0:  # Less than 10% battery
                self.get_logger().warn('Battery level too low, rejecting task')
                self.publish_safety_alert("Battery level too low for task execution")
                return

            # Process the task
            self.execute_task(command, priority)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid task request format')
        except Exception as e:
            self.get_logger().error(f'Error processing task request: {e}')

    def execute_task(self, command: str, priority: str = 'normal'):
        """Execute a task using the VLA pipeline."""
        with self.task_execution_lock:
            if self.system_state != "IDLE":
                self.get_logger().warn(f'System busy with {self.system_state}, rejecting new task')
                return

            self.system_state = "EXECUTING_TASK"
            self.current_task = command

            # Log the task
            self.log_task(f"Starting task: {command}")

            # Start task execution in a separate thread to avoid blocking
            self.task_execution_thread = threading.Thread(
                target=self._execute_task_pipeline,
                args=(command, priority),
                daemon=True
            )
            self.task_execution_thread.start()

    def _execute_task_pipeline(self, command: str, priority: str):
        """Execute the complete VLA pipeline for a task."""
        try:
            self.get_logger().info(f'Starting VLA pipeline for: {command}')

            # Step 1: Perception - Get current environment state
            self.get_logger().info('Step 1: Acquiring environment perception')
            env_context = self.get_environment_context()

            # Step 2: LLM Planning - Generate action sequence
            self.get_logger().info('Step 2: Planning with LLM')
            action_sequence = self.plan_task_with_llm(command, env_context)

            if not action_sequence:
                self.get_logger().error('Failed to generate action sequence')
                self.log_task(f"Task failed: Could not generate action sequence for '{command}'")
                return

            # Step 3: Action Execution - Execute the planned sequence
            self.get_logger().info('Step 3: Executing action sequence')
            success = self.execute_action_sequence(action_sequence)

            if success:
                self.get_logger().info('Task completed successfully')
                self.log_task(f"Task completed successfully: {command}")
            else:
                self.get_logger().error('Task execution failed')
                self.log_task(f"Task failed during execution: {command}")

        except Exception as e:
            self.get_logger().error(f'Error in task pipeline: {e}')
            self.log_task(f"Task failed with error: {command} - {str(e)}")
        finally:
            self.system_state = "IDLE"
            self.current_task = None

    def get_environment_context(self) -> Dict[str, Any]:
        """Get current environment context from perception system."""
        # In a real system, this would wait for perception data
        # For this example, we'll return a simulated context
        return {
            "timestamp": time.time(),
            "objects_in_view": [
                {"class": "cup", "location": "kitchen_table", "confidence": 0.85},
                {"class": "book", "location": "office_desk", "confidence": 0.92}
            ],
            "robot_location": "office",
            "navigable_locations": ["kitchen", "bedroom", "office", "living_room"],
            "battery_level": self.robot_battery_level
        }

    def plan_task_with_llm(self, command: str, env_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan a task using LLM with environment context."""
        # This would typically call the LLM planner node
        # For this example, we'll simulate the planning

        # Simulate different action sequences based on command
        if "bring me" in command.lower() or "fetch" in command.lower():
            # Example: "Bring me the cup from the kitchen"
            target_object = "cup"  # Extract from command
            target_location = "kitchen"  # Extract from command

            return [
                {
                    "action_type": "navigation",
                    "parameters": {"target_location": target_location},
                    "description": f"Navigate to {target_location}",
                    "estimated_duration": 15.0
                },
                {
                    "action_type": "perception",
                    "parameters": {"target_object": target_object},
                    "description": f"Locate {target_object}",
                    "estimated_duration": 5.0
                },
                {
                    "action_type": "manipulation",
                    "parameters": {"action": "pick_up", "target_object": target_object},
                    "description": f"Pick up {target_object}",
                    "estimated_duration": 10.0
                },
                {
                    "action_type": "navigation",
                    "parameters": {"target_location": "office"},
                    "description": "Return to user location",
                    "estimated_duration": 15.0
                },
                {
                    "action_type": "manipulation",
                    "parameters": {"action": "place", "target_object": target_object},
                    "description": f"Place {target_object} near user",
                    "estimated_duration": 5.0
                }
            ]
        elif "clean" in command.lower():
            # Example: "Clean the room"
            return [
                {
                    "action_type": "perception",
                    "parameters": {"task": "scan_room"},
                    "description": "Scan room for objects to clean",
                    "estimated_duration": 10.0
                },
                {
                    "action_type": "navigation",
                    "parameters": {"target_location": "kitchen"},
                    "description": "Move to cleaning station",
                    "estimated_duration": 10.0
                },
                {
                    "action_type": "manipulation",
                    "parameters": {"action": "collect_items"},
                    "description": "Collect items that need to be organized",
                    "estimated_duration": 30.0
                },
                {
                    "action_type": "navigation",
                    "parameters": {"target_location": "office"},
                    "description": "Return to starting location",
                    "estimated_duration": 10.0
                }
            ]
        else:
            # Default action sequence
            return [
                {
                    "action_type": "perception",
                    "parameters": {"task": "understand_command"},
                    "description": "Analyze command and environment",
                    "estimated_duration": 5.0
                }
            ]

    def execute_action_sequence(self, action_sequence: List[Dict[str, Any]]) -> bool:
        """Execute a sequence of actions."""
        for i, action in enumerate(action_sequence):
            self.get_logger().info(f'Executing action {i+1}/{len(action_sequence)}: {action["description"]}')

            success = self.execute_single_action(action)
            if not success:
                self.get_logger().error(f'Action failed: {action["description"]}')
                return False

            # Update battery level based on action
            self.update_battery_level(action["estimated_duration"])

            # Check safety after each action
            if not self.is_safe_to_operate:
                self.get_logger().error('Safety check failed during task execution')
                return False

        return True

    def execute_single_action(self, action: Dict[str, Any]) -> bool:
        """Execute a single action."""
        action_type = action["action_type"]
        parameters = action["parameters"]

        if action_type == "navigation":
            return self.execute_navigation_action(parameters)
        elif action_type == "manipulation":
            return self.execute_manipulation_action(parameters)
        elif action_type == "perception":
            return self.execute_perception_action(parameters)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation_action(self, parameters: Dict[str, Any]) -> bool:
        """Execute navigation action."""
        try:
            target_location = parameters.get("target_location", "unknown")

            # Get coordinates for the location
            location_data = self.env_config.locations.get(target_location)
            if not location_data:
                self.get_logger().error(f'Unknown location: {target_location}')
                return False

            # In a real system, this would send a navigation goal
            # For simulation, we'll just wait
            self.get_logger().info(f'Navigating to {target_location} at ({location_data["x"]}, {location_data["y"]})')

            # Simulate navigation time
            time.sleep(action.get("estimated_duration", 5.0) * 0.1)  # Faster simulation

            self.get_logger().info(f'Reached {target_location}')
            return True

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def execute_manipulation_action(self, parameters: Dict[str, Any]) -> bool:
        """Execute manipulation action."""
        try:
            action = parameters.get("action", "unknown")
            target_object = parameters.get("target_object", "unknown")

            self.get_logger().info(f'Executing manipulation: {action} {target_object}')

            # Simulate manipulation time
            time.sleep(action.get("estimated_duration", 5.0) * 0.1)  # Faster simulation

            self.get_logger().info(f'Manipulation completed: {action} {target_object}')
            return True

        except Exception as e:
            self.get_logger().error(f'Manipulation error: {e}')
            return False

    def execute_perception_action(self, parameters: Dict[str, Any]) -> bool:
        """Execute perception action."""
        try:
            task = parameters.get("task", "scan")

            self.get_logger().info(f'Executing perception task: {task}')

            # Simulate perception time
            time.sleep(action.get("estimated_duration", 5.0) * 0.1)  # Faster simulation

            self.get_logger().info(f'Perception task completed: {task}')
            return True

        except Exception as e:
            self.get_logger().error(f'Perception error: {e}')
            return False

    def update_battery_level(self, action_duration: float):
        """Update robot battery level based on action duration."""
        # Simulate battery drain (0.1% per second of operation)
        battery_drain = action_duration * 0.1
        self.robot_battery_level = max(0.0, self.robot_battery_level - battery_drain)

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop requests."""
        if msg.data:
            self.get_logger().warn('Emergency stop activated!')
            self.is_safe_to_operate = False
            self.system_state = "EMERGENCY_STOP"

            # Publish safety alert
            self.publish_safety_alert("Emergency stop activated by user")
        else:
            self.is_safe_to_operate = True
            if self.system_state == "EMERGENCY_STOP":
                self.system_state = "IDLE"

    def system_monitor(self):
        """Monitor system health and safety."""
        # Check battery level
        if self.robot_battery_level < 5.0:
            self.get_logger().warn(f'Critical battery level: {self.robot_battery_level}%')
            self.publish_safety_alert(f"Critical battery level: {self.robot_battery_level}%")

        # Check system state
        if self.system_state == "EXECUTING_TASK":
            # Additional safety checks during task execution
            pass

        # Publish system status
        status_msg = String()
        status_msg.data = json.dumps({
            "state": self.system_state,
            "battery_level": self.robot_battery_level,
            "current_task": self.current_task or "none",
            "safe_to_operate": self.is_safe_to_operate,
            "timestamp": time.time()
        })
        self.system_status_pub.publish(status_msg)

    def publish_safety_alert(self, alert: str):
        """Publish a safety alert."""
        alert_msg = String()
        alert_msg.data = alert
        self.safety_alert_pub.publish(alert_msg)

    def log_task(self, message: str):
        """Log a task message."""
        log_msg = String()
        log_msg.data = json.dumps({
            "timestamp": time.time(),
            "message": message,
            "robot_name": self.robot_config.name
        })
        self.task_log_pub.publish(log_msg)

def main(args=None):
    """Main function to run the capstone system."""
    rclpy.init(args=args)

    capstone_system = CapstoneVLASystem()

    try:
        rclpy.spin(capstone_system)
    except KeyboardInterrupt:
        capstone_system.get_logger().info('Shutting down capstone system...')
    finally:
        capstone_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Task Examples

Let's implement a few practical tasks that demonstrate the VLA system capabilities:

```python
# task_examples.py
import json
from std_msgs.msg import String

class TaskExamples:
    """Examples of tasks for the VLA system."""

    @staticmethod
    def bring_coffee_task():
        """Task: Bring me a cup of coffee from the kitchen."""
        return {
            "command": "Bring me a cup of coffee from the kitchen",
            "expected_actions": [
                "Navigate to kitchen",
                "Locate cup or coffee maker",
                "Pick up cup",
                "Navigate back to user",
                "Place cup near user"
            ],
            "success_criteria": "User receives cup in kitchen area"
        }

    @staticmethod
    def clean_room_task():
        """Task: Clean the room."""
        return {
            "command": "Clean the room",
            "expected_actions": [
                "Scan room for objects",
                "Identify objects that need organizing",
                "Navigate to object locations",
                "Pick up and organize objects",
                "Return to starting position"
            ],
            "success_criteria": "Room is tidier with objects in appropriate locations"
        }

    @staticmethod
    def find_lost_item_task():
        """Task: Find my keys."""
        return {
            "command": "Find my keys",
            "expected_actions": [
                "Scan environment for key-like objects",
                "Check common locations for keys",
                "Navigate to potential locations",
                "Identify and confirm key object",
                "Navigate to user with keys"
            ],
            "success_criteria": "Robot brings keys to user"
        }

    @staticmethod
    def escort_to_room_task():
        """Task: Escort me to the bedroom."""
        return {
            "command": "Escort me to the bedroom",
            "expected_actions": [
                "Maintain safe distance from user",
                "Navigate to bedroom",
                "Ensure path is clear",
                "Wait for user to follow",
                "Arrive at destination together"
            ],
            "success_criteria": "Both robot and user arrive at bedroom safely"
        }

def create_task_request(task_name: str, priority: str = "normal") -> String:
    """Create a task request message."""
    task_examples = TaskExamples()

    if task_name == "bring_coffee":
        task_data = task_examples.bring_coffee_task()
    elif task_name == "clean_room":
        task_data = task_examples.clean_room_task()
    elif task_name == "find_keys":
        task_data = task_examples.find_lost_item_task()
    elif task_name == "escort_bedroom":
        task_data = task_examples.escort_to_room_task()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    task_msg = String()
    task_msg.data = json.dumps({
        "command": task_data["command"],
        "priority": priority,
        "expected_outcome": task_data["success_criteria"],
        "timestamp": time.time()
    })

    return task_msg

# Example usage
def run_example_task(robot_node, task_name: str):
    """Run an example task."""
    task_request = create_task_request(task_name)

    # Publish the task request
    robot_node.get_logger().info(f'Starting example task: {task_name}')
    robot_node.task_request_pub.publish(task_request)
```

## System Testing and Validation

```python
# system_tester.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import time
from typing import Dict, List

class SystemTester(Node):
    """Test the VLA system with various scenarios."""

    def __init__(self):
        super().__init__('system_tester')

        # Publishers to send test commands
        self.task_request_pub = self.create_publisher(String, 'task_requests', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Subscribers to monitor system responses
        self.system_status_sub = self.create_subscription(
            String, 'system_status', self.system_status_callback, 10
        )
        self.task_log_sub = self.create_subscription(
            String, 'task_log', self.task_log_callback, 10
        )

        # Test state tracking
        self.system_status = {}
        self.test_results = []
        self.current_test = None

        # Timer to run tests
        self.test_timer = self.create_timer(5.0, self.run_next_test)
        self.test_index = 0

        # Define test cases
        self.test_cases = [
            {
                "name": "simple_navigation",
                "command": "Go to kitchen",
                "expected_duration": 30.0,
                "description": "Test basic navigation to a known location"
            },
            {
                "name": "object_interaction",
                "command": "Find the cup and bring it to me",
                "expected_duration": 60.0,
                "description": "Test perception and manipulation"
            },
            {
                "name": "complex_task",
                "command": "Clean the room",
                "expected_duration": 120.0,
                "description": "Test complex task decomposition"
            }
        ]

        self.get_logger().info('System tester initialized')

    def system_status_callback(self, msg: String):
        """Update system status."""
        try:
            self.system_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid system status message')

    def task_log_callback(self, msg: String):
        """Handle task log messages."""
        try:
            log_data = json.loads(msg.data)
            self.get_logger().info(f'Task log: {log_data["message"]}')
        except json.JSONDecodeError:
            self.get_logger().error('Invalid task log message')

    def run_next_test(self):
        """Run the next test in sequence."""
        if self.test_index >= len(self.test_cases):
            self.get_logger().info('All tests completed')
            return

        test_case = self.test_cases[self.test_index]
        self.current_test = test_case

        self.get_logger().info(f'Running test: {test_case["name"]}')
        self.get_logger().info(f'Description: {test_case["description"]}')

        # Create and send task request
        task_request = self.create_task_request(test_case)
        self.task_request_pub.publish(task_request)

        # Schedule test result evaluation
        self.create_timer(test_case["expected_duration"], self.evaluate_test_result)

        self.test_index += 1

    def create_task_request(self, test_case: Dict) -> String:
        """Create a task request for the test case."""
        task_msg = String()
        task_msg.data = json.dumps({
            "command": test_case["command"],
            "priority": "normal",
            "test_case": test_case["name"],
            "timestamp": time.time()
        })
        return task_msg

    def evaluate_test_result(self):
        """Evaluate the result of the current test."""
        if not self.current_test:
            return

        # Check if task was completed based on system status
        success = self.system_status.get("state") == "IDLE" and self.system_status.get("current_task") == "none"

        test_result = {
            "test_name": self.current_test["name"],
            "success": success,
            "system_state": self.system_status.get("state", "unknown"),
            "battery_level": self.system_status.get("battery_level", "unknown"),
            "timestamp": time.time()
        }

        self.test_results.append(test_result)

        status = "PASSED" if success else "FAILED"
        self.get_logger().info(f'Test {self.current_test["name"]} {status}')

        # Reset current test
        self.current_test = None

def main(args=None):
    """Main function for system testing."""
    rclpy.init(args=args)

    tester = SystemTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Test interrupted by user')
    finally:
        # Print test summary
        tester.get_logger().info('Test Results Summary:')
        for result in tester.test_results:
            status = "✓" if result["success"] else "✗"
            tester.get_logger().info(f'{status} {result["test_name"]}: {result["system_state"]}')

        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Evaluation

```python
# performance_evaluator.py
import time
import statistics
from typing import Dict, List, Tuple
import json

class PerformanceEvaluator:
    """Evaluate the performance of the VLA system."""

    def __init__(self):
        self.metrics = {
            "task_completion_rate": [],
            "average_task_time": [],
            "planning_time": [],
            "execution_success_rate": [],
            "battery_efficiency": [],
            "user_satisfaction": []
        }

        self.task_logs = []

    def log_task_start(self, task_id: str, command: str):
        """Log the start of a task."""
        self.task_logs.append({
            "task_id": task_id,
            "command": command,
            "start_time": time.time(),
            "status": "started"
        })

    def log_task_end(self, task_id: str, success: bool):
        """Log the end of a task."""
        for log in self.task_logs:
            if log["task_id"] == task_id and log["status"] == "started":
                log["end_time"] = time.time()
                log["duration"] = log["end_time"] - log["start_time"]
                log["success"] = success
                log["status"] = "completed"

                # Update metrics
                if success:
                    self.metrics["task_completion_rate"].append(1)
                    self.metrics["average_task_time"].append(log["duration"])
                else:
                    self.metrics["task_completion_rate"].append(0)

                break

    def calculate_planning_time(self, planning_start: float, planning_end: float):
        """Record planning time."""
        planning_duration = planning_end - planning_start
        self.metrics["planning_time"].append(planning_duration)

    def calculate_execution_success(self, task_id: str, actions_completed: int, total_actions: int):
        """Calculate execution success rate for a task."""
        success_rate = actions_completed / total_actions if total_actions > 0 else 0
        self.metrics["execution_success_rate"].append(success_rate)

    def calculate_battery_efficiency(self, battery_start: float, battery_end: float, task_duration: float):
        """Calculate battery efficiency."""
        battery_used = battery_start - battery_end
        efficiency = battery_used / task_duration if task_duration > 0 else 0
        self.metrics["battery_efficiency"].append(efficiency)

    def record_user_satisfaction(self, task_id: str, satisfaction_score: int):
        """Record user satisfaction (1-5 scale)."""
        if 1 <= satisfaction_score <= 5:
            self.metrics["user_satisfaction"].append(satisfaction_score)

    def get_performance_report(self) -> Dict:
        """Generate a performance report."""
        report = {}

        # Calculate averages
        if self.metrics["task_completion_rate"]:
            report["overall_completion_rate"] = statistics.mean(self.metrics["task_completion_rate"])

        if self.metrics["average_task_time"]:
            report["average_task_duration"] = statistics.mean(self.metrics["average_task_time"])
            report["median_task_duration"] = statistics.median(self.metrics["average_task_time"])

        if self.metrics["planning_time"]:
            report["average_planning_time"] = statistics.mean(self.metrics["planning_time"])

        if self.metrics["execution_success_rate"]:
            report["average_execution_success_rate"] = statistics.mean(self.metrics["execution_success_rate"])

        if self.metrics["battery_efficiency"]:
            report["average_battery_efficiency"] = statistics.mean(self.metrics["battery_efficiency"])

        if self.metrics["user_satisfaction"]:
            report["average_user_satisfaction"] = statistics.mean(self.metrics["user_satisfaction"])

        # Add raw counts
        report["total_tasks_attempted"] = len([log for log in self.task_logs if log["status"] == "completed"])
        report["successful_tasks"] = len([log for log in self.task_logs if log.get("success", False)])
        report["failed_tasks"] = len([log for log in self.task_logs if not log.get("success", True)])

        return report

# Example usage in the capstone system
def integrate_performance_evaluation():
    """Example of how to integrate performance evaluation."""
    evaluator = PerformanceEvaluator()

    # Example: Evaluate a task
    task_id = f"task_{int(time.time())}"
    command = "Bring me the cup from the kitchen"

    evaluator.log_task_start(task_id, command)

    # Simulate task execution
    time.sleep(2.0)  # Simulated task time

    # Log task completion
    success = True  # Simulated success
    evaluator.log_task_end(task_id, success)

    # Record planning time
    evaluator.calculate_planning_time(time.time() - 2.1, time.time() - 2.0)

    # Record execution success (3 out of 4 actions completed)
    evaluator.calculate_execution_success(task_id, 3, 4)

    # Record battery efficiency (started at 80%, ended at 79.5% over 2 seconds)
    evaluator.calculate_battery_efficiency(80.0, 79.5, 2.0)

    # Record user satisfaction (4 out of 5)
    evaluator.record_user_satisfaction(task_id, 4)

    # Generate report
    report = evaluator.get_performance_report()
    print("Performance Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
```

## Lab Exercises

### Exercise 1: Basic Task Execution
1. Deploy the VLA system on your robot platform
2. Execute a simple navigation task: "Go to the kitchen"
3. Observe the system's response and execution
4. Verify that the robot successfully navigates to the kitchen

### Exercise 2: Object Manipulation
1. Set up an environment with recognizable objects
2. Execute a manipulation task: "Bring me the red cup"
3. Monitor the perception, planning, and execution phases
4. Verify that the robot successfully identifies and brings the object

### Exercise 3: Complex Task Decomposition
1. Create a multi-step task: "Clean the room by putting books on the shelf and cups in the kitchen"
2. Observe how the LLM decomposes this into subtasks
3. Monitor the execution of each subtask
4. Evaluate the overall task completion

### Exercise 4: Performance Optimization
1. Run the system tester to evaluate performance
2. Identify bottlenecks in the system
3. Optimize one component (e.g., perception speed, planning efficiency)
4. Re-run tests to measure improvement

### Exercise 5: Error Handling and Recovery
1. Introduce an obstacle in the robot's path
2. Execute a navigation task and observe error handling
3. Verify that the system can recover from the error
4. Test emergency stop functionality

## Troubleshooting Common Issues

During the capstone lab, you may encounter various issues. Here are common problems and solutions:

1. **LLM API Timeout**: Ensure your API key is valid and network connection is stable
2. **Perception Failures**: Check camera calibration and lighting conditions
3. **Navigation Failures**: Verify map accuracy and localization
4. **Action Execution Failures**: Check robot hardware status and joint limits

## Summary

In this capstone lab, we've implemented a complete Vision-Language-Action system that integrates:

- Voice recognition with Whisper for natural language input
- LLM-based planning for task decomposition
- Computer vision perception for environment understanding
- ROS 2 action execution for robot control
- Safety monitoring and error handling
- Performance evaluation and metrics

The system demonstrates the full pipeline from voice command to robot action, showcasing the power of VLA robotics. In the next chapter, we'll cover troubleshooting techniques for common issues that may arise in VLA systems.