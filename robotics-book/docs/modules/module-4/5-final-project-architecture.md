---
sidebar_position: 5
title: "Final Project Architecture"
---

# Final Project Architecture

## Overview of the Complete VLA System

In this chapter, we'll design the complete architecture for our Vision-Language-Action (VLA) robotics system. This architecture integrates all components we've developed in previous chapters: voice recognition with Whisper, LLM-based planning, and computer vision perception. The goal is to create a cohesive system that can receive natural language commands, understand them, perceive the environment, plan actions, and execute them using ROS 2.

## System Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Human User    │───▶│  Voice Command   │───▶│    LLM Planner   │
│                 │    │   Recognition    │    │      (GPT-4)     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                              │                          │
                              ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Environment   │───▶│  Perception &    │───▶│  Action Executor │
│                 │    │   Computer Vision│    │   (ROS 2)        │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                              │                          │
                              └──────────────────────────┘
                                         │
                              ┌──────────────────┐
                              │   Robot Platform │
                              │   (Navigation,   │
                              │   Manipulation)  │
                              └──────────────────┘
```

## High-Level Architecture

### 1. Input Processing Layer
- **Voice Command Recognition**: Processes natural language using Whisper
- **Command Validation**: Ensures commands are safe and executable
- **Context Integration**: Combines voice commands with environmental context

### 2. Reasoning Layer
- **LLM Planner**: Converts natural language to action sequences
- **Task Decomposition**: Breaks complex tasks into manageable subtasks
- **Safety Validation**: Ensures planned actions are safe

### 3. Perception Layer
- **Object Detection**: Identifies objects in the environment
- **Scene Understanding**: Understands spatial relationships
- **Visual Grounding**: Connects visual information with language concepts

### 4. Execution Layer
- **ROS 2 Action Clients**: Interfaces with robot capabilities
- **Task Execution Monitor**: Tracks action progress
- **Feedback Processing**: Updates system based on execution results

## Component Architecture

### Voice Processing Component

```python
# voice_processor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import torch
import pyaudio
import numpy as np
from threading import Thread
import time

class VoiceProcessorNode(Node):
    def __init__(self):
        super().__init__('voice_processor_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

        # Audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 3

        # Publishers and subscribers
        self.command_publisher = self.create_publisher(String, 'processed_commands', 10)
        self.activation_subscriber = self.create_subscription(
            String, 'voice_activation', self.activation_callback, 10
        )

        # State management
        self.is_listening = False
        self.listening_thread = None

        self.get_logger().info('Voice Processor Node initialized')

    def activation_callback(self, msg):
        """Handle activation messages."""
        if msg.data.lower() == 'start_listening':
            self.start_listening()
        elif msg.data.lower() == 'stop_listening':
            self.stop_listening()

    def start_listening(self):
        """Start listening for voice commands."""
        if not self.is_listening:
            self.is_listening = True
            self.listening_thread = Thread(target=self._listening_loop, daemon=True)
            self.listening_thread.start()
            self.get_logger().info('Started listening for voice commands')

    def stop_listening(self):
        """Stop listening for voice commands."""
        self.is_listening = False
        self.get_logger().info('Stopped listening for voice commands')

    def _listening_loop(self):
        """Main listening loop."""
        while self.is_listening:
            try:
                command = self.record_and_recognize()
                if command and command.strip():
                    # Publish the recognized command
                    cmd_msg = String()
                    cmd_msg.data = command.strip()
                    self.command_publisher.publish(cmd_msg)
                    self.get_logger().info(f'Published command: "{command}"')
            except Exception as e:
                self.get_logger().error(f'Error in listening loop: {e}')

            time.sleep(0.5)  # Small delay between recordings

    def record_and_recognize(self):
        """Record audio and recognize speech."""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        # Transcribe using Whisper
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio_np).to("cuda")
            self.whisper_model = self.whisper_model.to("cuda")
        else:
            audio_tensor = torch.from_numpy(audio_np)

        result = self.whisper_model.transcribe(audio_tensor)
        transcription = result["text"].strip()

        return transcription
```

### LLM Planning Component

```python
# llm_planner.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class RobotAction:
    action_type: str
    parameters: Dict[str, Any]
    description: str
    estimated_duration: float

@dataclass
class PlanningResult:
    success: bool
    actions: List[RobotAction]
    reasoning: str
    error_message: str = None

class LLMPlannerNode(Node):
    def __init__(self):
        super().__init__('llm_planner_node')

        # Initialize OpenAI client
        openai.api_key = self.get_parameter('openai_api_key').value

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'processed_commands', self.command_callback, 10
        )

        # Publishers
        self.action_sequence_pub = self.create_publisher(
            String, 'action_sequences', 10
        )
        self.status_pub = self.create_publisher(
            String, 'planner_status', 10
        )

        # System prompt for the LLM
        self.system_prompt = self._create_system_prompt()

        self.get_logger().info('LLM Planner Node initialized')

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """
You are an expert robotic task planner. Convert natural language commands into sequences of robot actions.

Available robot capabilities:
- Navigation: move_to(location)
- Manipulation: pick_up(object), place_object(object, location)
- Perception: look_for(object), scan_area()
- Interaction: open_door(), press_button()

Output format (JSON):
{
    "actions": [
        {
            "action_type": "navigation|manipulation|perception|interaction",
            "parameters": {"target_location": "kitchen", "target_object": "cup"},
            "description": "Move to kitchen",
            "estimated_duration": 10.0
        }
    ],
    "reasoning": "Explanation of planning decisions"
}

Rules:
1. Return valid JSON only
2. Ensure actions are executable by the robot
3. Include safety considerations
4. Break complex tasks into simple steps
"""

    def command_callback(self, msg: String):
        """Process incoming commands."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Publish status
        status_msg = String()
        status_msg.data = f"Planning for: {command}"
        self.status_pub.publish(status_msg)

        # Get robot state and environment context
        robot_state = self.get_robot_state()
        env_context = self.get_environment_context()

        # Plan the task
        planning_result = self.plan_task(command, robot_state, env_context)

        if planning_result.success:
            # Publish action sequence
            action_msg = String()
            action_msg.data = json.dumps({
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

            self.action_sequence_pub.publish(action_msg)
            self.get_logger().info(f'Published action sequence with {len(planning_result.actions)} actions')
        else:
            self.get_logger().error(f'Planning failed: {planning_result.error_message}')

    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state."""
        return {
            "current_location": "office",
            "battery_level": 0.85,
            "gripper_status": "open",
            "available_actions": ["navigation", "manipulation", "perception"]
        }

    def get_environment_context(self) -> Dict[str, Any]:
        """Get environment context from perception system."""
        # In a real system, this would subscribe to perception data
        return {
            "objects_in_view": [],
            "navigable_locations": ["kitchen", "bedroom", "office", "living_room"],
            "obstacles": []
        }

    def plan_task(self, command: str, robot_state: Dict[str, Any], env_context: Dict[str, Any]) -> PlanningResult:
        """Plan a task using LLM."""
        user_prompt = f"""
Command: {command}

Robot state: {json.dumps(robot_state, indent=2)}
Environment context: {json.dumps(env_context, indent=2)}

Generate a sequence of actions to complete this task.
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            response_json = json.loads(response.choices[0].message.content)

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

        except Exception as e:
            return PlanningResult(
                success=False,
                actions=[],
                reasoning="",
                error_message=f"Error during planning: {str(e)}"
            )
```

### Perception Component

```python
# perception_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO
import json
import time

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Initialize YOLO detector
        self.detector = YOLO("yolov8n.pt")
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            String, '/environment_context', 10
        )

        # Configuration
        self.confidence_threshold = 0.5
        self.update_rate = 1.0  # seconds between updates

        # State tracking
        self.last_update_time = 0

        self.get_logger().info('Perception Node initialized')

    def image_callback(self, msg):
        """Process incoming camera images."""
        current_time = time.time()

        # Limit update rate to avoid overwhelming the system
        if current_time - self.last_update_time < self.update_rate:
            return

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run object detection
            detections = self.detect_objects(cv_image)

            # Create environment context
            env_context = {
                'timestamp': current_time,
                'detections': detections,
                'scene_description': self.describe_scene(detections),
                'object_locations': self.get_object_locations(detections)
            }

            # Publish environment context
            context_msg = String()
            context_msg.data = json.dumps(env_context)
            self.detection_pub.publish(context_msg)

            self.last_update_time = current_time
            self.get_logger().info(f'Published environment context with {len(detections)} objects')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        """Detect objects in an image."""
        results = self.detector(image, conf=self.confidence_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.detector.names[class_id]

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': class_name,
                        'confidence': float(confidence),
                        'class_id': class_id
                    })

        return detections

    def describe_scene(self, detections):
        """Create a natural language description of the scene."""
        if not detections:
            return "The scene appears empty."

        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        objects = []
        for class_name, count in class_counts.items():
            if count == 1:
                objects.append(f"a {class_name}")
            else:
                objects.append(f"{count} {class_name}s")

        if len(objects) == 1:
            scene_desc = f"The scene contains {objects[0]}."
        else:
            scene_desc = f"The scene contains {', '.join(objects[:-1])}, and {objects[-1]}."

        return scene_desc

    def get_object_locations(self, detections):
        """Get simplified location information for objects."""
        locations = {}
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Convert to relative positions (left, center, right)
            if center_x < 213:  # Assuming 640x480 image
                horizontal_pos = "left"
            elif center_x < 426:
                horizontal_pos = "center"
            else:
                horizontal_pos = "right"

            # Convert to relative positions (top, middle, bottom)
            if center_y < 160:
                vertical_pos = "top"
            elif center_y < 320:
                vertical_pos = "middle"
            else:
                vertical_pos = "bottom"

            key = f"{detection['class']}_{horizontal_pos}_{vertical_pos}"
            if key not in locations:
                locations[key] = []
            locations[key].append(detection)

        return locations
```

### Action Execution Component

```python
# action_executor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from move_base_msgs.action import MoveBase
import json
import time

class ActionExecutorNode(Node):
    def __init__(self):
        super().__init__('action_executor_node')

        # Action clients
        self.navigation_client = ActionClient(self, MoveBase, 'move_base')

        # Subscribers
        self.action_sequence_sub = self.create_subscription(
            String, 'action_sequences', self.action_sequence_callback, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, 'execution_status', 10)

        # State management
        self.is_executing = False
        self.current_action_index = 0
        self.action_sequence = []

        self.get_logger().info('Action Executor Node initialized')

    def action_sequence_callback(self, msg: String):
        """Process incoming action sequences."""
        if self.is_executing:
            self.get_logger().warn('Already executing a sequence, ignoring new sequence')
            return

        try:
            sequence_data = json.loads(msg.data)
            self.action_sequence = sequence_data['actions']
            self.current_action_index = 0

            self.get_logger().info(f'Received action sequence with {len(self.action_sequence)} actions')
            self.execute_action_sequence()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Error parsing action sequence: {e}')

    def execute_action_sequence(self):
        """Execute the action sequence."""
        self.is_executing = True
        self.get_logger().info('Starting action sequence execution')

        for i, action in enumerate(self.action_sequence):
            self.get_logger().info(f'Executing action {i+1}/{len(self.action_sequence)}: {action["description"]}')

            success = self.execute_single_action(action)
            if not success:
                self.get_logger().error(f'Action failed: {action["description"]}')
                break

        self.is_executing = False
        self.get_logger().info('Action sequence execution completed')

    def execute_single_action(self, action: dict) -> bool:
        """Execute a single action."""
        action_type = action['action_type']
        parameters = action['parameters']

        if action_type == 'navigation':
            return self.execute_navigation(parameters)
        elif action_type == 'manipulation':
            return self.execute_manipulation(parameters)
        elif action_type == 'perception':
            return self.execute_perception(parameters)
        elif action_type == 'interaction':
            return self.execute_interaction(parameters)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation(self, parameters: dict) -> bool:
        """Execute navigation action."""
        try:
            target_location = parameters.get('target_location', 'unknown')

            # Convert location name to coordinates (simplified)
            location_coords = self.get_coordinates_for_location(target_location)
            if not location_coords:
                self.get_logger().error(f'Unknown location: {target_location}')
                return False

            # Create navigation goal
            goal = MoveBase.Goal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = self.get_clock().now().to_msg()
            goal.target_pose.pose.position.x = location_coords[0]
            goal.target_pose.pose.position.y = location_coords[1]
            goal.target_pose.pose.orientation.w = 1.0

            # Send goal
            self.navigation_client.wait_for_server()
            future = self.navigation_client.send_goal_async(goal)

            # Wait for result with timeout
            timeout = parameters.get('estimated_duration', 30.0)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

            result = future.result()
            success = result is not None and result.status == 3  # SUCCEEDED

            status_msg = String()
            status_msg.data = f"Navigation to {target_location}: {'SUCCESS' if success else 'FAILED'}"
            self.status_pub.publish(status_msg)

            return success

        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def execute_manipulation(self, parameters: dict) -> bool:
        """Execute manipulation action."""
        try:
            action = parameters.get('action', 'unknown')
            target_object = parameters.get('target_object', 'unknown')

            self.get_logger().info(f'Executing manipulation: {action} {target_object}')

            # In a real system, this would interface with manipulator controllers
            # For now, simulate the action
            time.sleep(2.0)  # Simulate action time

            status_msg = String()
            status_msg.data = f"Manipulation {action} {target_object}: SUCCESS"
            self.status_pub.publish(status_msg)

            return True

        except Exception as e:
            self.get_logger().error(f'Manipulation error: {e}')
            return False

    def execute_perception(self, parameters: dict) -> bool:
        """Execute perception action."""
        try:
            task = parameters.get('task', 'scan')
            target_object = parameters.get('target_object', 'any')

            self.get_logger().info(f'Executing perception: {task} for {target_object}')

            # In a real system, this would trigger perception nodes
            # For now, simulate the action
            time.sleep(1.0)  # Simulate perception time

            status_msg = String()
            status_msg.data = f"Perception {task} for {target_object}: SUCCESS"
            self.status_pub.publish(status_msg)

            return True

        except Exception as e:
            self.get_logger().error(f'Perception error: {e}')
            return False

    def execute_interaction(self, parameters: dict) -> bool:
        """Execute interaction action."""
        try:
            action = parameters.get('action', 'press_button')
            target = parameters.get('target', 'unknown')

            self.get_logger().info(f'Executing interaction: {action} {target}')

            # In a real system, this would interface with interaction controllers
            # For now, simulate the action
            time.sleep(1.5)  # Simulate interaction time

            status_msg = String()
            status_msg.data = f"Interaction {action} {target}: SUCCESS"
            self.status_pub.publish(status_msg)

            return True

        except Exception as e:
            self.get_logger().error(f'Interaction error: {e}')
            return False

    def get_coordinates_for_location(self, location: str) -> tuple:
        """Get coordinates for a named location."""
        locations = {
            "kitchen": (2.0, 1.0),
            "bedroom": (0.5, 2.5),
            "office": (0.0, 0.0),
            "living_room": (1.5, 2.0),
            "bathroom": (-1.0, 1.0)
        }
        return locations.get(location, None)
```

## Main System Integration Node

```python
# vla_system.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class VLARobotSystem(Node):
    def __init__(self):
        super().__init__('vla_robot_system')

        # Publishers for system control
        self.voice_activation_pub = self.create_publisher(String, 'voice_activation', 10)
        self.status_pub = self.create_publisher(String, 'system_status', 10)

        # Subscribers for system monitoring
        self.planner_status_sub = self.create_subscription(
            String, 'planner_status', self.planner_status_callback, 10
        )
        self.execution_status_sub = self.create_subscription(
            String, 'execution_status', self.execution_status_callback, 10
        )

        # System state
        self.system_state = "IDLE"
        self.current_command = ""
        self.current_task = ""

        # Timer for system status updates
        self.status_timer = self.create_timer(1.0, self.update_system_status)

        self.get_logger().info('VLA Robot System initialized')

    def planner_status_callback(self, msg: String):
        """Handle planner status updates."""
        self.get_logger().info(f'Planner status: {msg.data}')
        self.current_task = msg.data

    def execution_status_callback(self, msg: String):
        """Handle execution status updates."""
        self.get_logger().info(f'Execution status: {msg.data}')

    def update_system_status(self):
        """Update system status."""
        status_msg = String()
        status_msg.data = json.dumps({
            'state': self.system_state,
            'current_command': self.current_command,
            'current_task': self.current_task,
            'timestamp': self.get_clock().now().seconds_nanoseconds()
        })
        self.status_pub.publish(status_msg)

    def activate_voice_recognition(self):
        """Activate voice recognition."""
        self.system_state = "LISTENING"
        activation_msg = String()
        activation_msg.data = 'start_listening'
        self.voice_activation_pub.publish(activation_msg)
        self.get_logger().info('Voice recognition activated')

    def deactivate_voice_recognition(self):
        """Deactivate voice recognition."""
        self.system_state = "IDLE"
        activation_msg = String()
        activation_msg.data = 'stop_listening'
        self.voice_activation_pub.publish(activation_msg)
        self.get_logger().info('Voice recognition deactivated')

def main(args=None):
    rclpy.init(args=args)

    # Create the main system node
    vla_system = VLARobotSystem()

    # Optionally activate voice recognition at startup
    vla_system.activate_voice_recognition()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.deactivate_voice_recognition()
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File Configuration

```xml
<!-- launch/vla_system.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Voice processor node
        Node(
            package='vla_robot',
            executable='voice_processor',
            name='voice_processor_node',
            parameters=[
                {'openai_api_key': os.environ.get('OPENAI_API_KEY', '')}
            ],
            output='screen'
        ),

        # LLM planner node
        Node(
            package='vla_robot',
            executable='llm_planner',
            name='llm_planner_node',
            parameters=[
                {'openai_api_key': os.environ.get('OPENAI_API_KEY', '')}
            ],
            output='screen'
        ),

        # Perception node
        Node(
            package='vla_robot',
            executable='perception',
            name='perception_node',
            output='screen'
        ),

        # Action executor node
        Node(
            package='vla_robot',
            executable='action_executor',
            name='action_executor_node',
            output='screen'
        ),

        # Main system node
        Node(
            package='vla_robot',
            executable='vla_system',
            name='vla_robot_system',
            output='screen'
        )
    ])
```

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    VLA Robot System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Voice     │───▶│    LLM      │───▶│   Action Executor   │ │
│  │  Processor  │    │   Planner   │    │                     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│         │                   │                       │          │
│         ▼                   ▼                       ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Microphone  │    │ Environment │    │   Robot Platform    │ │
│  │   Input     │    │  Context    │    │                     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                              │                       │          │
│                              ▼                       ▼          │
│                       ┌─────────────┐        ┌─────────────┐   │
│                       │ Perception  │───────▶│ Navigation  │   │
│                       │   System    │        │  System     │   │
│                       └─────────────┘        └─────────────┘   │
│                              │                       │          │
│                              ▼                       ▼          │
│                       ┌─────────────┐        ┌─────────────┐   │
│                       │  Camera     │        │ Manipulator │   │
│                       │   Input     │        │  System     │   │
│                       └─────────────┘        └─────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Deployment Considerations

### Hardware Requirements
- **Computing**: GPU-enabled system for real-time inference (minimum RTX 3060 or equivalent)
- **Robot Platform**: ROS 2 compatible mobile manipulator
- **Sensors**: RGB-D camera, microphone array
- **Connectivity**: Stable network connection for cloud-based LLMs

### Software Requirements
- **ROS 2**: Humble Hawksbill or later
- **Python**: 3.8 or later
- **CUDA**: For GPU acceleration
- **Docker**: For containerized deployment (optional but recommended)

### Performance Optimization
- **Model Quantization**: Reduce model sizes for edge deployment
- **Caching**: Cache frequently used responses
- **Asynchronous Processing**: Use threading for non-blocking operations
- **Resource Management**: Monitor and manage computational resources

## Summary

In this chapter, we've designed the complete architecture for our VLA robotics system. The architecture integrates:

- Voice processing with Whisper for natural language input
- LLM-based planning for task decomposition and action generation
- Computer vision perception for environment understanding
- ROS 2 action execution for robot control
- System integration for cohesive operation

The architecture is designed to be modular, allowing each component to be developed and tested independently while working together as a unified system. In the next chapter, we'll implement a capstone lab that demonstrates the complete system in action.