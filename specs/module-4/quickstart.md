# Module 4 Quickstart: Vision-Language-Action (VLA) Robotics

## Overview

This quickstart guide provides a rapid introduction to Vision-Language-Action (VLA) robotics concepts and implementation. This module teaches how to integrate Large Language Models (LLMs), computer vision, and voice recognition to control humanoid robots.

## Prerequisites

Before starting this module, you should have:

1. Basic understanding of ROS 2 concepts (covered in Module 1)
2. Familiarity with Python programming
3. Knowledge of basic computer vision concepts
4. Understanding of LLMs and their applications
5. Access to a humanoid robot platform (simulated or physical) with ROS 2 support

## Setup Requirements

### Software Dependencies

```bash
# Install ROS 2 Humble Hawksbill (if not already installed)
# Follow ROS 2 installation guide for your platform

# Install Python dependencies
pip install openai-whisper
pip install ultralytics
pip install openai
pip install torch torchvision
pip install opencv-python
pip install pyaudio
pip install rclpy

# For 3D perception (optional)
pip install open3d
```

### Hardware Requirements

- **Computing**: GPU-enabled system for real-time inference (minimum RTX 3060 or equivalent)
- **Robot Platform**: ROS 2 compatible mobile manipulator
- **Sensors**: RGB-D camera, microphone array
- **Connectivity**: Stable network connection for cloud-based LLMs

## Getting Started

### 1. Voice Recognition Setup

First, let's set up voice recognition using Whisper:

```python
import whisper
import pyaudio
import numpy as np

# Load Whisper model
model = whisper.load_model("base")

# Audio parameters
audio_format = pyaudio.paInt16
channels = 1
rate = 16000
chunk = 1024
record_seconds = 5

# Initialize audio
p = pyaudio.PyAudio()
stream = p.open(
    format=audio_format,
    channels=channels,
    rate=rate,
    input=True,
    frames_per_buffer=chunk
)

print("Recording for voice command...")
frames = []

for _ in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("Processing voice command...")

# Convert to numpy array
audio_data = b''.join(frames)
audio_np = np.frombuffer(audio_data, dtype=np.int16)
audio_np = audio_np.astype(np.float32) / 32768.0

# Transcribe
result = model.transcribe(audio_np)
command = result["text"].strip()
print(f"Recognized command: {command}")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
```

### 2. LLM Planning Pipeline

Next, implement the LLM planning pipeline:

```python
import openai
import json

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

def plan_task(command, robot_state, env_context):
    """Plan a task using LLM."""
    system_prompt = """
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        response_json = json.loads(response.choices[0].message.content)
        return response_json

    except Exception as e:
        print(f"Error during planning: {e}")
        return None

# Example usage
robot_state = {
    "current_location": "office",
    "battery_level": 0.85,
    "gripper_status": "open",
    "available_actions": ["navigation", "manipulation", "perception"]
}

env_context = {
    "objects_in_view": [{"class": "cup", "location": "kitchen_table", "confidence": 0.85}],
    "navigable_locations": ["kitchen", "bedroom", "office", "living_room"]
}

command = "Bring me the cup from the kitchen"
plan = plan_task(command, robot_state, env_context)

if plan:
    print("Generated action sequence:")
    for i, action in enumerate(plan["actions"]):
        print(f"  {i+1}. {action['description']}")
```

### 3. Object Detection Setup

Set up computer vision with YOLO:

```python
import cv2
from ultralytics import YOLO
import torch

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_objects(image):
    """Detect objects in an image."""
    results = model(image, conf=0.5)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': class_name,
                    'confidence': float(confidence),
                    'class_id': class_id
                })

    return detections

# Example usage with camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)

    # Draw bounding boxes
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                   (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. Complete VLA System Integration

Now let's integrate all components into a simple VLA system:

```python
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RobotAction:
    action_type: str
    parameters: Dict[str, Any]
    description: str
    estimated_duration: float

class SimpleVLASystem:
    def __init__(self):
        # Initialize components (in a real system, these would be full implementations)
        self.voice_active = False
        self.command_history = []

    def start_voice_recognition(self):
        """Start listening for voice commands."""
        print("Starting voice recognition...")
        self.voice_active = True

        # In a real system, this would run continuously
        # For this example, we'll simulate a command
        command = "Bring me the cup from the kitchen"
        print(f"Simulated voice command: {command}")
        self.process_command(command)

    def process_command(self, command: str):
        """Process a voice command through the VLA pipeline."""
        print(f"Processing command: {command}")

        # Step 1: Get environment context (simulated)
        env_context = {
            "objects_in_view": [{"class": "cup", "location": "kitchen_table", "confidence": 0.85}],
            "robot_location": "office",
            "navigable_locations": ["kitchen", "bedroom", "office", "living_room"]
        }

        # Step 2: Plan with LLM (simulated - in real system, call the LLM)
        print("Planning with LLM...")
        action_sequence = [
            RobotAction("navigation", {"target_location": "kitchen"}, "Navigate to kitchen", 15.0),
            RobotAction("perception", {"target_object": "cup"}, "Locate cup", 5.0),
            RobotAction("manipulation", {"action": "pick_up", "target_object": "cup"}, "Pick up cup", 10.0),
            RobotAction("navigation", {"target_location": "office"}, "Return to user", 15.0),
            RobotAction("manipulation", {"action": "place", "target_object": "cup"}, "Place cup near user", 5.0)
        ]

        # Step 3: Execute action sequence
        print("Executing action sequence...")
        for i, action in enumerate(action_sequence):
            print(f"  Action {i+1}/{len(action_sequence)}: {action.description}")
            time.sleep(action.estimated_duration * 0.1)  # Simulate execution time
            print(f"    Completed: {action.description}")

        print("Task completed successfully!")
        self.command_history.append(command)

    def run(self):
        """Run the VLA system."""
        print("VLA System starting...")
        self.start_voice_recognition()

# Run the system
vla_system = SimpleVLASystem()
vla_system.run()
```

## Key Concepts to Remember

### 1. The VLA Pipeline
- **Vision**: Understanding the environment through cameras and sensors
- **Language**: Processing natural language commands using LLMs
- **Action**: Executing complex behaviors in the physical world

### 2. Safety Considerations
- Always validate actions before execution
- Monitor robot state and environment
- Implement emergency stop capabilities
- Check battery levels and operational constraints

### 3. Performance Optimization
- Use appropriate model sizes for your hardware
- Implement efficient data processing pipelines
- Consider edge vs cloud processing trade-offs
- Optimize for real-time performance requirements

## Next Steps

1. Complete the full chapters in this module to understand each component in depth
2. Practice with the capstone lab to implement a complete VLA system
3. Experiment with different LLMs and computer vision models
4. Integrate with your specific robot platform
5. Explore advanced topics in the troubleshooting section

## Troubleshooting Quick Tips

- If voice recognition isn't working: Check microphone permissions and audio settings
- If LLM planning is slow: Consider using smaller models or local inference
- If object detection fails: Verify camera calibration and lighting conditions
- If actions don't execute: Check ROS 2 communication and robot status

This quickstart provides the foundation for understanding VLA robotics. Continue with the full module to gain comprehensive knowledge of implementing Vision-Language-Action systems for humanoid robots.