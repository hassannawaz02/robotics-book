---
title: Comprehensive Lab Exercises and Debugging Guide
sidebar_label: Lab Exercises
---

# Comprehensive Lab Exercises and Debugging Guide

This lesson provides extensive hands-on labs with multiple practical projects that integrate comprehensive perception model training with Isaac simulation testing, along with a comprehensive debugging guide for troubleshooting complex Isaac platform issues with optimization and advanced troubleshooting.

## Learning Objectives

By the end of this lesson, you will understand:
- How to implement comprehensive integrated exercises covering Isaac Sim, ROS, and Nav2 concepts
- How to train perception models using synthetic data from Isaac Sim
- How to test perception models in Isaac simulation environments
- How to debug complex issues in the complete AI-robot brain system
- How to optimize performance across all Isaac components
- How to validate system integration with multiple practical projects

## Prerequisites

- Complete understanding of Isaac Sim, Isaac ROS VSLAM, and Nav2 for bipedal robots
- Isaac Sim environment configured from Lesson 2
- Isaac ROS VSLAM pipeline from Lesson 3
- Nav2 bipedal navigation system from Lesson 4
- Python 3.11+ environment with Isaac-compatible libraries
- NVIDIA GPU with CUDA support for acceleration

## Lab Exercise 1: Comprehensive Isaac Sim Environment Creation and Synthetic Data Generation

### Objective
Create a complex simulation environment with multiple rooms, furniture, and dynamic elements, then generate synthetic datasets for perception model training.

### Setup Instructions
1. Launch Isaac Sim with the appropriate configuration
2. Create a multi-room environment with doors, windows, and furniture
3. Configure lighting with realistic shadows and reflections
4. Set up camera sensors for stereo vision and RGB-D capture
5. Configure semantic segmentation and depth sensors

### Implementation Steps

#### Step 1: Environment Creation
```python
import omni
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.semantics import add_semantic_label
import carb
import numpy as np

# Configure simulation app
config = {
    "headless": False,
    "enable_cameras": True,
    "window_width": 1920,
    "window_height": 1080,
}

simulation_app = SimulationApp(config)
world = World(stage_units_in_meters=1.0)

# Create multi-room environment
def create_complex_environment():
    # Create main room
    create_prim(
        prim_path="/World/MainRoom",
        prim_type="Cylinder",
        position=np.array([0, 0, 0]),
        scale=np.array([10, 10, 3]),
        orientation=np.array([0, 0, 0, 1])
    )

    # Add furniture
    create_prim(
        prim_path="/World/Furniture/Table",
        prim_type="Cuboid",
        position=np.array([2, 0, 0.5]),
        scale=np.array([1.5, 0.8, 0.8]),
        orientation=np.array([0, 0, 0, 1])
    )

    create_prim(
        prim_path="/World/Furniture/Chair",
        prim_type="Cylinder",
        position=np.array([1.5, 0.8, 0.4]),
        scale=np.array([0.4, 0.4, 0.8]),
        orientation=np.array([0, 0, 0, 1])
    )

    # Add semantic labels
    add_semantic_label(prim_path="/World/Furniture/Table", semantic_label="table")
    add_semantic_label(prim_path="/World/Furniture/Chair", semantic_label="chair")

    print("Complex environment created with semantic labels")

create_complex_environment()
```

#### Step 2: Sensor Configuration
```python
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import _range_sensor

def setup_sensors():
    # Create stereo camera pair
    left_camera = Camera(
        prim_path="/World/Robot/CameraLeft",
        frequency=30,
        resolution=(1280, 720),
        position=np.array([0.1, 0.05, 0.1]),
        orientation=np.array([0, 0, 0, 1])
    )

    right_camera = Camera(
        prim_path="/World/Robot/CameraRight",
        frequency=30,
        resolution=(1280, 720),
        position=np.array([0.1, -0.05, 0.1]),
        orientation=np.array([0, 0, 0, 1])
    )

    # Enable semantic segmentation
    left_camera.add_semantic_segmentation()
    right_camera.add_semantic_segmentation()

    # Enable depth sensing
    left_camera.add_distance_to_image_plane()
    right_camera.add_distance_to_image_plane()

    # Add LiDAR sensor
    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
    lidar_interface.add_segmentation("/World/Robot/Lidar")

    print("Sensors configured with semantic segmentation and depth")

setup_sensors()
```

#### Step 3: Synthetic Dataset Generation
```python
import cv2
import json
from PIL import Image
import os
import random

def generate_synthetic_dataset(num_samples=5000):
    dataset_dir = "synthetic_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(f"{dataset_dir}/images", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels", exist_ok=True)
    os.makedirs(f"{dataset_dir}/depth", exist_ok=True)
    os.makedirs(f"{dataset_dir}/semantic", exist_ok=True)

    annotations = []

    for i in range(num_samples):
        # Randomize environment for variation
        if i % 100 == 0:  # Change lighting every 100 samples
            # Randomize lighting parameters
            light_intensity = random.uniform(0.5, 2.0)
            carb.settings.get_settings().set("/rtx/lightCache/intensityScale", light_intensity)

        # Step simulation to capture new scene
        world.step(render=True)

        # Capture sensor data
        left_rgb = left_camera.get_rgb()
        left_depth = left_camera.get_distance_to_image_plane()
        left_semantic = left_camera.get_semantic_segmentation()

        # Save images
        Image.fromarray(left_rgb).save(f"{dataset_dir}/images/left_{i:04d}.png")
        Image.fromarray((left_depth * 1000).astype(np.uint16)).save(f"{dataset_dir}/depth/left_depth_{i:04d}.png")
        Image.fromarray(left_semantic).save(f"{dataset_dir}/semantic/left_semantic_{i:04d}.png")

        # Create annotation
        annotation = {
            "image_id": i,
            "filename": f"images/left_{i:04d}.png",
            "depth_path": f"depth/left_depth_{i:04d}.png",
            "semantic_path": f"semantic/left_semantic_{i:04d}.png",
            "width": left_rgb.shape[1],
            "height": left_rgb.shape[0],
            "objects": []
        }

        annotations.append(annotation)

    # Save annotations
    with open(f"{dataset_dir}/annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Generated {num_samples} synthetic samples in {dataset_dir}")

# Generate dataset
generate_synthetic_dataset(5000)
```

### Assessment Criteria
- Environment created with at least 5 different object types
- Synthetic dataset with 5000+ samples
- Proper semantic segmentation labels
- Depth and RGB data properly captured
- Dataset exported in standard format (COCO or similar)

## Lab Exercise 2: Isaac ROS VSLAM Perception Model Training and Testing

### Objective
Train a perception model using synthetic data and test it in Isaac Sim with Isaac ROS VSLAM pipeline.

### Setup Instructions
1. Prepare the synthetic dataset from Exercise 1
2. Train an object detection model using the synthetic data
3. Integrate the trained model with Isaac ROS
4. Test the model in Isaac Sim environment

### Implementation Steps

#### Step 1: Perception Model Training
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import json
import cv2
from PIL import Image

class SyntheticDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Load annotations
        with open(f"{dataset_dir}/annotations.json", 'r') as f:
            self.annotations = json.load(f)

        # Create class mapping
        self.class_names = ["background", "table", "chair"]  # Based on our environment
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # Load image
        img_path = os.path.join(self.dataset_dir, annotation['filename'])
        image = Image.open(img_path).convert("RGB")

        # Load semantic segmentation for object detection
        semantic_path = os.path.join(self.dataset_dir, annotation['semantic_path'])
        semantic = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)

        # Create bounding boxes from semantic segmentation
        boxes = []
        labels = []

        for class_id, class_name in enumerate(self.class_names[1:], 1):  # Skip background
            mask = (semantic == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x+w, y+h])
                labels.append(class_id)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

def train_perception_model():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = SyntheticDataset("synthetic_dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.roi_heads.box_predictor = nn.Linear(1024, len(dataset.class_names))  # Adjust for our classes

    # Move to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_perception_model.pth")
    print("Perception model trained and saved as 'trained_perception_model.pth'")

# Train the model
train_perception_model()
```

#### Step 2: Isaac ROS Perception Integration
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import cv2

class IsaacPerceptionInference(Node):
    def __init__(self):
        super().__init__('isaac_perception_inference')

        # Create subscriber for camera input
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Create publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/isaac_ros/detections',
            10
        )

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        self.model.roi_heads.box_predictor = torch.nn.Linear(1024, 3)  # 3 classes: background, table, chair
        self.model.load_state_dict(torch.load('trained_perception_model.pth', map_location='cpu'))
        self.model.eval()

        # Define class names
        self.class_names = ["background", "table", "chair"]

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.get_logger().info('Isaac Perception Inference initialized')

    def image_callback(self, msg):
        """Process image and run perception inference"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            input_tensor = self.transform(rgb_image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)

            # Process predictions
            detections = self.process_predictions(predictions[0], cv_image.shape)

            # Publish detections
            self.publish_detections(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in perception inference: {e}')

    def process_predictions(self, prediction, image_shape):
        """Process model predictions and convert to detections"""
        detections = []

        # Get prediction data
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        # Filter detections by confidence threshold
        confidence_threshold = 0.5
        valid_indices = scores >= confidence_threshold

        for i in valid_indices.nonzero()[0]:
            box = boxes[i]
            label = labels[i]
            score = scores[i]

            # Convert to Detection2D format
            detection = Detection2D()
            detection.bbox.center.x = (box[0] + box[2]) / 2
            detection.bbox.center.y = (box[1] + box[3]) / 2
            detection.bbox.size_x = box[2] - box[0]
            detection.bbox.size_y = box[3] - box[1]

            # Add hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(label)
            hypothesis.hypothesis.score = float(score)
            detection.results.append(hypothesis)

            detections.append(detection)

        return detections

    def publish_detections(self, detections, header):
        """Publish detections to ROS topic"""
        detection_array = Detection2DArray()
        detection_array.header = header
        detection_array.detections = detections

        self.detection_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacPerceptionInference()
    rclpy.spin(perception_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Assessment Criteria
- Model trained on synthetic dataset with >70% mAP
- Successful integration with Isaac ROS
- Real-time inference at >10 FPS
- Accurate detection of objects in simulation environment
- Proper ROS message formatting and publishing

## Lab Exercise 3: Nav2 Bipedal Navigation Integration

### Objective
Implement comprehensive Nav2 navigation for bipedal humanoid robots with integrated perception and path planning.

### Setup Instructions
1. Configure Nav2 with humanoid-specific planners
2. Integrate perception data for dynamic obstacle avoidance
3. Implement footstep planning for stable locomotion
4. Test navigation in complex environments

### Implementation Steps

#### Step 1: Humanoid Nav2 Configuration
```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, Imu
from vision_msgs.msg import Detection2DArray
import numpy as np

class HumanoidNav2Integration(Node):
    def __init__(self):
        super().__init__('humanoid_nav2_integration')

        # Initialize Nav2 action client
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/isaac_ros/detections',
            self.detection_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Initialize dynamic obstacle tracking
        self.dynamic_obstacles = []
        self.imu_data = None

        self.get_logger().info('Humanoid Nav2 Integration initialized')

    def detection_callback(self, msg):
        """Process object detections for dynamic obstacle avoidance"""
        # Update dynamic obstacle list based on detections
        self.dynamic_obstacles = []

        for detection in msg.detections:
            if len(detection.results) > 0:
                # Get the best hypothesis
                best_result = detection.results[0]

                # Convert detection to obstacle (simplified)
                obstacle = {
                    'class_id': best_result.hypothesis.class_id,
                    'confidence': best_result.hypothesis.score,
                    'center_x': detection.bbox.center.x,
                    'center_y': detection.bbox.center.y,
                    'size_x': detection.bbox.size_x,
                    'size_y': detection.bbox.size_y
                }

                self.dynamic_obstacles.append(obstacle)

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        self.imu_data = msg

    def send_navigation_goal(self, x, y, theta):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        from tf_transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        goal_msg.pose.pose.orientation.x = quat[0]
        goal_msg.pose.pose.orientation.y = quat[1]
        goal_msg.pose.pose.orientation.z = quat[2]
        goal_msg.pose.pose.orientation.w = quat[3]

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

def main(args=None):
    rclpy.init(args=args)
    nav_node = HumanoidNav2Integration()

    # Send a test navigation goal
    nav_node.send_navigation_goal(5.0, 5.0, 0.0)

    rclpy.spin(nav_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Advanced Navigation with Perception Integration
```python
import numpy as np
from scipy.spatial.distance import cdist
import math

class AdvancedHumanoidNavigation:
    def __init__(self):
        self.footstep_planner = self.initialize_footstep_planner()
        self.balance_controller = self.initialize_balance_controller()
        self.obstacle_avoidance = self.initialize_obstacle_avoidance()

    def initialize_footstep_planner(self):
        """Initialize footstep planning system"""
        return {
            'step_length': 0.4,
            'step_width': 0.2,
            'min_step_length': 0.1,
            'max_step_angle': 30
        }

    def initialize_balance_controller(self):
        """Initialize balance control system"""
        return {
            'kp_balance': 1.0,
            'kd_balance': 0.1,
            'max_tilt': 0.2
        }

    def initialize_obstacle_avoidance(self):
        """Initialize obstacle avoidance system"""
        return {
            'safety_distance': 0.5,
            'inflation_radius': 0.3,
            'prediction_horizon': 2.0
        }

    def plan_with_obstacle_avoidance(self, path, obstacles):
        """Plan path considering dynamic obstacles"""
        # Inflate obstacles based on safety margin
        inflated_obstacles = []
        for obs in obstacles:
            inflated_obstacles.append({
                'position': obs['position'],
                'radius': obs['radius'] + self.obstacle_avoidance['inflation_radius']
            })

        # Adjust path to avoid obstacles
        adjusted_path = []
        for i, point in enumerate(path):
            # Check distance to obstacles
            safe_point = point
            for obs in inflated_obstacles:
                dist = math.sqrt((point[0] - obs['position'][0])**2 + (point[1] - obs['position'][1])**2)
                if dist < obs['radius']:
                    # Adjust point to avoid obstacle
                    direction = np.array([point[0] - obs['position'][0], point[1] - obs['position'][1]])
                    direction = direction / np.linalg.norm(direction)
                    safe_point = (
                        obs['position'][0] + direction[0] * (obs['radius'] + 0.1),
                        obs['position'][1] + direction[1] * (obs['radius'] + 0.1)
                    )
                    break

            adjusted_path.append(safe_point)

        return adjusted_path

    def generate_footsteps(self, path, robot_state):
        """Generate footsteps for the path with balance considerations"""
        footsteps = []
        current_left_foot = robot_state['left_foot_pose']
        current_right_foot = robot_state['right_foot_pose']
        support_foot = 'left'  # Start with left foot as support

        for i in range(1, len(path)):
            target_pos = path[i]

            # Calculate next step position
            if i > 0:
                direction = np.array([target_pos[0] - path[i-1][0],
                                     target_pos[1] - path[i-1][1]])
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1, 0])

            # Calculate next step
            step_pos = self.calculate_next_step(
                current_left_foot,
                current_right_foot,
                direction,
                support_foot
            )

            # Verify step stability
            if self.is_step_stable(step_pos, current_left_foot, current_right_foot, support_foot):
                footstep = {
                    'position': step_pos,
                    'support_foot': support_foot,
                    'timestamp': i * 0.5,
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
        """Calculate next step position"""
        if support_foot == 'left':
            base_pos = np.array(right_foot)
        else:
            base_pos = np.array(left_foot)

        step_offset = self.footstep_planner['step_length'] * direction
        next_pos = base_pos + step_offset

        # Adjust for step width
        if support_foot == 'left':
            lateral_offset = np.array([-direction[1], direction[0]]) * self.footstep_planner['step_width']/2
        else:
            lateral_offset = np.array([direction[1], -direction[0]]) * self.footstep_planner['step_width']/2

        next_pos += lateral_offset
        return next_pos

    def is_step_stable(self, step_pos, left_foot, right_foot, support_foot):
        """Check if the step maintains stability"""
        if support_foot == 'left':
            support_positions = [left_foot, step_pos]
        else:
            support_positions = [step_pos, right_foot]

        # Calculate center of mass
        com = np.mean(support_positions, axis=0)

        # Check if COM is within support polygon (simplified)
        return True  # Simplified for this example

    def execute_navigation_with_balance(self, footsteps, imu_data):
        """Execute navigation with balance control"""
        execution_result = {
            'success': True,
            'balance_maintained': True,
            'execution_time': 0.0,
            'energy_consumption': 0.0
        }

        # Simulate execution of footsteps with balance control
        for i, footstep in enumerate(footsteps):
            # Check balance using IMU data
            if imu_data and abs(imu_data.roll) > self.balance_controller['max_tilt']:
                execution_result['balance_maintained'] = False
                execution_result['success'] = False
                break

            # Simulate step execution
            execution_result['execution_time'] += 0.5  # 0.5 seconds per step
            execution_result['energy_consumption'] += 1.0  # Energy units per step

        return execution_result

# Usage example
navigation_system = AdvancedHumanoidNavigation()

# Example path and obstacles
path = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]
obstacles = [
    {'position': (2.5, 1.5), 'radius': 0.3},
    {'position': (4.0, 2.5), 'radius': 0.4}
]

# Plan path with obstacle avoidance
safe_path = navigation_system.plan_with_obstacle_avoidance(path, obstacles)

# Robot state
robot_state = {
    'left_foot_pose': [0, 0.1],
    'right_foot_pose': [0, -0.1]
}

# Generate footsteps
footsteps = navigation_system.generate_footsteps(safe_path, robot_state)

# Execute navigation (simulated IMU data)
class IMUData:
    def __init__(self, roll=0.0, pitch=0.0):
        self.roll = roll
        self.pitch = pitch

imu_data = IMUData(roll=0.05, pitch=0.02)
result = navigation_system.execute_navigation_with_balance(footsteps, imu_data)

print(f"Navigation result: {result}")
```

### Assessment Criteria
- Successful path planning with obstacle avoidance
- Stable footstep generation for bipedal locomotion
- Real-time balance control during navigation
- Integration of perception data for dynamic obstacle handling
- Successful navigation completion in simulation

## Lab Exercise 4: Complete AI-Robot Brain System Integration

### Objective
Integrate all Isaac components into a complete AI-robot brain system with perception, simulation, and navigation working together.

### Setup Instructions
1. Set up complete Isaac ecosystem integration
2. Implement perception-to-navigation pipeline
3. Test complete system in complex scenarios
4. Validate system performance and reliability

### Implementation Steps

#### Step 1: Complete System Integration
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import numpy as np
import threading
import time

class CompleteAIHumanoidSystem(Node):
    def __init__(self):
        super().__init__('complete_ai_humanoid_system')

        # Initialize all subsystems
        self.simulation_interface = self.initialize_simulation_interface()
        self.perception_system = self.initialize_perception_system()
        self.navigation_system = self.initialize_navigation_system()
        self.balance_control = self.initialize_balance_control()

        # Create subscribers for all sensor inputs
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.detection_sub = self.create_subscription(Detection2DArray, '/isaac_ros/detections', self.detection_callback, 10)

        # Create publishers for control outputs
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)

        # Initialize system state
        self.current_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'imu_data': None,
            'joint_states': None,
            'detections': [],
            'target_pose': None,
            'navigation_active': False
        }

        # Start main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Complete AI-Humanoid System initialized')

    def initialize_simulation_interface(self):
        """Initialize interface to Isaac Sim"""
        return {
            'connected': True,
            'environment': 'complex_office',
            'robot_model': 'humanoid_a1'
        }

    def initialize_perception_system(self):
        """Initialize perception system"""
        return {
            'model_loaded': True,
            'confidence_threshold': 0.7,
            'detection_classes': ['person', 'table', 'chair', 'obstacle']
        }

    def initialize_navigation_system(self):
        """Initialize navigation system"""
        return {
            'global_planner_ready': True,
            'local_planner_ready': True,
            'footstep_planner_ready': True
        }

    def initialize_balance_control(self):
        """Initialize balance control system"""
        return {
            'balance_controller_ready': True,
            'max_tilt_angle': 0.2,
            'control_frequency': 100
        }

    def image_callback(self, msg):
        """Process camera images"""
        self.get_logger().debug('Received camera image')

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        self.current_state['imu_data'] = {
            'linear_acceleration': np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]),
            'angular_velocity': np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ]),
            'orientation': np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])
        }

    def joint_state_callback(self, msg):
        """Process joint state data"""
        self.current_state['joint_states'] = {
            'names': msg.name,
            'positions': msg.position,
            'velocities': msg.velocity,
            'effort': msg.effort
        }

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_state['position'] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        self.current_state['orientation'] = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        self.current_state['velocity'] = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

    def detection_callback(self, msg):
        """Process object detections"""
        self.current_state['detections'] = []
        for detection in msg.detections:
            if len(detection.results) > 0:
                best_result = detection.results[0]
                det_info = {
                    'class_id': best_result.hypothesis.class_id,
                    'confidence': best_result.hypothesis.score,
                    'bbox_center': (detection.bbox.center.x, detection.bbox.center.y),
                    'bbox_size': (detection.bbox.size_x, detection.bbox.size_y)
                }
                self.current_state['detections'].append(det_info)

    def control_loop(self):
        """Main control loop for integrated system"""
        try:
            # Check system health
            if not self.check_system_health():
                self.emergency_stop()
                return

            # Process perception data
            self.process_perception_data()

            # Plan navigation if target set
            if self.current_state['target_pose'] is not None and not self.current_state['navigation_active']:
                self.plan_navigation_to_target()

            # Execute navigation if active
            if self.current_state['navigation_active']:
                cmd_vel = self.execute_navigation_step()
                self.cmd_vel_pub.publish(cmd_vel)

            # Update system status
            self.publish_system_status()

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            self.emergency_stop()

    def check_system_health(self):
        """Check if all systems are operational"""
        # Check if we have recent sensor data
        if self.current_state['imu_data'] is None:
            self.get_logger().warn('No IMU data received')
            return False

        if self.current_state['joint_states'] is None:
            self.get_logger().warn('No joint state data received')
            return False

        # Check balance - if tilt is too high, stop
        if self.current_state['imu_data']:
            orientation = self.current_state['imu_data']['orientation']
            # Simplified balance check
            # In practice, this would involve more complex balance algorithms
            pass

        return True

    def process_perception_data(self):
        """Process perception data and update world model"""
        # Update world model based on detections
        for detection in self.current_state['detections']:
            if detection['confidence'] > self.perception_system['confidence_threshold']:
                # Add object to world model
                self.update_world_model(detection)

    def update_world_model(self, detection):
        """Update internal world model with detected objects"""
        # In a real system, this would update a 3D map with object positions
        # For simulation, we'll just log the detection
        self.get_logger().info(f'Detected {detection["class_id"]} with confidence {detection["confidence"]:.2f}')

    def plan_navigation_to_target(self):
        """Plan navigation to target pose"""
        if self.current_state['target_pose'] is None:
            return

        # In a real system, this would call the global planner
        # For this example, we'll just set navigation as active
        self.current_state['navigation_active'] = True
        self.get_logger().info(f'Navigation to target started: {self.current_state["target_pose"]}')

    def execute_navigation_step(self):
        """Execute one step of navigation"""
        cmd_vel = Twist()

        # Check if we need to stop (reached target or emergency)
        if self.has_reached_target():
            self.current_state['navigation_active'] = False
            self.get_logger().info('Target reached, navigation completed')
            return cmd_vel  # Zero velocity to stop

        # Calculate desired velocity based on target
        target_pos = self.current_state['target_pose']['position']
        current_pos = self.current_state['position']

        # Calculate direction to target
        direction = target_pos - current_pos[:2]  # Only x,y for 2D navigation
        distance = np.linalg.norm(direction)

        if distance > 0.2:  # If not very close to target
            direction = direction / distance  # Normalize

            # Set linear velocity proportional to distance (slow down when close)
            cmd_vel.linear.x = min(0.3, distance * 0.5)  # Max 0.3 m/s
            cmd_vel.linear.y = 0.0
            cmd_vel.linear.z = 0.0

            # Set angular velocity to face target
            current_yaw = self.get_yaw_from_orientation(self.current_state['orientation'])
            desired_yaw = np.arctan2(direction[1], direction[0])

            angle_diff = desired_yaw - current_yaw
            # Normalize angle difference
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            cmd_vel.angular.z = max(-0.5, min(0.5, angle_diff * 1.0))  # Max 0.5 rad/s
        else:
            # Very close to target, stop
            self.current_state['navigation_active'] = False
            self.get_logger().info('Target reached, navigation completed')

        return cmd_vel

    def has_reached_target(self):
        """Check if robot has reached the target"""
        if self.current_state['target_pose'] is None:
            return True

        target_pos = self.current_state['target_pose']['position']
        current_pos = self.current_state['position']

        distance = np.linalg.norm(target_pos - current_pos[:2])
        return distance < 0.3  # 30cm tolerance

    def get_yaw_from_orientation(self, orientation):
        """Extract yaw from quaternion orientation"""
        x, y, z, w = orientation
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def publish_system_status(self):
        """Publish system status"""
        status_msg = String()
        status_msg.data = f"Position: ({self.current_state['position'][0]:.2f}, {self.current_state['position'][1]:.2f}), " \
                         f"Navigation: {'Active' if self.current_state['navigation_active'] else 'Inactive'}, " \
                         f"Detections: {len(self.current_state['detections'])}"
        self.status_pub.publish(status_msg)

    def emergency_stop(self):
        """Emergency stop all motion"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)
        self.current_state['navigation_active'] = False
        self.get_logger().warn('Emergency stop activated')

    def set_target_pose(self, x, y, z=0.0):
        """Set navigation target pose"""
        self.current_state['target_pose'] = {
            'position': np.array([x, y]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0])
        }
        self.get_logger().info(f'Target pose set to: ({x}, {y})')

def main(args=None):
    rclpy.init(args=args)
    ai_system = CompleteAIHumanoidSystem()

    # Set a test target after a short delay
    def set_test_target():
        time.sleep(2)  # Wait for systems to initialize
        ai_system.set_target_pose(5.0, 5.0)

    # Start target setting in a separate thread
    target_thread = threading.Thread(target=set_test_target)
    target_thread.start()

    try:
        rclpy.spin(ai_system)
    except KeyboardInterrupt:
        pass
    finally:
        ai_system.emergency_stop()
        rclpy.shutdown()
        target_thread.join()

if __name__ == '__main__':
    main()
```

### Assessment Criteria
- Complete integration of all Isaac components
- Successful perception-to-navigation pipeline
- Real-time system performance
- Robust error handling and safety measures
- Successful completion of complex navigation tasks

## Comprehensive Debugging Guide

### System Architecture Debugging

#### Isaac Sim Debugging
1. **Simulation Performance Issues**:
   - Check GPU memory usage with `nvidia-smi`
   - Reduce scene complexity if frame rate is low
   - Verify CUDA and driver compatibility
   - Check for memory leaks in long-running simulations

2. **Sensor Data Issues**:
   - Verify sensor prim paths and connections
   - Check sensor configuration parameters
   - Validate data format and resolution
   - Confirm proper calibration

#### Isaac ROS Debugging
1. **VSLAM Pipeline Issues**:
   - Check camera calibration parameters
   - Verify feature detection and tracking
   - Validate IMU integration and synchronization
   - Monitor computational performance

2. **Perception Pipeline Issues**:
   - Verify model input/output dimensions
   - Check data preprocessing pipelines
   - Validate ROS message formats
   - Monitor inference performance

#### Nav2 Debugging
1. **Path Planning Issues**:
   - Verify map quality and resolution
   - Check costmap parameters
   - Validate local planner configuration
   - Monitor footprint and inflation settings

2. **Bipedal Navigation Issues**:
   - Check footstep planning algorithms
   - Validate balance control parameters
   - Monitor joint state feedback
   - Verify gait pattern execution

### Common Debugging Commands

#### Isaac Sim
```bash
# Check Isaac Sim status
isaac-sim --version

# Launch with verbose logging
isaac-sim --/app/window/hideAdvancedSettings=False --/app/showDevTools=True

# Monitor GPU usage during simulation
watch -n 1 nvidia-smi
```

#### Isaac ROS
```bash
# Check ROS 2 nodes
ros2 node list

# Check ROS 2 topics
ros2 topic list

# Monitor specific topic
ros2 topic echo /camera/rgb/image_raw

# Check Isaac ROS specific nodes
ros2 run isaac_ros_apriltag isaac_ros_apriltag_node
```

#### Nav2
```bash
# Launch Nav2 with debugging
ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  autostart:=true \
  params_file:=/path/to/nav2_params.yaml

# Monitor Nav2 status
ros2 action list
ros2 service list
```

### Performance Optimization

#### Isaac Sim Optimization
```python
# Rendering optimization settings
carb.settings.get_settings().set("/rtx/sceneDb/lightStepSize", 0.1)
carb.settings.get_settings().set("/rtx/sceneDb/mediumStepSize", 0.01)
carb.settings.get_settings().set("/rtx/indirectDiffuseLighting/quality", 1)  # Medium quality
carb.settings.get_settings().set("/rtx/directLighting/quality", 1)  # Medium quality
carb.settings.get_settings().set("/rtx/reflections/quality", 0)  # Low quality for performance
```

#### Isaac ROS Optimization
```python
# Processing optimization
# Reduce image resolution for faster processing
image_processing_params = {
    'input_resolution': (640, 480),  # Lower resolution for speed
    'processing_frequency': 10,      # Process every 100ms instead of every frame
    'feature_count': 500             # Reduce number of features to track
}
```

#### System Integration Optimization
```python
# Multi-threading for better performance
import threading
from concurrent.futures import ThreadPoolExecutor

def optimize_system_performance():
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Process perception in parallel
        perception_future = executor.submit(process_perception)

        # Plan navigation in parallel
        navigation_future = executor.submit(plan_navigation)

        # Update UI in parallel
        ui_future = executor.submit(update_ui)

        # Wait for all tasks to complete
        perception_result = perception_future.result()
        navigation_result = navigation_future.result()
        ui_result = ui_future.result()
```

### Troubleshooting Checklist

#### Pre-Deployment Checklist
- [ ] Isaac Sim environment configured and tested
- [ ] Isaac ROS VSLAM pipeline validated
- [ ] Nav2 navigation system calibrated
- [ ] Sensor data quality verified
- [ ] Safety systems tested
- [ ] Performance benchmarks met

#### Runtime Monitoring
- [ ] Monitor system resource usage
- [ ] Track navigation success rates
- [ ] Log perception accuracy metrics
- [ ] Monitor balance stability
- [ ] Record system performance data

## Summary

This lesson provided comprehensive lab exercises that integrate Isaac Sim, Isaac ROS, and Nav2 concepts to validate your understanding of the complete AI-robot brain system. You implemented perception model training with synthetic data, integrated perception with navigation, and created a complete AI-robot brain system that combines simulation, perception, and navigation capabilities.

The debugging guide provided systematic approaches to troubleshooting complex Isaac platform issues, with specific guidance for each component and performance optimization techniques. The assessment criteria ensure that learners can validate their implementations and achieve competency in the complete AI-robot brain system.