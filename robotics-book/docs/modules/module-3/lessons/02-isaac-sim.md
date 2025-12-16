---
title: Isaac Sim Fundamentals
sidebar_label: Isaac Sim
---

# Isaac Sim Fundamentals

NVIDIA Isaac Sim is a photorealistic simulation application and synthetic data generation tool for robotics. It provides a virtual environment for testing and training robotic systems.

![Isaac Architecture](../assets/diagrams/isaac-architecture.svg)

## Key Features

- **Photorealistic Simulation**: High-fidelity physics and rendering
- **Synthetic Data Generation**: Create labeled training data
- **Hardware Acceleration**: Leverage NVIDIA GPUs for performance
- **ROS/ROS2 Integration**: Seamless integration with robotics frameworks

## Getting Started

To begin with Isaac Sim, ensure you have the following installed:
- NVIDIA Isaac Sim
- Compatible hardware (NVIDIA GPU)
- Python 3.11+

### Basic Setup

```python
# Isaac Sim setup example
import omni
from omni.isaac.kit import SimulationApp

# Initialize simulation
config = {"headless": False}
simulation_app = SimulationApp(config)
```

## Simulation Concepts

- Scenes and environments
- Robot models and assets
- Physics properties
- Sensors and cameras

## Integration with Isaac ROS and Nav2

Isaac Sim works as the foundation for the entire Isaac ecosystem. The simulation environment allows you to test Isaac ROS perception algorithms and Nav2 path planning in a safe, controlled environment before deploying to real robots.