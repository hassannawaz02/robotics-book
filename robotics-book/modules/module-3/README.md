# Module 3 - AI-Robot Brain (NVIDIA Isaac™)

Welcome to Module 3 of the Humanoid Robotics Interactive Textbook. This module focuses on NVIDIA Isaac™ technologies including Isaac Sim, Isaac ROS, and Nav2 for path planning in bipedal humanoid movement.

## Overview

This module covers:
- NVIDIA Isaac Sim for photorealistic simulation and synthetic data generation
- Isaac ROS for hardware-accelerated VSLAM and navigation
- Nav2 for path planning specifically for bipedal humanoid movement

## Prerequisites

Before starting this module, ensure you have:

1. **NVIDIA Isaac Sim** installed and configured
2. **Python 3.11+** environment
3. **ROS 2** (compatible with Isaac ROS)
4. **Compatible hardware** (NVIDIA GPU recommended)

## Module Structure

The module is organized as follows:

- `lessons/` - Contains the 5 lesson files:
  - `01-intro.md` - Introduction to Isaac Ecosystem
  - `02-isaac-sim.md` - Isaac Sim Fundamentals
  - `03-isaac-ros.md` - Isaac ROS Navigation
  - `04-nav2-planning.md` - Nav2 Path Planning
  - `05-exercises.md` - Module Exercises and Assessments

- `assets/` - Contains diagrams and images:
  - `diagrams/` - Architecture diagrams
  - `images/` - Example screenshots and illustrations

- `code-examples/` - Contains Python code examples:
  - `python/isaac-scripts/` - Isaac-specific scripts and tutorials

## Lessons

Each lesson builds on the previous one:

1. **Introduction**: Overview of the Isaac ecosystem and learning objectives
2. **Isaac Sim**: Simulation fundamentals and setup
3. **Isaac ROS**: Navigation and perception concepts
4. **Nav2 Planning**: Path planning for bipedal movement
5. **Exercises**: Integrated challenges combining all concepts

## Code Examples

The module includes several Python code examples demonstrating Isaac concepts:

- `isaac-sim-setup.py` - Basic Isaac Sim initialization
- `vslam-tutorial.py` - Visual SLAM implementation
- `navigation-tutorial.py` - Isaac ROS navigation
- `nav2-path-planning.py` - Nav2 path planning
- `bipedal-path-planning.py` - Bipedal movement planning
- `exercise-solutions/` - Integrated solution examples

## Authentication

All lessons require JWT-based authentication. Make sure you have valid credentials before accessing the content.

## Getting Started

1. Ensure all prerequisites are installed
2. Navigate to the lessons in the Docusaurus interface
3. Start with `01-intro.md` and progress sequentially
4. Complete the code examples as you go
5. Finish with the integrated exercises in `05-exercises.md`

## Support

For questions about this module, use the AI assistant chat interface available on each lesson page (backend functionality deferred to future RAG implementation).