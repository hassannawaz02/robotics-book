---
title: URDF - Unified Robot Description Format
sidebar_position: 4
---

# URDF - Unified Robot Description Format

## Overview

URDF (Unified Robot Description Format) is an XML-based format used to describe robots in ROS. It defines the physical structure of a robot, including its links (rigid parts), joints (connections between links), visual and collision properties, and other aspects. For humanoid robots, URDF is particularly important as it describes the complex kinematic structure with multiple limbs and degrees of freedom.

## What is URDF?

URDF stands for **Unified Robot Description Format**. It's an XML format that describes:
- **Links**: Rigid parts of the robot (e.g., torso, arms, legs)
- **Joints**: Connections between links (e.g., revolute, prismatic, fixed)
- **Visual properties**: How the robot appears in simulation and visualization
- **Collision properties**: How the robot interacts with its environment
- **Inertial properties**: Mass, center of mass, and inertia for physics simulation

## URDF Structure

A basic URDF file has this structure:

```xml
<robot name="robot_name">
  <link name="link_name">
    <inertial>
      <!-- Mass, origin, and inertia properties -->
    </inertial>
    <visual>
      <!-- Visual representation -->
    </visual>
    <collision>
      <!-- Collision representation -->
    </collision>
  </link>

  <joint name="joint_name" type="joint_type">
    <parent link="parent_link_name"/>
    <child link="child_link_name"/>
    <origin xyz="x y z" rpy="roll pitch yaw"/>
  </joint>
</robot>
```

## Links - The Building Blocks

Links represent rigid bodies in the robot. Each link contains:

### Inertial Properties
```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
</inertial>
```

### Visual Properties
```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.1"/>
    <!-- or <cylinder radius="0.1" length="0.2"/> -->
    <!-- or <sphere radius="0.1"/> -->
    <!-- or <mesh filename="mesh.stl"/> -->
  </geometry>
</visual>
```

### Collision Properties
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>
```

## Joints - Connecting Links

Joints define how links connect and move relative to each other. Common joint types:

- **Fixed**: No movement between links
- **Revolute**: Rotational movement around an axis
- **Continuous**: Like revolute but unlimited rotation
- **Prismatic**: Linear sliding movement
- **Planar**: Movement in a plane
- **Floating**: 6-DOF movement

Example joint:
```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
</joint>
```

## URDF for Humanoid Robots

Humanoid robots have a complex structure that URDF describes through:

### Main Body Structure
- **Base link**: Usually the torso or pelvis
- **Head**: Connected to torso
- **Arms**: Left and right, with multiple segments (upper arm, lower arm, hand)
- **Legs**: Left and right, with multiple segments (thigh, shin, foot)

### Kinematic Chains
URDF defines kinematic chains from the base to each end effector (hands, feet), which is crucial for:
- Forward kinematics: calculating end effector position from joint angles
- Inverse kinematics: calculating joint angles for desired end effector position

## URDF Example

Here's a simplified example of a humanoid URDF structure:

```xml
<robot name="simple_humanoid">
  <!-- Base link (torso) -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.8"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint connecting head to torso -->
  <joint name="neck_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <!-- Additional links and joints for arms and legs would follow -->
</robot>
```

## Working with URDF in Python

Here's an example of working with URDF in Python (from our code examples):

```python
#!/usr/bin/env python3
"""
URDF Examples for Humanoid Robots
This script demonstrates how to work with URDF (Unified Robot Description Format) for humanoid robots.
"""

import xml.etree.ElementTree as ET


def create_simple_humanoid_urdf():
    """
    Create a simple example of a humanoid URDF structure.
    This is a basic representation showing the main components of a humanoid robot.
    """
    # Create the root robot element
    robot = ET.Element('robot', name='simple_humanoid')

    # Add a base link (torso)
    base_link = ET.SubElement(robot, 'link', name='base_link')
    inertial = ET.SubElement(base_link, 'inertial')
    mass = ET.SubElement(inertial, 'mass', value='10.0')
    origin = ET.SubElement(inertial, 'origin', xyz='0 0 0')
    inertia = ET.SubElement(inertial, 'inertia', ixx='1.0', ixy='0.0', ixz='0.0', iyy='1.0', iyz='0.0', izz='1.0')

    visual = ET.SubElement(base_link, 'visual')
    origin = ET.SubElement(visual, 'origin', xyz='0 0 0')
    geometry = ET.SubElement(visual, 'geometry')
    box = ET.SubElement(geometry, 'box', size='0.5 0.3 0.8')

    collision = ET.SubElement(base_link, 'collision')
    origin = ET.SubElement(collision, 'origin', xyz='0 0 0')
    geometry = ET.SubElement(collision, 'geometry')
    box = ET.SubElement(geometry, 'box', size='0.5 0.3 0.8')

    # Add a head link
    head_link = ET.SubElement(robot, 'link', name='head')
    # ... (similar structure for head)

    # Add a joint connecting head to base
    head_joint = ET.SubElement(robot, 'joint', name='head_joint', type='fixed')
    parent = ET.SubElement(head_joint, 'parent', link='base_link')
    child = ET.SubElement(head_joint, 'child', link='head')
    origin = ET.SubElement(head_joint, 'origin', xyz='0 0 0.5')

    # Convert to string for display
    urdf_string = ET.tostring(robot, encoding='unicode')

    # Format with proper indentation
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(urdf_string)
    pretty_xml = dom.toprettyxml()

    return pretty_xml


def parse_urdf_file(file_path):
    """
    Parse an existing URDF file and extract information.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        print(f"Parsing URDF file: {file_path}")
        print(f"Robot name: {root.get('name')}")

        # Count links and joints
        links = root.findall('link')
        joints = root.findall('joint')

        print(f"Number of links: {len(links)}")
        print(f"Number of joints: {len(joints)}")

        # Print link names
        print("Links in the robot:")
        for link in links:
            print(f"  - {link.get('name')}")

        # Print joint names and types
        print("Joints in the robot:")
        for joint in joints:
            print(f"  - {joint.get('name')} (type: {joint.get('type')})")

    except ET.ParseError as e:
        print(f"Error parsing URDF file: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")


def demonstrate_urdf_concepts():
    """
    Demonstrate key concepts about URDF for humanoid robots.
    """
    print("=== URDF Concepts for Humanoid Robots ===\n")

    print("1. Links:")
    print("   - Represent rigid bodies in the robot")
    print("   - Each link has properties like mass, inertia, visual, and collision")
    print("   - Examples: torso, head, arms, legs\n")

    print("2. Joints:")
    print("   - Connect links together")
    print("   - Types: fixed, revolute, continuous, prismatic, etc.")
    print("   - Define how links can move relative to each other\n")

    print("3. Common Humanoid Structure:")
    print("   - Base (torso/trunk)")
    print("   - Head")
    print("   - Arms (upper arm, lower arm, hand)")
    print("   - Legs (upper leg, lower leg, foot)\n")

    print("4. URDF for Humanoids:")
    print("   - Defines kinematic chain from torso to limbs")
    print("   - Includes visual and collision properties")
    print("   - May include transmission elements for actuators\n")


if __name__ == '__main__':
    print("=== URDF Examples for Humanoid Robots ===\n")

    # Demonstrate URDF concepts
    demonstrate_urdf_concepts()

    # Create and display a simple URDF example
    print("=== Simple Humanoid URDF Example ===\n")
    simple_urdf = create_simple_humanoid_urdf()
    print(simple_urdf)

    print("\n=== Example Usage ===")
    print("To parse an existing URDF file, call:")
    print("parse_urdf_file('path/to/your/robot.urdf')")
```

## URDF Tools and Visualization

ROS provides several tools for working with URDF:

- **RViz**: 3D visualization of robot models
- **Robot State Publisher**: Publishes robot joint states for visualization
- **TF (Transforms)**: Manages coordinate frame relationships
- **URDF Parser**: Various libraries for parsing and validating URDF

## Best Practices for URDF

1. **Start Simple**: Begin with a basic model and add complexity gradually
2. **Use Standard Formats**: Follow established conventions for joint names and conventions
3. **Validate Regularly**: Use tools to validate URDF syntax and kinematics
4. **Include All Links**: Ensure all physical components are represented
5. **Proper Inertial Values**: Use realistic mass and inertia values for physics simulation

## URDF in Humanoid Robotics Applications

URDF is essential for humanoid robotics because it:
- Enables physics simulation for testing control algorithms
- Provides visualization for debugging and monitoring
- Supports motion planning by defining collision geometry
- Facilitates kinematic calculations for walking and manipulation
- Allows for standardized robot descriptions across the community

## Exercises

1. Create a URDF file for a simple 3-link robot arm and visualize it in RViz.
2. Research and describe the URDF structure of a real humanoid robot (e.g., NAO, Pepper, or Atlas).
3. Explain how URDF enables motion planning for humanoid robots.

---

**Continue to [Exercises](./05-exercises.md) to practice what you've learned.**