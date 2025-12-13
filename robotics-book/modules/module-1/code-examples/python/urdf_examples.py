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