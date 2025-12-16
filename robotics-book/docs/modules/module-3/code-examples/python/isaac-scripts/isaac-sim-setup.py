#!/usr/bin/env python3
"""
Isaac Sim Setup Example
This script demonstrates basic Isaac Sim initialization and setup
"""

import omni
from omni.isaac.kit import SimulationApp
import carb


def main():
    # Initialize simulation application
    config = {
        "headless": False,  # Set to True for headless mode
        "window_width": 1280,
        "window_height": 720
    }

    # Create simulation app
    simulation_app = SimulationApp(config)

    print("Isaac Sim initialized successfully!")

    # Get the world interface
    world = omni.isaac.core.World.World()

    # Add a simple robot to the simulation (example with a basic cube)
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.stage import add_reference_to_stage

    # Create a simple cube as a placeholder robot
    from omni.isaac.core.utils.prims import create_primitive
    cube = create_primitive(
        prim_path="/World/Cube",
        primitive_props={"size": 0.1},
        position=[0, 0, 1.0],
        orientation=[0, 0, 0, 1]
    )

    # Reset the world to apply changes
    world.reset()

    # Run simulation for a few steps
    for i in range(100):
        world.step(render=True)
        if i % 20 == 0:
            print(f"Simulation step: {i}")

    # Close the simulation
    simulation_app.close()


if __name__ == "__main__":
    main()