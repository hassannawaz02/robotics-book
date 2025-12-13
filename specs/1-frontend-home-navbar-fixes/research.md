# Research: Frontend Fixes — Home + Navbar

## Decision: Navigation Bar Changes
- **What was chosen**: Modify the navbar configuration in `docusaurus.config.ts` to remove GitHub button and replace "Tutorial" with "Home"
- **Rationale**: Docusaurus allows navbar customization through the config file, which is the standard approach for such changes

## Decision: Homepage Module Cards Implementation
- **What was chosen**: Modify the `HomepageFeatures` component to display 4 robotics modules with icons and descriptions
- **Rationale**: The existing `HomepageFeatures` component already provides the structure for feature cards, so we'll customize it for robotics modules

## Decision: Module Selection for Homepage
- **What was chosen**: Use the 4 main robotics modules mentioned in the project: ROS 2, Simulation (Gazebo/Unity), NVIDIA Isaac™, and VLA (Vision-Language-Action)
- **Rationale**: These are the core modules mentioned in the project constitution and represent the main learning areas

## Alternatives considered:
1. For navbar changes:
   - Creating custom Navbar component vs. modifying docusaurus.config.ts
   - Chose config modification as it's simpler and follows Docusaurus patterns

2. For homepage cards:
   - Creating new component vs. modifying existing HomepageFeatures
   - Chose to modify existing component as it already has the proper structure

## Technical Implementation Plan:
1. Update `docusaurus.config.ts` to:
   - Change "Tutorial" to "Home" in navbar items
   - Remove the GitHub link completely

2. Update `HomepageFeatures/index.tsx` to:
   - Replace current features with 4 robotics modules
   - Use appropriate icons for each module
   - Add short descriptions for each module

3. Update `HomepageFeatures/styles.module.css` if needed for styling adjustments