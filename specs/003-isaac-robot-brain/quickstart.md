# Quickstart Guide: Module 3 - AI-Robot Brain (NVIDIA Isaac™)

## Prerequisites

1. **NVIDIA Isaac Sim** installed and configured
2. **Python 3.11+** environment
3. **Docusaurus 2** frontend environment
4. **Spec-Kit Plus** and **Claude Code** tools

## Setup Steps

### 1. Environment Setup
```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install dependencies
npm install  # For Docusaurus frontend

# Ensure Python 3.11+ is available
python --version
```

### 2. Module Structure Creation
```bash
# Navigate to the robotics book directory
cd robotics-book/modules/

# Create module-3 directory structure
mkdir -p module-3/lessons
mkdir -p module-3/assets/diagrams
mkdir -p module-3/assets/images
mkdir -p module-3/code-examples/python/isaac-scripts
```

### 3. Lesson Content Generation
```bash
# Create the 5 required lesson files
touch module-3/lessons/01-intro.md
touch module-3/lessons/02-isaac-sim.md
touch module-3/lessons/03-isaac-ros.md
touch module-3/lessons/04-nav2-planning.md
touch module-3/lessons/05-exercises.md
```

### 4. Authentication Setup
```bash
# Configure JWT authentication for lesson pages
# This typically involves setting up auth middleware in your Docusaurus setup
# Configuration will vary based on your specific auth system
```

### 5. Content Integration
```bash
# Build the Docusaurus site to integrate the new module
npm run build

# Start the development server
npm start
```

## Key Components

### Lesson Files
- `01-intro.md` - Introduction to Isaac ecosystem
- `02-isaac-sim.md` - Isaac Sim fundamentals and simulation
- `03-isaac-ros.md` - Isaac ROS navigation and VSLAM
- `04-nav2-planning.md` - Nav2 path planning for bipedal movement
- `05-exercises.md` - Integrated exercises and assessments

### Code Examples
- Located in `code-examples/python/isaac-scripts/`
- Include Isaac ROS scripts and simulation examples
- Compatible with Isaac Sim environment

### Assets
- Diagrams in `assets/diagrams/` (architecture diagrams, system overviews)
- Images in `assets/images/` (screenshots, illustrations)

## Testing

### 1. Content Verification
```bash
# Verify all lesson files exist
ls -la robotics-book/modules/module-3/lessons/

# Check that Docusaurus sidebar shows Module 3
# This should be visible when running the development server
```

### 2. Authentication Testing
```bash
# Test JWT-protected access
# Attempt to access lesson pages without authentication
# Verify that unauthenticated users are redirected to login
```

### 3. Build Verification
```bash
# Build the complete site
npm run build

# Check that all module content is included in the build
# Verify that links work correctly in the built version
```

## Common Issues and Solutions

### Issue: Lesson pages not showing in sidebar
**Solution**: Verify that the Docusaurus sidebar configuration includes the new module

### Issue: JWT authentication not working
**Solution**: Check that the auth middleware is properly configured and tokens are being validated

### Issue: Code examples not displaying correctly
**Solution**: Verify that code examples are in the correct directory and have proper syntax highlighting

## Next Steps

1. Implement the 5 lesson files with content following the specification
2. Add Isaac ROS code examples to the appropriate directory
3. Create architecture diagrams for Isaac Sim, ROS, and Nav2 integration
4. Test authentication flow with JWT tokens
5. Verify that all success criteria from the specification are met