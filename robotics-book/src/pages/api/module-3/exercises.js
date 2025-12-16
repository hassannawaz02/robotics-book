/**
 * Mock API for JWT-protected exercises access
 * In a real implementation, this would be handled by a backend server
 */

// This is a mock API endpoint that demonstrates JWT-protected access for exercises
// In a real application, this would be implemented with a proper backend

export default function handler(req, res) {
  // In a real implementation, we would:
  // 1. Extract JWT token from Authorization header
  // 2. Verify the token
  // 3. Check if user has access to requested exercises
  // 4. Return exercise content if authorized

  const { exerciseId } = req.query || {};

  // Mock token validation (in real app, use proper JWT verification)
  const authHeader = req.headers.authorization;
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({
      error: {
        code: 'AUTH_REQUIRED',
        message: 'Access denied. No token provided.'
      }
    });
  }

  // Mock token verification (in real app, use jwt.verify())
  // For this mock, we'll just check if token is not empty
  if (token.length < 10) {
    return res.status(401).json({
      error: {
        code: 'INVALID_TOKEN',
        message: 'Invalid or expired token.'
      }
    });
  }

  // Check if exercise exists
  const validExercises = ['05-exercises'];

  if (!exerciseId || !validExercises.includes(exerciseId)) {
    return res.status(404).json({
      error: {
        code: 'EXERCISE_NOT_FOUND',
        message: 'Requested exercise not found.'
      }
    });
  }

  // Return exercise content (in real app, this would come from a database or file system)
  const exercise = {
    id: '05-exercises',
    title: 'Module Exercises and Assessments',
    content: 'This section contains hands-on exercises that integrate Isaac Sim, Isaac ROS, and Nav2 concepts...',
    module: 'module-3',
    lessonNumber: 5,
    type: 'exercise',
    assets: ['../assets/diagrams/isaac-architecture.svg'],
    codeExamples: [
      '../code-examples/python/isaac-scripts/isaac-sim-setup.py',
      '../code-examples/python/isaac-scripts/vslam-tutorial.py',
      '../code-examples/python/isaac-scripts/navigation-tutorial.py',
      '../code-examples/python/isaac-scripts/nav2-path-planning.py',
      '../code-examples/python/isaac-scripts/bipedal-path-planning.py',
      '../code-examples/python/isaac-scripts/exercise-solutions/integrated-solution.py'
    ]
  };

  // Return exercise data
  res.status(200).json(exercise);
}