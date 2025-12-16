/**
 * Mock API for JWT-protected Isaac ROS lesson access
 * In a real implementation, this would be handled by a backend server
 */

// This is a mock API endpoint that demonstrates JWT-protected access for Isaac ROS lessons
// In a real application, this would be implemented with a proper backend

export default function handler(req, res) {
  // In a real implementation, we would:
  // 1. Extract JWT token from Authorization header
  // 2. Verify the token
  // 3. Check if user has access to requested Isaac ROS lesson
  // 4. Return lesson content if authorized

  const { lessonId } = req.query || {};

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

  // Check if lesson exists and is an Isaac ROS lesson
  const validLessons = ['03-isaac-ros'];

  if (!lessonId || !validLessons.includes(lessonId)) {
    return res.status(404).json({
      error: {
        code: 'LESSON_NOT_FOUND',
        message: 'Requested Isaac ROS lesson not found.'
      }
    });
  }

  // Return lesson content (in real app, this would come from a database or file system)
  const lesson = {
    id: '03-isaac-ros',
    title: 'Isaac ROS Navigation',
    content: 'Isaac ROS provides hardware-accelerated perception and navigation capabilities...',
    module: 'module-3',
    lessonNumber: 3,
    type: 'lesson',
    assets: ['../assets/diagrams/isaac-architecture.svg'],
    codeExamples: [
      '../code-examples/python/isaac-scripts/vslam-tutorial.py',
      '../code-examples/python/isaac-scripts/navigation-tutorial.py'
    ]
  };

  // Return lesson data
  res.status(200).json(lesson);
}