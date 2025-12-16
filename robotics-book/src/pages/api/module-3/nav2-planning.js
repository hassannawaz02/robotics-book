/**
 * Mock API for JWT-protected Nav2 lesson access
 * In a real implementation, this would be handled by a backend server
 */

// This is a mock API endpoint that demonstrates JWT-protected access for Nav2 lessons
// In a real application, this would be implemented with a proper backend

export default function handler(req, res) {
  // In a real implementation, we would:
  // 1. Extract JWT token from Authorization header
  // 2. Verify the token
  // 3. Check if user has access to requested Nav2 lesson
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

  // Check if lesson exists and is a Nav2 lesson
  const validLessons = ['04-nav2-planning'];

  if (!lessonId || !validLessons.includes(lessonId)) {
    return res.status(404).json({
      error: {
        code: 'LESSON_NOT_FOUND',
        message: 'Requested Nav2 lesson not found.'
      }
    });
  }

  // Return lesson content (in real app, this would come from a database or file system)
  const lesson = {
    id: '04-nav2-planning',
    title: 'Nav2 Path Planning',
    content: 'Navigation2 (Nav2) is the navigation stack for ROS 2, providing path planning and execution capabilities...',
    module: 'module-3',
    lessonNumber: 4,
    type: 'lesson',
    assets: ['../assets/diagrams/isaac-architecture.svg'],
    codeExamples: [
      '../code-examples/python/isaac-scripts/nav2-path-planning.py',
      '../code-examples/python/isaac-scripts/bipedal-path-planning.py'
    ]
  };

  // Return lesson data
  res.status(200).json(lesson);
}