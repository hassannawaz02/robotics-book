/**
 * Mock API for JWT-protected lesson access
 * In a real implementation, this would be handled by a backend server
 */

// This is a mock API endpoint that demonstrates JWT-protected access
// In a real application, this would be implemented with a proper backend

export default function handler(req, res) {
  // In a real implementation, we would:
  // 1. Extract JWT token from Authorization header
  // 2. Verify the token
  // 3. Check if user has access to requested lesson
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

  // Check if lesson exists
  const validLessons = ['01-intro', '02-isaac-sim', '03-isaac-ros', '04-nav2-planning', '05-exercises'];

  if (!lessonId || !validLessons.includes(lessonId)) {
    return res.status(404).json({
      error: {
        code: 'LESSON_NOT_FOUND',
        message: 'Requested lesson not found.'
      }
    });
  }

  // Return lesson content (in real app, this would come from a database or file system)
  const lessons = {
    '01-intro': {
      id: '01-intro',
      title: 'Introduction to Isaac Ecosystem',
      content: 'Welcome to Module 3 - AI-Robot Brain, focusing on NVIDIA Isaac™ technologies...',
      module: 'module-3',
      lessonNumber: 1,
      type: 'lesson'
    },
    '02-isaac-sim': {
      id: '02-isaac-sim',
      title: 'Isaac Sim Fundamentals',
      content: 'NVIDIA Isaac Sim is a photorealistic simulation application and synthetic data generation tool...',
      module: 'module-3',
      lessonNumber: 2,
      type: 'lesson'
    }
  };

  const lesson = lessons[lessonId];

  if (!lesson) {
    return res.status(404).json({
      error: {
        code: 'LESSON_NOT_FOUND',
        message: 'Requested lesson not found.'
      }
    });
  }

  // Return lesson data
  res.status(200).json(lesson);
}