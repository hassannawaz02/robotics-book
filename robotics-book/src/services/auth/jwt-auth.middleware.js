/**
 * JWT Authentication Middleware for Isaac Robot Brain Module
 * Provides JWT-based authentication for lesson access
 */

const jwt = require('jsonwebtoken');

// JWT verification middleware
const jwtAuth = (req, res, next) => {
  // Get token from header
  const token = req.header('Authorization')?.replace('Bearer ', '');

  // Check if token exists
  if (!token) {
    return res.status(401).json({
      error: {
        code: 'AUTH_REQUIRED',
        message: 'Access denied. No token provided.'
      }
    });
  }

  try {
    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'default_secret');
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({
      error: {
        code: 'INVALID_TOKEN',
        message: 'Invalid or expired token.'
      }
    });
  }
};

// Middleware to check if user has access to specific lesson
const lessonAccess = (req, res, next) => {
  // In a real implementation, this would check if the user has access
  // to the specific lesson based on their role or subscription
  const lessonId = req.params.lessonId || req.body.lessonId;

  // For now, we'll just check if user is authenticated
  if (!req.user) {
    return res.status(401).json({
      error: {
        code: 'LESSON_ACCESS_DENIED',
        message: 'Access denied. User not authenticated.'
      }
    });
  }

  next();
};

module.exports = {
  jwtAuth,
  lessonAccess
};