/**
 * Placeholder Authentication Service for Robotics Textbook
 * This service handles JWT-based authentication for the frontend
 * In a real implementation, this would connect to the backend API
 */

// Define user types
export interface User {
  id: string;
  email: string;
  role: 'learner' | 'admin';
  firstName?: string;
  lastName?: string;
}

// Define authentication state
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
}

// Initialize auth state
let authState: AuthState = {
  user: null,
  token: null,
  isAuthenticated: false
};

// Placeholder for backend API URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Login function - placeholder implementation
 */
export const login = async (email: string, password: string): Promise<{ user: User; token: string } | null> => {
  // In a real implementation, this would make an API call to the backend
  // For now, we'll return a mock user
  console.log('Login attempt for:', email);

  // Mock response - in real implementation, this comes from backend
  const mockUser: User = {
    id: 'mock-user-id',
    email,
    role: 'learner',
    firstName: 'Test',
    lastName: 'User'
  };

  const mockToken = 'mock-jwt-token';

  // Update auth state
  authState = {
    user: mockUser,
    token: mockToken,
    isAuthenticated: true
  };

  // Store token in localStorage (or secure cookie in production)
  localStorage.setItem('authToken', mockToken);

  return { user: mockUser, token: mockToken };
};

/**
 * Logout function
 */
export const logout = (): void => {
  // Clear auth state
  authState = {
    user: null,
    token: null,
    isAuthenticated: false
  };

  // Remove token from storage
  localStorage.removeItem('authToken');
};

/**
 * Check if user is authenticated
 */
export const isAuthenticated = (): boolean => {
  // Check if we have a token in storage
  const token = localStorage.getItem('authToken');
  if (!token) {
    return false;
  }

  // In a real implementation, we would verify the token
  // For now, assume it's valid if it exists
  authState.isAuthenticated = true;
  return true;
};

/**
 * Get current user
 */
export const getCurrentUser = (): User | null => {
  if (!isAuthenticated()) {
    return null;
  }

  // In a real implementation, we would decode the token to get user info
  // For now, return the stored user or null
  return authState.user;
};

/**
 * Get auth token
 */
export const getAuthToken = (): string | null => {
  // Check if token exists in storage
  const token = localStorage.getItem('authToken');
  if (token) {
    authState.token = token;
    return token;
  }
  return null;
};

/**
 * Refresh token (placeholder)
 */
export const refreshToken = async (): Promise<string | null> => {
  // In a real implementation, this would refresh the JWT token
  // For now, return the existing token or null
  const currentToken = getAuthToken();
  return currentToken;
};

/**
 * Verify module access for current user
 */
export const canAccessModule = (moduleId: string, userRole: 'learner' | 'admin'): boolean => {
  // For this implementation, learners can access all modules
  // Admins have additional permissions
  return userRole === 'learner' || userRole === 'admin';
};

// Initialize auth state on module load
if (isAuthenticated()) {
  // If user is authenticated, get their info
  // In a real implementation, this would be done by decoding the JWT or making an API call
}