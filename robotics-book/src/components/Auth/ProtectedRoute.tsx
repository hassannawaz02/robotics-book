/**
 * ProtectedRoute Component
 * This component protects routes that require authentication
 * In a real implementation, this would integrate with the auth service
 */

import React, { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { isAuthenticated, getCurrentUser } from '../../services/auth';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: 'learner' | 'admin';
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole = 'learner'
}) => {
  const location = useLocation();
  const [isAuthorized, setIsAuthorized] = useState<boolean | null>(null);

  useEffect(() => {
    // Check authentication status
    const checkAuth = async () => {
      const authenticated = isAuthenticated();
      if (!authenticated) {
        setIsAuthorized(false);
        return;
      }

      // If authenticated, check role permissions
      const user = getCurrentUser();
      if (!user) {
        setIsAuthorized(false);
        return;
      }

      // Check if user has required role
      if (requiredRole === 'admin' && user.role !== 'admin') {
        setIsAuthorized(false);
        return;
      }

      setIsAuthorized(true);
    };

    checkAuth();
  }, [requiredRole]);

  // While checking auth status, show loading state
  if (isAuthorized === null) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <p>Checking authentication...</p>
          </div>
        </div>
      </div>
    );
  }

  // If not authorized, redirect to login
  if (!isAuthorized) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // If authorized, render the protected content
  return <>{children}</>;
};

export default ProtectedRoute;