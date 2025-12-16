import React, { useState } from 'react';
import ChatInterface from '../Chat/ChatInterface';

const FloatingChatButton: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {isOpen && (
        <div className="chat-overlay" onClick={toggleChat} style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          zIndex: 9998,
        }} />
      )}

      {isOpen && (
        <div className="chat-window" style={{
          position: 'fixed',
          bottom: '80px',
          right: '20px',
          zIndex: 9999,
          width: '400px',
          height: '500px',
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        }}>
          <ChatInterface />
        </div>
      )}

      <button
        className="chat-button"
        onClick={toggleChat}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: 9999,
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          backgroundColor: '#4a90e2',
          color: 'white',
          border: 'none',
          cursor: 'pointer',
          fontSize: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        }}
        aria-label="Open chat"
      >
        💬
      </button>
    </>
  );
};

export default FloatingChatButton;