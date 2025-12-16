/**
 * AI Chat Component
 * UI-only component for AI learning assistant (backend deferred for future RAG implementation)
 */

import React, { useState } from 'react';
import './AIChat.css';

const AIChat = ({ lessonTitle, moduleId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! I\'m your AI learning assistant for the Isaac Robot Brain module. How can I help you with your robotics studies today?', sender: 'ai' }
  ]);
  const [inputValue, setInputValue] = useState('');

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSend = () => {
    if (inputValue.trim()) {
      // Add user message
      const userMessage = {
        id: messages.length + 1,
        text: inputValue,
        sender: 'user'
      };

      setMessages(prev => [...prev, userMessage]);
      setInputValue('');

      // Simulate AI response (in real implementation, this would call the backend)
      setTimeout(() => {
        const aiResponse = {
          id: messages.length + 2,
          text: 'Thank you for your question. In the full implementation, I would provide a detailed response based on the Isaac Robot Brain course materials using RAG (Retrieval Augmented Generation).',
          sender: 'ai'
        };
        setMessages(prev => [...prev, aiResponse]);
      }, 1000);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className={`ai-chat-container ${isOpen ? 'open' : 'closed'}`}>
      <button className="ai-chat-toggle" onClick={toggleChat}>
        {isOpen ? '×' : '🤖'}
      </button>

      {isOpen && (
        <div className="ai-chat-window">
          <div className="ai-chat-header">
            <h3>AI Learning Assistant</h3>
            <p>Module: {moduleId || 'Isaac Robot Brain'}, Lesson: {lessonTitle || 'Current'}</p>
          </div>

          <div className="ai-chat-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`ai-chat-message ${message.sender === 'user' ? 'user-message' : 'ai-message'}`}
              >
                <div className="message-content">
                  {message.text}
                </div>
              </div>
            ))}
          </div>

          <div className="ai-chat-input">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about this lesson..."
            />
            <button onClick={handleSend}>Send</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIChat;