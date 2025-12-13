import React from 'react';

/**
 * AI Chat Placeholder Component
 * This component serves as a placeholder for the AI chat interface
 * Backend integration will be implemented after all modules are built
 */
const AIChatPlaceholder = () => {
  return (
    <div className="ai-chat-placeholder">
      <div className="chat-header">
        <h3>AI Assistant</h3>
        <p>Ask questions about Digital Twin concepts</p>
      </div>
      <div className="chat-interface">
        <div className="chat-messages">
          <div className="message bot-message">
            <p>Hello! I'm your AI assistant for the Digital Twin module.</p>
            <p>I'm currently a placeholder and will be fully functional after RAG backend implementation.</p>
          </div>
        </div>
        <div className="chat-input-section">
          <input
            type="text"
            placeholder="Ask a question about Gazebo, Unity, or sensor simulation..."
            disabled
            className="chat-input-disabled"
          />
          <button disabled className="send-button-disabled">
            Send
          </button>
        </div>
      </div>
      <div className="chat-info">
        <p><small>Note: Backend integration is deferred to future RAG implementation</small></p>
      </div>
    </div>
  );
};

export default AIChatPlaceholder;