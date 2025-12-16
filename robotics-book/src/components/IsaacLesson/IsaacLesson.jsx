/**
 * IsaacLesson Component
 * Renders Isaac-specific lesson content with proper formatting
 */

import React from 'react';
import PropTypes from 'prop-types';
import ChatInterface from '../Chat/ChatInterface';

const IsaacLesson = ({
  title,
  content,
  codeExamples = [],
  diagrams = [],
  lessonNumber,
  moduleId = 'module-3'
}) => {
  return (
    <div className="isaac-lesson-container">
      <header className="isaac-lesson-header">
        <h1>{title}</h1>
        <div className="lesson-meta">
          <span className="module-id">Module: {moduleId}</span>
          <span className="lesson-number">Lesson: {lessonNumber}</span>
        </div>
      </header>

      <div className="isaac-lesson-content">
        <div
          className="lesson-text-content"
          dangerouslySetInnerHTML={{ __html: content }}
        />
      </div>

      {diagrams.length > 0 && (
        <div className="isaac-lesson-diagrams">
          <h3>Architecture Diagrams</h3>
          <div className="diagrams-grid">
            {diagrams.map((diagram, index) => (
              <div key={index} className="diagram-container">
                <img
                  src={diagram.src}
                  alt={diagram.alt || `Diagram ${index + 1}`}
                  className="diagram-image"
                />
                {diagram.caption && (
                  <p className="diagram-caption">{diagram.caption}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {codeExamples.length > 0 && (
        <div className="isaac-lesson-code-examples">
          <h3>Code Examples</h3>
          <div className="code-examples-container">
            {codeExamples.map((example, index) => (
              <div key={index} className="code-example">
                <h4>{example.title || `Example ${index + 1}`}</h4>
                <pre className="code-block">
                  <code>{example.code}</code>
                </pre>
                {example.description && (
                  <p className="code-description">{example.description}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Chat Component */}
      <div style={{ marginTop: '2rem' }}>
        <ChatInterface />
      </div>
    </div>
  );
};

IsaacLesson.propTypes = {
  title: PropTypes.string.isRequired,
  content: PropTypes.string.isRequired,
  codeExamples: PropTypes.array,
  diagrams: PropTypes.array,
  lessonNumber: PropTypes.number,
  moduleId: PropTypes.string
};

export default IsaacLesson;