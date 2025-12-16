import React from 'react';
import AIChat from '../components/AIChat/AIChat.jsx';

const Root = ({ children }) => {
  return (
    <>
      {children}
      <AIChat lessonTitle="General" moduleId="All Modules" />
    </>
  );
};

export default Root;