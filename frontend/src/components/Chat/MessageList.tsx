import React, { useRef, useEffect, useState } from 'react';
import { Message } from './types';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';

interface MessageListProps {
  messages: Message[];
  loading: boolean;
  streamingContent?: string;
}

const MessageList: React.FC<MessageListProps> = ({ messages, loading, streamingContent = "" }) => {
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  
  // Scroll to bottom on new messages if user was already at bottom
  useEffect(() => {
    if (messagesContainerRef.current && shouldAutoScroll) {
      const container = messagesContainerRef.current;
      
      // Add small delay to ensure DOM updates before scrolling
      setTimeout(() => {
        container.scrollTop = container.scrollHeight;
      }, 50);
    }
  }, [messages, loading, shouldAutoScroll, streamingContent]);
  
  // Handle scroll events to determine if auto-scroll should be enabled
  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const container = messagesContainerRef.current;
      const isAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150;
      setShouldAutoScroll(isAtBottom);
    }
  };
  
  // Process markdown content to fix triple backticks
  const processMarkdown = (content: string) => {
    // Ensure proper code block formatting with triple backticks
    return content
      // Replace consecutive backticks with properly spaced ones if needed
      .replace(/```(\w+)/g, '``` $1')
      // Fix any potential double spaces in language identifier
      .replace(/```\s\s+(\w+)/g, '``` $1');
  };
  
  return (
    <div 
      className="chat-messages" 
      ref={messagesContainerRef}
      onScroll={handleScroll}
    >
      {messages.map((msg, idx) => (
        <div key={idx} className={`message ${msg.role}`}>
          <div className="message-content">
            {msg.role === 'assistant' ? (
              <ReactMarkdown
                components={{
                  // Custom renderer for thinking tokens
                  think({ node, children, ...props }) {
                    return <span className="thinking">{children}</span>;
                  },
                  // Custom styling for code blocks
                  code({ node, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    // Remove the 'inline' reference and use props.inline if needed
                    if ((props as any).inline) {
                      return (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                    return (
                      <pre>
                        <code className={match ? `language-${match[1]}` : ''} {...props}>
                          {children}
                        </code>
                      </pre>
                    );
                  },
                  // Better table rendering
                  table({ node, children, ...props }) {
                    return (
                      <div className="table-container">
                        <table {...props}>{children}</table>
                      </div>
                    );
                  }
                }}
                rehypePlugins={[rehypeRaw]}
              >
                {processMarkdown(msg.content)}
              </ReactMarkdown>
            ) : (
              msg.content
            )}
          </div>
        </div>
      ))}
      {loading && !messages[messages.length - 1]?.content && (
        <div className="message assistant">
          <div className="message-content">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      )}
      {/* Extra space to ensure content isn't hidden behind input */}
      <div style={{ height: "100px" }} />
    </div>
  );
};

export default MessageList;