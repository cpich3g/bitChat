import React, { useRef, useEffect, useState } from 'react';
import { Message } from './types';
import ReactMarkdown, { Components } from 'react-markdown';
import rehypeRaw from 'rehype-raw';
// Removed incorrect imports

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
  
  // Process markdown content to fix triple backticks and remove system tags
  const processMarkdown = (content: string) => {
    // Remove <think>...</think> blocks
    let processedContent = content.replace(/<think>.*?<\/think>/gs, '');
    // Remove <im_start> tags
    processedContent = processedContent.replace(/<im_start>/g, '');
    
    // Ensure proper code block formatting with triple backticks
    return processedContent
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
              <>
                {/* Removed the <pre> block that was displaying raw content */}
                <ReactMarkdown
                  components={
                    {
                    // Custom styling for code blocks - Simplified props type
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    code: ({ inline, className, children }: any) => { // Removed unused ...props
                      const match = /language-(\w+)/.exec(className || '');
                      // Removed unused variables: const { node, ...restProps } = props;
                      if (inline) {
                        // Pass only className for inline code
                        return <code className={className}>{String(children).replace(/\n$/, '')}</code>;
                      }
                      // Pass only className to the inner code block
                      return (
                        <pre> {/* Avoid spreading potentially incompatible props */}
                          <code className={match ? `language-${match[1]}` : className}>
                            {String(children).replace(/\n$/, '')}
                          </code>
                        </pre>
                      );
                    },
                    // Better table rendering - Simplified props type
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    table: ({ children, ...props }: any) => {
                      // Filter out potentially problematic props like 'node' - keep restProps here
                      const { node, ...restProps } = props; // Removed unused 'node' from destructuring
                      return (
                        <div className="table-container">
                          <table {...restProps}>{children}</table>
                        </div>
                      );
                    }
                  } satisfies Components // Use 'satisfies' for better type checking without casting
                }
                  rehypePlugins={[rehypeRaw]}
                >
                  {processMarkdown(msg.content)}
                </ReactMarkdown>
              </>
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
