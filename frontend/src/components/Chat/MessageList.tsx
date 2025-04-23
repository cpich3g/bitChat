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
  
  // Helper function to parse content into think and solution parts
  const parseContent = (content: string): { thinkContent: string | null; solutionContent: string } => {
    const thinkMatch = content.match(/<think>(.*?)<\/think>\s*(.*)/s); // Use 's' flag for dotall
    
    if (thinkMatch && thinkMatch[1]) {
      // Found think block
      const thinkContent = thinkMatch[1].trim();
      const solutionContent = thinkMatch[2] ? thinkMatch[2].trim() : ''; // Content after think block
      return { thinkContent, solutionContent };
    } else {
      // No think block found, treat entire content as solution
      // Also remove potential leftover <im_start> tags just in case
      const solutionContent = content.replace(/<im_start>/g, '').trim();
      return { thinkContent: null, solutionContent };
    }
  };

  // Process only the solution part for markdown rendering (e.g., backticks)
  const processSolutionMarkdown = (solutionContent: string) => {
    // Ensure proper code block formatting with triple backticks
    return solutionContent
      // Replace consecutive backticks with properly spaced ones if needed
      .replace(/``` ?(\w+)/g, '```$1') // Handle optional space after ```
      // Fix any potential double spaces in language identifier - less likely now
      .replace(/```\s\s+(\w+)/g, '```$1'); 
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
            {msg.role === 'assistant' ? (() => {
              // Use explicit props if present (preferred: coming from API), else fallback
              const thinkContent = msg.thinking ?? null;
              const solutionContent = msg.solution ?? null;
              // If neither present, fall back to parseContent
              if (thinkContent !== null || solutionContent !== null) {
                return (
                  <>
                    {thinkContent && (
                      <details className="thinking-process">
                        <summary>
                          <span className="details-summary-default">
                            Show Thoughts
                          </span>
                          <span className="details-summary-open">
                            Hide Thoughts
                          </span>
                        </summary>
                        <pre className="think-content">{thinkContent}</pre>
                      </details>
                    )}
                    {solutionContent && (
                      <ReactMarkdown
                        components={{
                          code: ({ inline, className, children }: any) => {
                            const match = /language-(\w+)/.exec(className || '');
                            if (inline) {
                              return <code className={className}>{String(children).replace(/\n$/, '')}</code>;
                            }
                            return (
                              <pre>
                                <code className={match ? `language-${match[1]}` : className}>
                                  {String(children).replace(/\n$/, '')}
                                </code>
                              </pre>
                            );
                          },
                          table: ({ children, ...props }: any) => {
                            const { node, ...restProps } = props;
                            return (
                              <div className="table-container">
                                <table {...restProps}>{children}</table>
                              </div>
                            );
                          },
                        } satisfies Components}
                        rehypePlugins={[rehypeRaw]}
                      >
                        {processSolutionMarkdown(solutionContent)}
                      </ReactMarkdown>
                    )}
                  </>
                );
              } else {
                // fallback: parse out thinking/solution from content
                const fallback = parseContent(msg.content);
                return (
                  <>
                    {fallback.thinkContent && (
                      <details className="thinking-process">
                        <summary>Thinking Process</summary>
                        <pre className="think-content">{fallback.thinkContent}</pre>
                      </details>
                    )}
                    <ReactMarkdown
                      components={{
                        code: ({ inline, className, children }: any) => {
                          const match = /language-(\w+)/.exec(className || '');
                          if (inline) {
                            return <code className={className}>{String(children).replace(/\n$/, '')}</code>;
                          }
                          return (
                            <pre>
                              <code className={match ? `language-${match[1]}` : className}>
                                {String(children).replace(/\n$/, '')}
                              </code>
                            </pre>
                          );
                        },
                        table: ({ children, ...props }: any) => {
                          const { node, ...restProps } = props;
                          return (
                            <div className="table-container">
                              <table {...restProps}>{children}</table>
                            </div>
                          );
                        },
                      } satisfies Components}
                      rehypePlugins={[rehypeRaw]}
                    >
                      {processSolutionMarkdown(fallback.solutionContent)}
                    </ReactMarkdown>
                  </>
                );
              }
            })() : (
              msg.content // User messages are displayed directly
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
