import { useState, useRef, useEffect } from "react";
import "./App.css";
import { Welcome, MessageList, ChatInput, Message } from "./components/Chat";
import Paper from '@mui/material/Paper';
import Grow from '@mui/material/Grow';

const API_URL = "/chat";
const STREAM_API_URL = "/chat/stream";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [chatStarted, setChatStarted] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (messages.length > 0 && !chatStarted) {
      setChatStarted(true);
    }
    
    // Apply a small delay to ensure DOM updates before scrolling
    const scrollTimeout = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
    
    return () => clearTimeout(scrollTimeout);
  }, [messages, chatStarted, loading, streamingContent]);

  const handleSendMessage = async (inputText: string) => {
    if (!inputText.trim()) return;
    
    const userMessage: Message = { role: "user", content: inputText };
    
    // Immediately update messages with user message
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setStreamingContent(""); // Reset streaming content
    
    // Create a temporary assistant message with empty content that will be streamed
    const streamingMessage: Message = { role: "assistant", content: "" };
    
    // Add the empty assistant message to messages
    setMessages(prev => [...prev, streamingMessage]);
    
    try {
      // Use fetch with proper streaming approach for plain text
      const response = await fetch(STREAM_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: [...messages, userMessage],
          max_new_tokens: 500
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      // Get the response as a readable stream
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get stream reader");
      }
      
      // Create a decoder to convert the binary stream to text
      const decoder = new TextDecoder();
      let assistantResponse = "";
      
      // Read the stream chunk by chunk
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // Decode the chunk as plain text
        const chunk = decoder.decode(value, { stream: true });
        
        // Add token to the response
        assistantResponse += chunk;
        setStreamingContent(assistantResponse);
        
        // Update the last message (which is the assistant's response)
        setMessages(prev => {
          const updatedMessages = [...prev];
          updatedMessages[updatedMessages.length - 1] = {
            role: "assistant", 
            content: assistantResponse
          };
          return updatedMessages;
        });
      }
      
    } catch (err) {
      console.error("Error with streaming:", err);
      // Fallback to non-streaming API if streaming fails
      try {
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: [...messages, userMessage] }),
        });
        
        const data = await res.json();
        
        setMessages(prev => {
          const updatedMessages = [...prev];
          // data: { full_output, solution }
          let thinking = null;
          if (data.full_output && typeof data.full_output === "string") {
            const thinkMatch = data.full_output.match(/<think>(.*?)<\/think>/s);
            if (thinkMatch && thinkMatch[1]) {
              thinking = thinkMatch[1].trim();
            }
          }
          updatedMessages[updatedMessages.length - 1] = {
            role: "assistant",
            content: data.full_output ?? data.solution ?? "",
            thinking,
            solution: typeof data.solution === "string" ? data.solution : (data.full_output ?? "")
          };
          return updatedMessages;
        });
      } catch (fallbackErr) {
        setMessages(prev => {
          const updatedMessages = [...prev];
          updatedMessages[updatedMessages.length - 1] = {
            role: "assistant", 
            content: "Error: Could not get response."
          };
          return updatedMessages;
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setChatStarted(false);
    setStreamingContent("");
  };

  return (
    <div className={`app-container${chatStarted ? " chat-started" : " chat-initial"}`}>
      <div className="chat-container">
        {!chatStarted ? (
          <Grow in timeout={500}>
            <Paper elevation={6} className="welcome-card">
              <div className="center-stack">
                <Welcome />
                <div className="centered-input-section">
                  <ChatInput
                    onSendMessage={handleSendMessage}
                    onNewChat={handleNewChat}
                    loading={loading}
                    showNewChatButton={false}
                  />
                </div>
              </div>
            </Paper>
          </Grow>
        ) : (
          <>
            <MessageList 
              messages={messages} 
              loading={loading}
              streamingContent={streamingContent}
            />
            <div className="docked-input-section" ref={messagesEndRef}>
              <ChatInput
                onSendMessage={handleSendMessage}
                onNewChat={handleNewChat}
                loading={loading}
                showNewChatButton={messages.length > 0}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
