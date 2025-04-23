export interface Message {
  role: "user" | "assistant";
  content: string;
  // For assistant, split outputs for clearer UI rendering
  thinking?: string | null;
  solution?: string | null;
}

export interface ChatProps {
  messages: Message[];
  loading: boolean;
  onSendMessage: (message: string) => void;
  onNewChat: () => void;
}
