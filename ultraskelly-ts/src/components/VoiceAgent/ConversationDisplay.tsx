import {useVoiceAgentContext} from "./VoiceAgentProvider.tsx";
import {useEffect, useRef} from "react";

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function ConversationDisplay(): JSX.Element {
  const { conversation } = useVoiceAgentContext();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [conversation.messages]);

  const getSpeakerLabel = (role: string): string => {
    switch (role) {
      case 'user':
        return 'You';
      case 'assistant':
        return 'Assistant';
      default:
        return 'System';
    }
  };

  return (
    <div ref={containerRef} className="conversation">
      {conversation.messages.map((message) => (
        <div key={message.id} className="message">
          <strong>
            {getSpeakerLabel(message.role)} ({formatTime(message.timestamp)}):
          </strong>{' '}
          {message.content}
        </div>
      ))}
    </div>
  );
}
