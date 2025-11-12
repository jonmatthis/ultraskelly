import { useVoiceAgentContext } from "../../services/VoiceAgentProvider.tsx";
import { useEffect, useRef } from "react";

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
                return 'Human';
            case 'assistant':
                return 'AI';
            default:
                return 'System';
        }
    };

    return (
        <div ref={containerRef} className="conversation">
            {conversation.messages.map((message) => (
                <div
                    key={message.id}
                    className={`message message-${message.role}`}
                >
                    <div className="message-header">
                        <strong className="message-speaker">
                            {getSpeakerLabel(message.role)}
                        </strong>
                        <span className="message-time">
              {formatTime(message.timestamp)}
            </span>
                    </div>
                    <div className="message-content">
                        {message.content}
                    </div>
                </div>
            ))}
        </div>
    );
}
