import { useState, useCallback } from 'react';
import { ConversationHistoryItem, ConversationMessage } from "../types/types.ts";

export interface UseConversationHistoryResult {
    messages: ConversationMessage[];
    addMessage: (message: ConversationMessage) => void;
    clearMessages: () => void;
    updateFromHistory: (history: ConversationHistoryItem[]) => void;
}

function parseHistoryItem(item: ConversationHistoryItem, index: number): ConversationMessage | null {
    if (item.type === 'message') {
        const transcript = item.formatted?.transcript || item.content?.[0]?.transcript;

        if (transcript && item.role) {
            return {
                id: `msg-${item.role}-${index}`,
                role: item.role === 'user' ? 'user' : 'assistant',
                content: transcript,
                timestamp: new Date(),
                type: 'message',
            };
        }
    } else if (item.type === 'tool_call' && item.tool?.name) {
        return {
            id: `tool-${index}`,
            role: 'assistant',
            content: `Calling tool: ${item.tool.name}`,
            timestamp: new Date(),
            type: 'tool_call',
            metadata: { toolName: item.tool.name },
        };
    } else if (item.type === 'tool_result') {
        return {
            id: `tool-result-${index}`,
            role: 'system',
            content: `Tool result: ${JSON.stringify(item.output)}`,
            timestamp: new Date(),
            type: 'tool_result',
        };
    }

    return null;
}

export function useConversationHistory(): UseConversationHistoryResult {
    const [messages, setMessages] = useState<ConversationMessage[]>([]);

    const addMessage = useCallback((message: ConversationMessage): void => {
        setMessages(prev => [...prev, message]);
    }, []);

    const clearMessages = useCallback((): void => {
        setMessages([]);
    }, []);

    const updateFromHistory = useCallback((history: ConversationHistoryItem[]): void => {
        // Simply parse and set the full history from the API
        // The Realtime API maintains its own conversation state
        const parsedMessages = history
            .map((item, index) => parseHistoryItem(item, index))
            .filter((msg): msg is ConversationMessage => msg !== null);

        setMessages(parsedMessages);
    }, []);

    return {
        messages,
        addMessage,
        clearMessages,
        updateFromHistory,
    };
}
