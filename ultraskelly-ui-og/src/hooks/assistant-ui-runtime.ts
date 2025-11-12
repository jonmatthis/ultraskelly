import { useExternalStoreRuntime, type ThreadMessageLike, type AppendMessage } from '@assistant-ui/react';
import { useCallback, useEffect, useRef } from 'react';
import type { ConversationHistoryItem } from '../types/types';
import type { RealtimeSession } from '@openai/agents-realtime';

interface UseRealtimeAssistantRuntimeProps {
    session: RealtimeSession | null;
    isConnected: boolean;
    history: ConversationHistoryItem[];
    onSendMessage: (message: string) => Promise<void>;
    onInterrupt: () => Promise<void>;
}

function convertHistoryToMessages(history: ConversationHistoryItem[]): ThreadMessageLike[] {
    const messages: ThreadMessageLike[] = [];

    for (const item of history) {
        if (item.type === 'message' && item.role) {
            const content = item.formatted?.transcript ||
                item.content?.[0]?.transcript ||
                item.content?.[0]?.text;

            if (content) {
                // Create message with minimal required fields
                // Don't include metadata or status unless they have actual values
                const message: ThreadMessageLike = {
                    id: `msg-${item.role}-${messages.length}`,
                    role: item.role === 'user' ? 'user' : 'assistant',
                    content: [{ type: 'text', text: content }],
                    createdAt: new Date(),
                };
                messages.push(message);
            }
        } else if (item.type === 'tool_call' && item.tool?.name) {
            // Include tool calls as assistant messages
            const message: ThreadMessageLike = {
                id: `tool-call-${messages.length}`,
                role: 'assistant',
                content: [{
                    type: 'tool-call' as const,
                    toolCallId: `tool-${messages.length}`,
                    toolName: item.tool.name,
                    args: {},
                    result: item.output,
                }],
                createdAt: new Date(),
            };
            messages.push(message);
        }
    }

    return messages;
}

export function useRealtimeAssistantRuntime({
                                                session,
                                                isConnected,
                                                history,
                                                onSendMessage,
                                                onInterrupt,
                                            }: UseRealtimeAssistantRuntimeProps) {
    const isRunningRef = useRef(false);
    const messagesRef = useRef<ThreadMessageLike[]>([]);

    // Convert OpenAI Realtime history to assistant-ui message format
    const messages = convertHistoryToMessages(history);
    messagesRef.current = messages;

    // Handle new messages from the user
    const onNew = useCallback(async (message: AppendMessage): Promise<void> => {
        if (!session || !isConnected) {
            console.error('Not connected to Realtime session');
            return;
        }

        // Extract text content from the message
        const textContent = message.content
            .filter(part => part.type === 'text')
            .map(part => ('text' in part ? part.text : ''))
            .join(' ')
            .trim();

        if (!textContent) {
            console.error('No text content in message');
            return;
        }

        // Mark as running
        isRunningRef.current = true;

        try {
            // Send message through OpenAI Realtime session
            await onSendMessage(textContent);
        } catch (error) {
            console.error('Failed to send message:', error);
            // Mark as not running on error
            isRunningRef.current = false;
        }
    }, [session, isConnected, onSendMessage]);

    // Handle cancellation
    const onCancel = useCallback(async (): Promise<void> => {
        if (!session || !isConnected) return;

        try {
            await onInterrupt();
            isRunningRef.current = false;
        } catch (error) {
            console.error('Failed to interrupt:', error);
        }
    }, [session, isConnected, onInterrupt]);

    // Track when audio/responses are being generated
    useEffect(() => {
        if (!session) return;

        const handleResponseStart = (): void => {
            isRunningRef.current = true;
        };

        const handleResponseEnd = (): void => {
            isRunningRef.current = false;
        };

        // Listen to audio events to track when assistant is speaking
        session.on('audio_start', handleResponseStart);
        session.on('audio_stopped', handleResponseEnd);
        session.on('audio_interrupted', handleResponseEnd);

        return () => {
            session.off('audio_start', handleResponseStart);
            session.off('audio_stopped', handleResponseEnd);
            session.off('audio_interrupted', handleResponseEnd);
        };
    }, [session]);

    // Create the runtime with properly structured state
    const runtime = useExternalStoreRuntime({
        messages: messagesRef.current,
        isRunning: isRunningRef.current,
        onNew,
        onCancel,
        // Use convertMessage to ensure proper format
        convertMessage: (message: ThreadMessageLike): ThreadMessageLike => {
            // Ensure the message conforms to the expected structure
            // Only include defined fields to work with exactOptionalPropertyTypes
            const result: ThreadMessageLike = {
                id: message.id,
                role: message.role,
                content: message.content,
                createdAt: message.createdAt || new Date(),
            };

            // Only add optional fields if they have actual values
            if (message.metadata && Object.keys(message.metadata).length > 0) {
                result.metadata = message.metadata;
            }
            if (message.status) {
                result.status = message.status;
            }

            return result;
        },
    });

    return runtime;
}
