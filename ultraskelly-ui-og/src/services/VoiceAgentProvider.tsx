import { createContext, type ReactNode, useContext, useEffect } from 'react';
import { AssistantRuntimeProvider } from '@assistant-ui/react';
import {useConversationHistory, UseConversationHistoryResult} from "../hooks/useConversationHistory.ts";
import {useVoiceAgent, UseVoiceAgentResult} from "../hooks/useVoiceAgent.ts";
import {useBackendAPI, UseBackendAPIResult} from "../hooks/useBackendApi.ts";
import {useRealtimeAssistantRuntime} from "../hooks/assistant-ui-runtime.ts";

export function useVoiceAgentContext(): VoiceAgentContextValue {
    const context = useContext(VoiceAgentContext);
    if (!context) {
        throw new Error('useVoiceAgentContext must be used within VoiceAgentProvider');
    }
    return context;
}

interface VoiceAgentContextValue {
    voiceAgent: UseVoiceAgentResult;
    conversation: UseConversationHistoryResult;
    backend: UseBackendAPIResult;
}

const VoiceAgentContext = createContext<VoiceAgentContextValue | null>(null);

interface VoiceAgentProviderProps {
    children: ReactNode;
    backendUrl: string;
}

export function VoiceAgentProvider({ children, backendUrl }: VoiceAgentProviderProps): JSX.Element {
    const voiceAgent = useVoiceAgent();
    const conversation = useConversationHistory();
    const backend = useBackendAPI(backendUrl);

    // Update conversation history when voice agent updates
    useEffect(() => {
        voiceAgent.onHistoryUpdate((history) => {
            conversation.updateFromHistory(history);
        });
    }, [voiceAgent, conversation]);

    // Create the assistant-ui runtime that integrates with OpenAI Realtime
    const assistantRuntime = useRealtimeAssistantRuntime({
        session: voiceAgent.session,
        isConnected: voiceAgent.isConnected,
        history: conversation.messages.map(msg => ({
            type: 'message',
            role: msg.role,
            content: [{ transcript: msg.content }],
            formatted: { transcript: msg.content },
        })),
        onSendMessage: voiceAgent.sendMessage,
        onInterrupt: voiceAgent.interrupt,
    });

    // Provide both the voice agent context and assistant-ui runtime
    return (
        <VoiceAgentContext.Provider value={{ voiceAgent, conversation, backend }}>
            <AssistantRuntimeProvider runtime={assistantRuntime}>
                {children}
            </AssistantRuntimeProvider>
        </VoiceAgentContext.Provider>
    );
}
