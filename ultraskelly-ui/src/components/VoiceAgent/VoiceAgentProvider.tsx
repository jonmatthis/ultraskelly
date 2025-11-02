import {createContext, type ReactNode, useContext, useEffect} from "react";
import {useVoiceAgent, type UseVoiceAgentResult} from '../../hooks/useVoiceAgent';
import {useConversationHistory, type UseConversationHistoryResult} from '../../hooks/useConversationHistory';
import {useBackendAPI, UseBackendAPIResult} from "../../hooks/useBackendApi.ts";

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

  useEffect(() => {
    voiceAgent.onHistoryUpdate((history) => {
      conversation.updateFromHistory(history);
    });
  }, [voiceAgent, conversation]);

  return (
    <VoiceAgentContext.Provider value={{ voiceAgent, conversation, backend }}>
      {children}
    </VoiceAgentContext.Provider>
  );
}
