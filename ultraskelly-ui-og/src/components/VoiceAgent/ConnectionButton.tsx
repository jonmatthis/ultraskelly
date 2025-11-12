import {useVoiceAgentContext} from "../../services/VoiceAgentProvider.tsx";
import {useState} from "react";
import {VoiceAgentConfig} from "../../types/types.ts";

interface ConnectionButtonProps {
  agentConfig: VoiceAgentConfig;
}

export function ConnectionButton({ agentConfig }: ConnectionButtonProps) {
  const { voiceAgent, backend } = useVoiceAgentContext();
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async (): Promise<void> => {
    if (voiceAgent.isConnected) {
      await voiceAgent.disconnect();
      return;
    }

    try {
      setIsLoading(true);
      const ephemeralKey = await backend.getEphemeralKey();
      await voiceAgent.connect(ephemeralKey, agentConfig);
    } catch (error) {
      console.error('Connection error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getButtonText = (): string => {
    if (isLoading) {
      return voiceAgent.status === 'connecting' ? '🔌 Connecting...' : '🔑 Getting key...';
    }
    return voiceAgent.isConnected ? 'Disconnect' : 'Connect to Agent';
  };

  return (
    <button
      onClick={handleClick}
      disabled={isLoading || !backend.isHealthy}
      className="connect-btn"
    >
      {getButtonText()}
    </button>
  );
}
