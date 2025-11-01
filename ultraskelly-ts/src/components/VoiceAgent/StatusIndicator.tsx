import {useVoiceAgentContext} from "./VoiceAgentProvider.tsx";

export function StatusIndicator() {
  const { voiceAgent, backend } = useVoiceAgentContext();

  const getStatusText = (): string => {
    if (!backend.isHealthy) {
      return '❌ Backend not running. Please start the server!';
    }
    
    if (backend.isChecking) {
      return 'Checking backend...';
    }

    switch (voiceAgent.status) {
      case 'connected':
        return '🟢 Connected! Start speaking...';
      case 'connecting':
        return 'Connecting to OpenAI...';
      case 'disconnecting':
        return 'Disconnecting...';
      case 'error':
        return `❌ Error: ${voiceAgent.error || 'Unknown error'}`;
      default:
        return '✅ Ready to connect';
    }
  };

  const getClassName = (): string => {
    if (voiceAgent.status === 'connected') {
      return 'status connected';
    }
    if (voiceAgent.status === 'error' || !backend.isHealthy) {
      return 'status error';
    }
    return 'status';
  };

  return (
    <div className={getClassName()}>
      {getStatusText()}
    </div>
  );
}