import {useVoiceAgentContext} from "../../services/VoiceAgentProvider.tsx";

export function AudioControls() {
  const { voiceAgent } = useVoiceAgentContext();

  const handleInterrupt = async (): Promise<void> => {
    try {
      await voiceAgent.interrupt();
    } catch (error) {
      console.error('Failed to interrupt:', error);
    }
  };

  if (!voiceAgent.isConnected) {
    return null;
  }

  return (
    <div className="audio-controls">
      <button
        onClick={handleInterrupt}
        className="interrupt-btn"
      >
        ⏸️ Interrupt Agent
      </button>
    </div>
  )
}
