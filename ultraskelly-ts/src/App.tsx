import { VoiceAgentProvider } from './components/VoiceAgent/VoiceAgentProvider';
import { ConnectionButton } from './components/VoiceAgent/ConnectionButton';
import { StatusIndicator } from './components/VoiceAgent/StatusIndicator';
import { ConversationDisplay } from './components/VoiceAgent/ConversationDisplay';
import { AudioControls } from './components/VoiceAgent/AudioControls';
import { AudioStreamController } from './components/VoiceAgent/AudioStreamController';
import './style.css';
import {VoiceAgentConfig} from "./types/types.ts";

const BACKEND_URL = import.meta.env["VITE_BACKEND_URL"] || 'http://localhost:3001';

const AGENT_CONFIG: VoiceAgentConfig = {
  name: 'Assistant',
  instructions: 'You are a helpful and friendly assistant. Be concise and clear.',
  model: 'gpt-realtime',
};

// OPTION 1: Let TypeScript infer (RECOMMENDED - most modern)
export function App() {
  return (
    <VoiceAgentProvider backendUrl={BACKEND_URL}>
      <div className="app">
        <h1>🎤 Voice Agent with Audio Control</h1>
        <StatusIndicator />
        <ConnectionButton agentConfig={AGENT_CONFIG} />
        <AudioControls />
        <AudioStreamController />
        <ConversationDisplay />
      </div>
    </VoiceAgentProvider>
  );
}
