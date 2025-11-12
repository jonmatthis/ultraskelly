import { ConnectionButton } from './components/VoiceAgent/ConnectionButton';
import { StatusIndicator } from './components/VoiceAgent/StatusIndicator';
import { AudioControls } from './components/VoiceAgent/AudioControls';
import { AudioStreamController } from './components/VoiceAgent/AudioStreamController';
import { MicrophoneController } from './components/VoiceAgent/MicrophoneController';
import { VoiceAgentConfig } from './types/types';

// Import assistant-ui styles
import '@assistant-ui/styles/index.css';
import '@assistant-ui/styles/markdown.css';
import './style.css';
import {VoiceAgentProvider} from "./services/VoiceAgentProvider.tsx";
import {Thread} from "./components/VoiceAgent/assistant-ui-thread.tsx";

const BACKEND_URL = import.meta.env["VITE_BACKEND_URL"] || 'http://localhost:8174';

const AGENT_CONFIG: VoiceAgentConfig = {
    name: 'Assistant',
    instructions: 'You are a helpful and friendly assistant. Be concise and clear.',
    model: 'gpt-realtime',
};

export function App() {
    return (
        <VoiceAgentProvider backendUrl={BACKEND_URL}>
            <div className="app">
                <div className="app-header">
                    <h1>🎤 Voice Agent with assistant-ui</h1>
                    <StatusIndicator />
                </div>

                <div className="app-controls">
                    <ConnectionButton agentConfig={AGENT_CONFIG} />
                    <AudioControls />
                    <MicrophoneController />
                </div>

                {/* Audio visualization and controls */}
                <div className="audio-section">
                    <AudioStreamController />
                </div>

                {/* Chat interface using assistant-ui */}
                <div className="chat-section">
                    <Thread />
                </div>
            </div>
        </VoiceAgentProvider>
    );
}
