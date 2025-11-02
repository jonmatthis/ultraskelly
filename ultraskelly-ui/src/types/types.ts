import type { RealtimeAgent, RealtimeSession } from '@openai/agents-realtime';

// Connection Types
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'disconnecting' | 'error';

export interface VoiceAgentConfig {
    name: string;
    instructions: string;
    model?: string;
}

export interface VoiceAgentState {
    status: ConnectionStatus;
    isConnected: boolean;
    session: RealtimeSession | null;
    agent: RealtimeAgent | null;
    error: string | null;
}

// Conversation Types
export interface ConversationMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    type: 'message' | 'tool_call' | 'tool_result' | 'handoff';
    metadata?: Record<string, unknown>;
}

export interface ConversationHistoryItem {
    type: string;
    role?: string;
    content?: Array<{ transcript?: string; text?: string }>;
    formatted?: { transcript?: string };
    tool?: { name: string };
    output?: unknown;
    [key: string]: unknown;
}

// API Types
export interface EphemeralKeyResponse {
    value: string;
    expires_at: number;
}

export interface BackendConfig {
    baseUrl: string;
}

export interface HealthCheckResponse {
    status: 'ok' | 'error';
    message?: string;
}

export interface SendMessageRequest {
    conversationId: string;
    message: string;
    metadata?: Record<string, unknown>;
}

export interface SendMessageResponse {
    success: boolean;
    messageId?: string;
    error?: string;
}

// Audio Event Types - Properly typed!
export interface AudioEvent {
    data: ArrayBuffer;
    itemId: string;
}

export interface AudioStartEvent {
    itemId: string;
}

export interface AudioStoppedEvent {
    itemId: string;
}

export interface ToolStartEvent {
    tool: { name: string };
}

export interface ToolEndEvent {
    tool: { name: string };
    output: unknown;
}
