

import {OpenAIRealtimeWebSocket, RealtimeAgent, RealtimeSession} from '@openai/agents-realtime';
import {ConversationHistoryItem, VoiceAgentConfig} from "../types/types.ts";

export interface RealtimeServiceCallbacks {
  onConnected: () => void;
  onDisconnected: () => void;
  onError: (error: Error) => void;
  onHistoryUpdated: (history: ConversationHistoryItem[]) => void;
  onToolStart?: (toolName: string) => void;
  onToolEnd?: (toolName: string, output: unknown) => void;
  onAudioInterrupted?: () => void;
}
export interface AudioStreamCallbacks {
  onAudioChunk?: (audioData: ArrayBuffer, itemId: string) => void;
  onAudioStart?: (itemId: string) => void;
  onAudioEnd?: (itemId: string) => void;
}
export class RealtimeService {
  private session: RealtimeSession | null = null;
  private agent: RealtimeAgent | null = null;

  createAgent(config: VoiceAgentConfig): RealtimeAgent {
    this.agent = new RealtimeAgent({
      name: config.name,
      instructions: config.instructions,
    });
    return this.agent;
  }

  createSession(
    agent: RealtimeAgent,
    callbacks: RealtimeServiceCallbacks,
    model: string = 'gpt-realtime'
  ): RealtimeSession {
    this.session = new RealtimeSession(agent, { model });

    this.session.on('connected', (): void => {
      callbacks.onConnected();
    });

    this.session.on('disconnected', (): void => {
      callbacks.onDisconnected();
    });

    this.session.on('error', (error: Error): void => {
      callbacks.onError(error);
    });

    this.session.on('history_updated', (history: ConversationHistoryItem[]): void => {
      callbacks.onHistoryUpdated(history);
    });

    if (callbacks.onToolStart) {
      this.session.on('agent_tool_start', (event: { tool: { name: string } }): void => {
        callbacks.onToolStart?.(event.tool.name);
      });
    }

    if (callbacks.onToolEnd) {
      this.session.on('agent_tool_end', (event: { tool: { name: string }; output: unknown }): void => {
        callbacks.onToolEnd?.(event.tool.name, event.output);
      });
    }

    if (callbacks.onAudioInterrupted) {
      this.session.on('audio_interrupted', (): void => {
        callbacks.onAudioInterrupted?.();
      });
    }

    return this.session;
  }

  async connect(apiKey: string): Promise<void> {
    if (!this.session) {
      throw new Error('Session not created. Call createSession first.');
    }
    await this.session.connect({ apiKey });
  }

  async disconnect(): Promise<void> {
    if (!this.session) {
      return;
    }

    try {
      if ('close' in this.session && typeof (this.session as { close?: () => Promise<void> }).close === 'function') {
        await (this.session as { close: () => Promise<void> }).close();
      }
    } catch (error) {
      console.error('Error during disconnect:', error);
    }

    this.session = null;
  }

  async sendMessage(message: string): Promise<void> {
    if (!this.session) {
      throw new Error('Session not connected');
    }
    await this.session.sendMessage(message);
  }

  async interrupt(): Promise<void> {
    if (!this.session) {
      throw new Error('Session not connected');
    }
    await this.session.interrupt();
  }

  getSession(): RealtimeSession | null {
    return this.session;
  }

  getAgent(): RealtimeAgent | null {
    return this.agent;
  }
  createSessionWithAudioControl(
    agent: RealtimeAgent,
    callbacks: RealtimeServiceCallbacks & AudioStreamCallbacks,
    model: string = 'gpt-realtime'
  ): RealtimeSession {
    // Use WebSocket transport for full audio control
    const transport = new OpenAIRealtimeWebSocket();

    this.session = new RealtimeSession(agent, {
      transport,
      model,
    });

    // Standard event handlers
    this.session.on('connected', (): void => {
      callbacks.onConnected();
    });

    this.session.on('disconnected', (): void => {
      callbacks.onDisconnected();
    });

    this.session.on('error', (error: Error): void => {
      callbacks.onError(error);
    });

    this.session.on('history_updated', (history: ConversationHistoryItem[]): void => {
      callbacks.onHistoryUpdated(history);
    });

    // AUDIO STREAMING EVENTS
    this.session.on('audio_start', (event: { itemId: string }): void => {
      console.log('🎵 Audio started:', event.itemId);
      callbacks.onAudioStart?.(event.itemId);
    });

    this.session.on('audio', (event: { data: ArrayBuffer; itemId: string }): void => {
      // This fires for EACH audio chunk as it streams in
      console.log('🎵 Audio chunk received:', event.data.byteLength, 'bytes');
      callbacks.onAudioChunk?.(event.data, event.itemId);
    });

    this.session.on('audio_stopped', (event: { itemId: string }): void => {
      console.log('🎵 Audio stopped:', event.itemId);
      callbacks.onAudioEnd?.(event.itemId);
    });

    this.session.on('audio_interrupted', (): void => {
      console.log('🎵 Audio interrupted by user');
      callbacks.onAudioInterrupted?.();
    });

    return this.session;
  }
}