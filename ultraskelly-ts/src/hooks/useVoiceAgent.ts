
import { useState, useCallback, useRef } from 'react';
import { RealtimeService } from '../services/realtimeService';
import {ConnectionStatus, ConversationHistoryItem, VoiceAgentConfig, VoiceAgentState} from "../types/types.ts";

export interface UseVoiceAgentResult extends VoiceAgentState {
  connect: (apiKey: string, config: VoiceAgentConfig) => Promise<void>;
  disconnect: () => Promise<void>;
  sendMessage: (message: string) => Promise<void>;
  interrupt: () => Promise<void>;
  onHistoryUpdate: (callback: (history: ConversationHistoryItem[]) => void) => void;
}

export function useVoiceAgent(): UseVoiceAgentResult {
  const [state, setState] = useState<VoiceAgentState>({
    status: 'disconnected',
    isConnected: false,
    session: null,
    agent: null,
    error: null,
  });

  const serviceRef = useRef<RealtimeService | null>(null);
  const historyCallbackRef = useRef<((history: ConversationHistoryItem[]) => void) | null>(null);

  const updateStatus = useCallback((status: ConnectionStatus, error: string | null = null): void => {
    setState(prev => ({
      ...prev,
      status,
      isConnected: status === 'connected',
      error,
    }));
  }, []);

  const connect = useCallback(async (apiKey: string, config: VoiceAgentConfig): Promise<void> => {
    try {
      updateStatus('connecting');

      const service = new RealtimeService();
      serviceRef.current = service;

      const agent = service.createAgent(config);

      const session = service.createSession(agent, {
        onConnected: (): void => {
          updateStatus('connected');
        },
        onDisconnected: (): void => {
          updateStatus('disconnected');
        },
        onError: (error: Error): void => {
          console.error('Session error:', error);
          updateStatus('error', error.message);
        },
        onHistoryUpdated: (history: ConversationHistoryItem[]): void => {
          historyCallbackRef.current?.(history);
        },
      }, config.model);

      setState(prev => ({ ...prev, agent, session }));

      await service.connect(apiKey);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      updateStatus('error', errorMessage);
      throw error;
    }
  }, [updateStatus]);

  const disconnect = useCallback(async (): Promise<void> => {
    if (!serviceRef.current) {
      return;
    }

    try {
      updateStatus('disconnecting');
      await serviceRef.current.disconnect();

      setState({
        status: 'disconnected',
        isConnected: false,
        session: null,
        agent: null,
        error: null,
      });

      serviceRef.current = null;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      updateStatus('error', errorMessage);
    }
  }, [updateStatus]);

  const sendMessage = useCallback(async (message: string): Promise<void> => {
    if (!serviceRef.current) {
      throw new Error('Service not initialized');
    }
    await serviceRef.current.sendMessage(message);
  }, []);

  const interrupt = useCallback(async (): Promise<void> => {
    if (!serviceRef.current) {
      throw new Error('Service not initialized');
    }
    await serviceRef.current.interrupt();
  }, []);

  const onHistoryUpdate = useCallback((
    callback: (history: ConversationHistoryItem[]) => void
  ): void => {
    historyCallbackRef.current = callback;
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    interrupt,
    onHistoryUpdate,
  };
}
