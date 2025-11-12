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
      console.log('🔌 Starting connection process...');
      updateStatus('connecting');

      const service = new RealtimeService();
      serviceRef.current = service;

      console.log('👤 Creating agent...');
      const agent = service.createAgent(config);

      console.log('🎯 Creating session with audio control...');
      const session = service.createSessionWithAudioControl(agent, {
        onError: (error: Error): void => {
          console.error('❌ Session error:', error);
          updateStatus('error', error.message);
        },
        onHistoryUpdated: (history: ConversationHistoryItem[]): void => {
          console.log('📝 History updated:', history.length, 'items');
          historyCallbackRef.current?.(history);
        },
      }, config.model);

      setState(prev => ({ ...prev, agent, session }));

      console.log('🔐 Connecting with API key...');
      console.log('Key starts with:', apiKey.substring(0, 10) + '...');

      // When this resolves, the connection is established
      await service.connect(apiKey);

      console.log('✅ Connection complete!');

      // Update status to connected - the session.connect() promise resolving means we're connected
      updateStatus('connected');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('❌ Connection failed:', errorMessage, error);
      updateStatus('error', errorMessage);
      throw error;
    }
  }, [updateStatus]);

  const disconnect = useCallback(async (): Promise<void> => {
    if (!serviceRef.current) {
      return;
    }

    try {
      console.log('🔌 Disconnecting...');
      updateStatus('disconnecting');
      await serviceRef.current.disconnect();

      console.log('✅ Disconnected successfully');
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
      console.error('❌ Disconnect error:', error);
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