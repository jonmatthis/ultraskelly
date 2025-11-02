import { useState, useCallback, useEffect } from 'react';
import { BackendService } from '../services/backendService';
import { SendMessageRequest } from '../types/types';

export interface UseBackendAPIResult {
    isHealthy: boolean;
    isChecking: boolean;
    getEphemeralKey: () => Promise<string>;
    sendMessage: (request: SendMessageRequest) => Promise<void>;
    saveHistory: (conversationId: string, history: unknown[]) => Promise<void>;
    uploadFullAudioStream: (itemId: string, chunks: ArrayBuffer[]) => Promise<void>;
}

export function useBackendAPI(baseUrl: string): UseBackendAPIResult {
    const [service] = useState(() => new BackendService(baseUrl));
    const [isHealthy, setIsHealthy] = useState(false);
    const [isChecking, setIsChecking] = useState(true);

    const checkHealth = useCallback(async (): Promise<void> => {
        try {
            setIsChecking(true);
            await service.healthCheck();
            setIsHealthy(true);
        } catch (error) {
            console.error('Backend health check failed:', error);
            setIsHealthy(false);
        } finally {
            setIsChecking(false);
        }
    }, [service]);

    useEffect(() => {
        checkHealth();
    }, [checkHealth]);

    const getEphemeralKey = useCallback(async (): Promise<string> => {
        return service.getEphemeralKey();
    }, [service]);

    const sendMessage = useCallback(async (request: SendMessageRequest): Promise<void> => {
        await service.sendMessage(request);
    }, [service]);

    const saveHistory = useCallback(async (
        conversationId: string,
        history: unknown[]
    ): Promise<void> => {
        await service.saveConversationHistory(conversationId, history);
    }, [service]);

    const uploadFullAudioStream = useCallback(async (
        itemId: string,
        chunks: ArrayBuffer[]
    ): Promise<void> => {
        await service.uploadFullAudioStream(itemId, chunks);
    }, [service]);

    return {
        isHealthy,
        isChecking,
        getEphemeralKey,
        sendMessage,
        saveHistory,
        uploadFullAudioStream,
    };
}
