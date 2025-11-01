
import { useState, useCallback, useRef, useEffect } from 'react';
import { AudioService } from '../services/audioService';
import type { AudioProcessor } from '../services/audioService';

export interface UseAudioProcessingResult {
  isPlaying: boolean;
  playAudio: (pcm16Data: ArrayBuffer) => Promise<void>;
  stopAudio: () => void;
  addProcessor: (processor: AudioProcessor) => void;
  clearProcessors: () => void;
}

export function useAudioProcessing(): UseAudioProcessingResult {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioServiceRef = useRef<AudioService | null>(null);

  useEffect(() => {
    audioServiceRef.current = new AudioService();
    return () => {
      audioServiceRef.current?.stopPlayback();
    };
  }, []);

  const playAudio = useCallback(async (pcm16Data: ArrayBuffer): Promise<void> => {
    if (!audioServiceRef.current) {
      throw new Error('Audio service not initialized');
    }
    setIsPlaying(true);
    await audioServiceRef.current.playAudioChunk(pcm16Data);
  }, []);

  const stopAudio = useCallback((): void => {
    audioServiceRef.current?.stopPlayback();
    setIsPlaying(false);
  }, []);

  const addProcessor = useCallback((processor: AudioProcessor): void => {
    audioServiceRef.current?.addProcessor(processor);
  }, []);

  const clearProcessors = useCallback((): void => {
    audioServiceRef.current?.clearProcessors();
  }, []);

  return {
    isPlaying,
    playAudio,
    stopAudio,
    addProcessor,
    clearProcessors,
  };
}