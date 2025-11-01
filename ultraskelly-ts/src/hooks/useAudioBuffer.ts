
import { useState, useCallback, useRef } from 'react';

export interface AudioBufferItem {
  itemId: string;
  chunks: ArrayBuffer[];
  startTime: Date;
  endTime?: Date;
}

export interface UseAudioBufferResult {
  buffers: Map<string, AudioBufferItem>;
  addChunk: (itemId: string, chunk: ArrayBuffer) => void;
  startBuffer: (itemId: string) => void;
  endBuffer: (itemId: string) => AudioBufferItem | null;
  clearBuffer: (itemId: string) => void;
  getAllBuffers: () => AudioBufferItem[];
}

export function useAudioBuffer(): UseAudioBufferResult {
  const [buffers] = useState(() => new Map<string, AudioBufferItem>());
  const buffersRef = useRef(buffers);

  const startBuffer = useCallback((itemId: string): void => {
    buffersRef.current.set(itemId, {
      itemId,
      chunks: [],
      startTime: new Date(),
    });
  }, []);

  const addChunk = useCallback((itemId: string, chunk: ArrayBuffer): void => {
    const buffer = buffersRef.current.get(itemId);
    if (buffer) {
      buffer.chunks.push(chunk);
    }
  }, []);

  const endBuffer = useCallback((itemId: string): AudioBufferItem | null => {
    const buffer = buffersRef.current.get(itemId);
    if (buffer) {
      buffer.endTime = new Date();
    }
    return buffer || null;
  }, []);

  const clearBuffer = useCallback((itemId: string): void => {
    buffersRef.current.delete(itemId);
  }, []);

  const getAllBuffers = useCallback((): AudioBufferItem[] => {
    return Array.from(buffersRef.current.values());
  }, []);

  return {
    buffers: buffersRef.current,
    addChunk,
    startBuffer,
    endBuffer,
    clearBuffer,
    getAllBuffers,
  };
}
