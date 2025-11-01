
import { useCallback, useRef } from 'react';
import type { BackendService } from '../services/backendService';
import {AudioAnalyzer, AudioMetrics} from "../services/audioAnalyzer.ts";

export interface UploadStrategy {
  shouldUpload: (audioData: Int16Array, metrics: AudioMetrics) => boolean;
}

export class SpeechOnlyStrategy implements UploadStrategy {
  private analyzer = new AudioAnalyzer();

  shouldUpload(audioData: Int16Array): boolean {
    return this.analyzer.detectSpeech(audioData);
  }
}

export class QualityThresholdStrategy implements UploadStrategy {
  private analyzer = new AudioAnalyzer();

  constructor(private minRms: number = 1000) {}

  shouldUpload(audioData: Int16Array): boolean {
    const metrics = this.analyzer.analyze(audioData);
    return metrics.rms >= this.minRms && !metrics.clipping;
  }
}

export function useConditionalUpload(
  backend: BackendService,
  strategy: UploadStrategy
) {
  const pendingChunks = useRef<Map<string, ArrayBuffer[]>>(new Map());

  const processChunk = useCallback(async (
    itemId: string,
    audioData: ArrayBuffer
  ): Promise<void> => {
    const int16Data = new Int16Array(audioData);

    // Check if this chunk should be uploaded
    if (strategy.shouldUpload(int16Data)) {
      // Get existing chunks for this item
      if (!pendingChunks.current.has(itemId)) {
        pendingChunks.current.set(itemId, []);
      }

      const chunks = pendingChunks.current.get(itemId)!;
      chunks.push(audioData);

      // Upload when we have enough chunks (e.g., 1 second of audio)
      // At 24kHz, 16-bit mono: 24000 samples/sec * 2 bytes = 48000 bytes/sec
      const totalBytes = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);

      if (totalBytes >= 48000) { // ~1 second
        await backend.uploadFullAudioStream(itemId, chunks);
        pendingChunks.current.delete(itemId);
      }
    }
  }, [backend, strategy]);

  return { processChunk };
}
