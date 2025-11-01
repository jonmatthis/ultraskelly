
import { useAudioProcessing } from '../../hooks/useAudioProcessing';
import { useAudioBuffer } from '../../hooks/useAudioBuffer';
import { VolumeProcessor, NoiseGateProcessor } from '../../utils/audioProcessors';
import {useEffect, useRef} from "react";
import {useVoiceAgentContext} from "./VoiceAgentProvider.tsx";

export function AudioStreamController(): JSX.Element {
  const { voiceAgent, backend } = useVoiceAgentContext();
  const audioProcessing = useAudioProcessing();
  const audioBuffer = useAudioBuffer();
  const uploadIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Set up audio processors on mount
  useEffect(() => {
    // Add volume boost processor
    audioProcessing.addProcessor(new VolumeProcessor(1.2));

    // Add noise gate processor
    audioProcessing.addProcessor(new NoiseGateProcessor(100));

    return () => {
      audioProcessing.clearProcessors();
    };
  }, [audioProcessing]);

  // Handle audio streaming
  useEffect(() => {
    if (!voiceAgent.session) return;

    // When audio starts, create a new buffer
    voiceAgent.session.on('audio_start', (event: { itemId: string }) => {
      console.log('📦 Starting audio buffer for:', event.itemId);
      audioBuffer.startBuffer(event.itemId);
    });

    // When audio chunk arrives
    voiceAgent.session.on('audio', async (event: { data: ArrayBuffer; itemId: string }) => {
      console.log('🎵 Processing audio chunk:', event.data.byteLength, 'bytes');

      // 1. Store in buffer for backend upload
      audioBuffer.addChunk(event.itemId, event.data);

      // 2. Process and play the audio
      try {
        await audioProcessing.playAudio(event.data);
      } catch (error) {
        console.error('Failed to play audio:', error);
      }
    });

    // When audio stream ends
    voiceAgent.session.on('audio_stopped', async (event: { itemId: string }) => {
      console.log('🏁 Audio stream ended:', event.itemId);

      const completedBuffer = audioBuffer.endBuffer(event.itemId);

      if (completedBuffer && completedBuffer.chunks.length > 0) {
        // Upload complete audio stream to backend
        try {
          await backend.uploadFullAudioStream(
            completedBuffer.itemId,
            completedBuffer.chunks
          );
          console.log('✅ Audio uploaded to backend');
        } catch (error) {
          console.error('❌ Failed to upload audio:', error);
        }

        // Clean up buffer
        audioBuffer.clearBuffer(event.itemId);
      }
    });

    // When user interrupts
    voiceAgent.session.on('audio_interrupted', () => {
      console.log('⏸️ User interrupted - stopping playback');
      audioProcessing.stopAudio();
    });

  }, [voiceAgent.session, audioProcessing, audioBuffer, backend]);

  // Optional: Upload buffered audio periodically (every 5 seconds)
  useEffect(() => {
    if (!voiceAgent.isConnected) return;

    uploadIntervalRef.current = setInterval(async () => {
      const allBuffers = audioBuffer.getAllBuffers();

      for (const buffer of allBuffers) {
        // Upload if buffer has been active for more than 5 seconds
        const duration = Date.now() - buffer.startTime.getTime();
        if (duration > 5000 && buffer.chunks.length > 0) {
          try {
            await backend.uploadFullAudioStream(buffer.itemId, buffer.chunks);
            console.log('📤 Periodic upload complete for:', buffer.itemId);
          } catch (error) {
            console.error('Failed periodic upload:', error);
          }
        }
      }
    }, 5000);

    return () => {
      if (uploadIntervalRef.current) {
        clearInterval(uploadIntervalRef.current);
      }
    };
  }, [voiceAgent.isConnected, audioBuffer, backend]);

  return (
    <div className="audio-stream-status">
      {audioProcessing.isPlaying && (
        <div className="audio-indicator">
          🔊 Audio playing...
        </div>
      )}
    </div>
  );
}
