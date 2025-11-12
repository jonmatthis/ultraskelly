import { useAudioProcessing } from '../../hooks/useAudioProcessing';
import { useAudioBuffer } from '../../hooks/useAudioBuffer';
import {
    VolumeProcessor,
    NoiseGateProcessor,
} from '../../utils/audioProcessors';
import { useEffect, useRef, useState } from 'react';
import { useVoiceAgentContext } from '../../services/VoiceAgentProvider.tsx';
import { AudioAnalyzer } from '../../services/audioAnalyzer';
import { AudioEvent, AudioStartEvent, AudioStoppedEvent } from "../../types/types";
import { AudioWaveformVisualizer } from "./WaveformVisualizer.tsx";
import { useAudioAnalyzer } from "../../hooks/useAudioAnalyzer.ts";

export function AudioStreamController() {
    const { voiceAgent, backend } = useVoiceAgentContext();
    const audioProcessing = useAudioProcessing();
    const audioBuffer = useAudioBuffer();
    const audioAnalyser = useAudioAnalyzer({ audioContext: audioProcessing.audioContext });
    const uploadIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const analyzerRef = useRef(new AudioAnalyzer());

    const [metrics, setMetrics] = useState({ rms: 0, peak: 0, clipping: false });

    useEffect(() => {
        audioProcessing.addProcessor(new NoiseGateProcessor(200));
        audioProcessing.addProcessor(new VolumeProcessor(1.2)); // Slight boost

        return (): void => {
            audioProcessing.clearProcessors();
        };
    }, [audioProcessing]);

    useEffect(() => {
        if (!voiceAgent.session) return;

        const handleAudioStart = (event: AudioStartEvent): void => {
            console.log('📦 Starting audio buffer for:', event.itemId);
            audioBuffer.startBuffer(event.itemId);
        };

        const handleAudioChunk = (event: AudioEvent): void => {
            console.log('🎵 Received chunk:', event.data.byteLength, 'bytes');

            const int16Data = new Int16Array(event.data);
            const audioMetrics = analyzerRef.current.analyze(int16Data);
            setMetrics(audioMetrics);

            if (audioMetrics.clipping) {
                console.warn('⚠️ Audio clipping detected! Peak:', audioMetrics.peak);
            }

            audioBuffer.addChunk(event.itemId, event.data);

            try {
                // Queue the audio chunk for sequential playback
                audioProcessing.queueAudio(event.data, audioAnalyser.analyserNode);
            } catch (error) {
                console.error('Failed to queue audio:', error);
            }
        };

        const handleAudioStopped = async (event: AudioStoppedEvent): Promise<void> => {
            console.log('🛑 Audio stream ended:', event.itemId);

            const completedBuffer = audioBuffer.endBuffer(event.itemId);

            if (completedBuffer && completedBuffer.chunks.length > 0) {
                try {
                    await backend.uploadFullAudioStream(
                        completedBuffer.itemId,
                        completedBuffer.chunks
                    );
                    console.log('✅ Audio uploaded to backend');
                } catch (error) {
                    console.error('❌ Failed to upload audio:', error);
                }

                audioBuffer.clearBuffer(event.itemId);
            }
        };

        const handleAudioInterrupted = (): void => {
            console.log('⏸️ User interrupted - stopping playback');
            audioProcessing.stopAudio();
        };

        voiceAgent.session.on('audio_start', handleAudioStart);
        voiceAgent.session.on('audio', handleAudioChunk);
        voiceAgent.session.on('audio_stopped', handleAudioStopped);
        voiceAgent.session.on('audio_interrupted', handleAudioInterrupted);

        return (): void => {
            voiceAgent.session?.off('audio_start', handleAudioStart);
            voiceAgent.session?.off('audio', handleAudioChunk);
            voiceAgent.session?.off('audio_stopped', handleAudioStopped);
            voiceAgent.session?.off('audio_interrupted', handleAudioInterrupted);
        };
    }, [voiceAgent.session, audioProcessing, audioBuffer, backend, audioAnalyser]);

    useEffect(() => {
        if (!voiceAgent.isConnected) return;

        uploadIntervalRef.current = setInterval(async (): Promise<void> => {
            const allBuffers = audioBuffer.getAllBuffers();

            for (const buffer of allBuffers) {
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

        return (): void => {
            if (uploadIntervalRef.current) {
                clearInterval(uploadIntervalRef.current);
            }
        };
    }, [voiceAgent.isConnected, audioBuffer, backend]);

    return (
        <div className="audio-stream-status" style={{ width: '100%', maxWidth: '1200px', margin: '0 auto' }}>
            {/* Waveform Visualizer - now responsive */}
            <AudioWaveformVisualizer
                analyserNode={audioAnalyser.analyserNode}
                height={150}
            />

            {audioProcessing.isPlaying && (
                <div className="audio-indicator">
                    🔊 Audio playing...
                </div>
            )}

            {voiceAgent.isConnected && (
                <div className="audio-metrics">
                    <div className="metric">
                        <span className="label">RMS:</span> {Math.round(metrics.rms)}
                    </div>
                    <div className="metric">
                        <span className="label">Peak:</span> {Math.round(metrics.peak)}
                    </div>
                    {metrics.clipping && (
                        <div className="metric warning">⚠️ Clipping!</div>
                    )}
                </div>
            )}
        </div>
    );
}
