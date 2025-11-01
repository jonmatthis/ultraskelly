import { useAudioProcessing } from '../../hooks/useAudioProcessing';
import { useAudioBuffer } from '../../hooks/useAudioBuffer';
import {
    VolumeProcessor,
    NoiseGateProcessor,
    CompressorProcessor,
    BandpassFilterProcessor,
} from '../../utils/audioProcessors';
import { useEffect, useRef, useState } from 'react';
import { useVoiceAgentContext } from './VoiceAgentProvider';
import { AudioAnalyzer } from '../../services/audioAnalyzer';
import {AudioEvent, AudioStartEvent, AudioStoppedEvent} from "../../types/types.ts";

export function AudioStreamController(): JSX.Element {
    const { voiceAgent, backend } = useVoiceAgentContext();
    const audioProcessing = useAudioProcessing();
    const audioBuffer = useAudioBuffer();
    const uploadIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const analyzerRef = useRef(new AudioAnalyzer());

    const [metrics, setMetrics] = useState({ rms: 0, peak: 0, clipping: false });

    // Set up enhanced audio processors on mount
    useEffect(() => {
        // 1. Noise gate - remove background noise (more aggressive than before)
        audioProcessing.addProcessor(new NoiseGateProcessor(200));

        // 2. Bandpass filter - focus on speech frequencies (300-3400 Hz)
        // This is KEY for voice clarity - removes rumble and hiss
        audioProcessing.addProcessor(new BandpassFilterProcessor(300, 3400));

        // 3. Compressor - even out volume levels for consistent listening
        audioProcessing.addProcessor(new CompressorProcessor(20000, 4.0));

        // 4. Volume boost - make it louder (increased from 1.2x to 1.5x)
        audioProcessing.addProcessor(new VolumeProcessor(1.5));

        return () => {
            audioProcessing.clearProcessors();
        };
    }, [audioProcessing]);

    // Handle audio streaming with quality analysis
    useEffect(() => {
        if (!voiceAgent.session) return;

        const handleAudioStart = (event: AudioStartEvent): void => {
            console.log('📦 Starting audio buffer for:', event.itemId);
            audioBuffer.startBuffer(event.itemId);
        };

        const handleAudioChunk = async (event: AudioEvent): Promise<void> => {
            console.log('🎵 Processing audio chunk:', event.data.byteLength, 'bytes');

            // Analyze audio quality
            const int16Data = new Int16Array(event.data);
            const audioMetrics = analyzerRef.current.analyze(int16Data);
            setMetrics(audioMetrics);

            // Log warnings for quality issues
            if (audioMetrics.clipping) {
                console.warn('⚠️ Audio clipping detected! Peak:', audioMetrics.peak);
            }

            // Buffer the audio
            audioBuffer.addChunk(event.itemId, event.data);

            // Process and play the audio
            try {
                await audioProcessing.playAudio(event.data);
            } catch (error) {
                console.error('Failed to play audio:', error);
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

        return () => {
            voiceAgent.session?.off('audio_start', handleAudioStart);
            voiceAgent.session?.off('audio', handleAudioChunk);
            voiceAgent.session?.off('audio_stopped', handleAudioStopped);
            voiceAgent.session?.off('audio_interrupted', handleAudioInterrupted);
        };
    }, [voiceAgent.session, audioProcessing, audioBuffer, backend]);

    // Periodic upload of buffered audio (every 5 seconds)
    useEffect(() => {
        if (!voiceAgent.isConnected) return;

        uploadIntervalRef.current = setInterval(async () => {
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
