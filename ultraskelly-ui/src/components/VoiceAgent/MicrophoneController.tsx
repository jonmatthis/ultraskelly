import { useEffect, useRef, useState } from 'react';
import { useVoiceAgentContext } from './VoiceAgentProvider';

export function MicrophoneController() {
    const { voiceAgent } = useVoiceAgentContext();
    const [isMuted, setIsMuted] = useState<boolean>(false);
    const [micError, setMicError] = useState<string | null>(null);

    const mediaStreamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

    useEffect(() => {
        if (!voiceAgent.isConnected) return;

        const startMicrophone = async (): Promise<void> => {
            try {
                console.log('🎤 Starting microphone capture...');

                // Request 24kHz sample rate directly from getUserMedia
                mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 24000,  // Request 24kHz to match OpenAI Realtime API
                        channelCount: 1,    // Mono
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    },
                });

                // Check what sample rate we actually got
                const audioTrack = mediaStreamRef.current.getAudioTracks()[0];
                const settings = audioTrack?.getSettings();
                const actualSampleRate = settings?.sampleRate || 48000; // Default to 48kHz if not reported

                console.log('🎚️ Requested sample rate: 24000 Hz');
                console.log('🎚️ Actual sample rate:', actualSampleRate, 'Hz');

                // Create AudioContext at the actual sample rate of the microphone
                audioContextRef.current = new AudioContext({
                    sampleRate: actualSampleRate
                });

                // Build the worklet URL
                const workletUrl = new URL(
                    '../../worklets/microphone-processor.worklet.ts',
                    import.meta.url
                ).href;

                console.log('🔍 Attempting to load worklet from:', workletUrl);

                // Verify the worklet file exists before loading
                try {
                    const checkResponse = await fetch(workletUrl, { method: 'HEAD' });
                    if (!checkResponse.ok) {
                        throw new Error(`Worklet file not found at ${workletUrl}. Status: ${checkResponse.status}`);
                    }
                    console.log('✅ Worklet file found');
                } catch (fetchError) {
                    const errorMsg = `Cannot find worklet file at ${workletUrl}. Check the file path!`;
                    console.error('❌', errorMsg, fetchError);
                    throw new Error(errorMsg);
                }

                // Load the worklet
                console.log('📦 Loading worklet module...');
                await audioContextRef.current.audioWorklet.addModule(workletUrl);
                console.log('✅ Worklet module loaded');

                sourceRef.current = audioContextRef.current.createMediaStreamSource(
                    mediaStreamRef.current
                );

                workletNodeRef.current = new AudioWorkletNode(
                    audioContextRef.current,
                    'microphone-processor'
                );

                const targetSampleRate = 24000;

                workletNodeRef.current.port.onmessage = (event: MessageEvent<Float32Array>): void => {
                    if (isMuted || !voiceAgent.session) return;

                    let audioData: Float32Array = event.data;

                    // Resample if necessary (e.g., 48kHz -> 24kHz)
                    if (actualSampleRate !== targetSampleRate) {
                        audioData = resampleAudio(audioData, actualSampleRate, targetSampleRate);
                    }

                    // Convert Float32 to Int16 PCM
                    const int16Data = new Int16Array(audioData.length);
                    for (let i = 0; i < audioData.length; i++) {
                        const sample: number = audioData[i] ?? 0;
                        const s: number = Math.max(-1, Math.min(1, sample));
                        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }

                    try {
                        voiceAgent.session.sendAudio(int16Data.buffer);
                    } catch (error) {
                        console.error('Failed to send audio:', error);
                    }
                };

                sourceRef.current.connect(workletNodeRef.current);

                console.log('✅ Microphone capture started successfully');

                if (actualSampleRate !== targetSampleRate) {
                    console.log(`⚙️ Will resample from ${actualSampleRate} Hz to ${targetSampleRate} Hz`);
                }

                setMicError(null);
            } catch (error) {
                const errorMessage: string = error instanceof Error ? error.message : 'Unknown error';
                console.error('❌ Microphone error:', error);
                setMicError(errorMessage);
            }
        };

        startMicrophone();

        return (): void => {
            console.log('🛑 Stopping microphone capture...');

            if (workletNodeRef.current) {
                workletNodeRef.current.port.onmessage = null;
                workletNodeRef.current.disconnect();
            }
            if (sourceRef.current) {
                sourceRef.current.disconnect();
            }
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }

            if (mediaStreamRef.current) {
                mediaStreamRef.current.getTracks().forEach((track: MediaStreamTrack) => {
                    track.stop();
                    console.log('Track stopped:', track.kind);
                });
            }

            workletNodeRef.current = null;
            sourceRef.current = null;
            audioContextRef.current = null;
            mediaStreamRef.current = null;
        };
    }, [voiceAgent.isConnected, voiceAgent.session, isMuted]);

    const toggleMute = (): void => {
        setIsMuted((prev: boolean) => !prev);
    };

    if (!voiceAgent.isConnected) {
        return null;
    }

    return (
        <div className="microphone-controller">
            <button
                onClick={toggleMute}
                className={`mute-btn ${isMuted ? 'muted' : ''}`}
            >
                {isMuted ? '🔇 Unmute' : '🎤 Mute'}
            </button>

            {micError && (
                <div className="mic-error">
                    ⚠️ Microphone error: {micError}
                </div>
            )}

            {!micError && (
                <div className="mic-status">
                    {isMuted ? '🔇 Muted' : '🎤 Listening...'}
                </div>
            )}
        </div>
    );
}

// Simple linear interpolation resampler
function resampleAudio(input: Float32Array, inputRate: number, outputRate: number): Float32Array {
    if (inputRate === outputRate) {
        return input;
    }

    const ratio = inputRate / outputRate;
    const outputLength = Math.round(input.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
        const srcIndex = i * ratio;
        const srcIndexFloor = Math.floor(srcIndex);
        const srcIndexCeil = Math.min(srcIndexFloor + 1, input.length - 1);
        const t = srcIndex - srcIndexFloor;

        const sample1 = input[srcIndexFloor] ?? 0;
        const sample2 = input[srcIndexCeil] ?? 0;

        // Linear interpolation
        output[i] = sample1 + (sample2 - sample1) * t;
    }

    return output;
}