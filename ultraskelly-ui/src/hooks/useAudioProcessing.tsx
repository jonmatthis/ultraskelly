import { useRef, useState, useCallback } from 'react';

interface AudioProcessor {
    process: (data: Float32Array) => Float32Array;
}

export interface UseAudioProcessingResult {
    queueAudio: (pcmData: ArrayBuffer, analyserNode?: AnalyserNode | null) => void;
    stopAudio: () => void;
    isPlaying: boolean;
    addProcessor: (processor: AudioProcessor) => void;
    clearProcessors: () => void;
    audioContext: AudioContext | null;
}

export function useAudioProcessing(): UseAudioProcessingResult {
    const audioContextRef = useRef<AudioContext | null>(null);
    const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
    const processorsRef = useRef<AudioProcessor[]>([]);
    const audioQueueRef = useRef<AudioBuffer[]>([]);
    const isPlayingQueueRef = useRef<boolean>(false);
    const analyserNodeRef = useRef<AnalyserNode | null>(null);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

    const getAudioContext = (): AudioContext => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext({ sampleRate: 24000 });
        }
        return audioContextRef.current;
    };

    const addProcessor = useCallback((processor: AudioProcessor): void => {
        processorsRef.current.push(processor);
    }, []);

    const clearProcessors = useCallback((): void => {
        processorsRef.current = [];
    }, []);

    const convertToAudioBuffer = useCallback((pcmData: ArrayBuffer): AudioBuffer => {
        const audioContext = getAudioContext();

        // Convert Int16 PCM to Float32 for Web Audio API
        const int16Array = new Int16Array(pcmData);
        let float32Array = new Float32Array(int16Array.length);

        for (let i = 0; i < int16Array.length; i++) {
            const sample = int16Array[i] ?? 0;
            float32Array[i] = sample / 32768.0;
        }

        // Apply all processors in sequence
        for (const processor of processorsRef.current) {
            float32Array = processor.process(float32Array);
        }

        // Create AudioBuffer from processed PCM data
        const audioBuffer = audioContext.createBuffer(
            1, // mono
            float32Array.length,
            24000 // sample rate
        );

        // Copy the float data into the buffer
        audioBuffer.copyToChannel(float32Array, 0);

        return audioBuffer;
    }, []);

    const playNextInQueue = useCallback((): void => {
        if (audioQueueRef.current.length === 0) {
            isPlayingQueueRef.current = false;
            setIsPlaying(false);
            currentSourceRef.current = null;
            return;
        }

        const audioContext = getAudioContext();

        // Resume context if suspended
        if (audioContext.state === 'suspended') {
            audioContext.resume().catch(err => {
                console.error('Failed to resume audio context:', err);
            });
        }

        const audioBuffer = audioQueueRef.current.shift()!;

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        currentSourceRef.current = source;

        // Connect to analyser (if available) then to destination
        if (analyserNodeRef.current) {
            source.connect(analyserNodeRef.current);
            // Analyser is already connected to destination
        } else {
            source.connect(audioContext.destination);
        }

        source.onended = (): void => {
            // Immediately play next chunk when this one ends
            playNextInQueue();
        };

        // Start immediately
        source.start(0);
        setIsPlaying(true);
    }, []);

    const queueAudio = useCallback((
        pcmData: ArrayBuffer,
        analyserNode?: AnalyserNode | null
    ): void => {
        // Store analyser node reference
        if (analyserNode !== undefined) {
            analyserNodeRef.current = analyserNode;
        }

        // Convert PCM to AudioBuffer
        const audioBuffer = convertToAudioBuffer(pcmData);

        // Add to queue
        audioQueueRef.current.push(audioBuffer);

        // Start playing if not already playing
        if (!isPlayingQueueRef.current) {
            isPlayingQueueRef.current = true;
            playNextInQueue();
        }
    }, [convertToAudioBuffer, playNextInQueue]);

    const stopAudio = useCallback((): void => {
        // Clear the queue
        audioQueueRef.current = [];
        isPlayingQueueRef.current = false;

        // Stop current source
        if (currentSourceRef.current) {
            try {
                currentSourceRef.current.stop();
                currentSourceRef.current.disconnect();
            } catch (error) {
                console.error('Error stopping audio:', error);
            }
            currentSourceRef.current = null;
        }

        setIsPlaying(false);
    }, []);

    return {
        queueAudio,
        stopAudio,
        isPlaying,
        addProcessor,
        clearProcessors,
        audioContext: audioContextRef.current,
    };
}
