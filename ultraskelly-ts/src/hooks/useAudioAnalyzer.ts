import { useRef, useEffect, useState } from 'react';

export interface UseAudioAnalyserResult {
    analyserNode: AnalyserNode | null;
}

interface UseAudioAnalyzerProps {
    audioContext: AudioContext | null;
}

export function useAudioAnalyzer(props: UseAudioAnalyzerProps): UseAudioAnalyserResult {
    const { audioContext } = props;
    const analyserNodeRef = useRef<AnalyserNode | null>(null);
    const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

    useEffect(() => {
        if (!audioContext) {
            return;
        }

        try {
            analyserNodeRef.current = audioContext.createAnalyser();
            analyserNodeRef.current.fftSize = 2048;
            analyserNodeRef.current.smoothingTimeConstant = 0.8;

            // Connect analyser directly to destination so audio plays through
            analyserNodeRef.current.connect(audioContext.destination);

            setAnalyserNode(analyserNodeRef.current);

            console.log('✅ Audio analyser initialized');
        } catch (error) {
            console.error('Failed to initialize audio analyser:', error);
        }

        return (): void => {
            if (analyserNodeRef.current) {
                analyserNodeRef.current.disconnect();
            }
            analyserNodeRef.current = null;
            setAnalyserNode(null);
        };
    }, [audioContext]);

    return {
        analyserNode,
    };
}
