import { useEffect, useRef, useState } from 'react';

interface AudioWaveformVisualizerProps {
    analyserNode: AnalyserNode | null;
    height?: number;
    backgroundColor?: string;
    waveColor?: string;
    lineWidth?: number;
    timelineSeconds?: number;
}

export function AudioWaveformVisualizer({
                                            analyserNode,
                                            height = 150,
                                            backgroundColor = '#1a1a2e',
                                            waveColor = '#00ff88',
                                            lineWidth = 2,
                                            timelineSeconds = 10,
                                        }: AudioWaveformVisualizerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const animationFrameRef = useRef<number | null>(null);
    const waveformHistoryRef = useRef<number[]>([]);
    const [isAnimating, setIsAnimating] = useState<boolean>(false);

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container || !analyserNode) {
            setIsAnimating(false);
            return;
        }

        const canvasContext = canvas.getContext('2d');
        if (!canvasContext) {
            console.error('Failed to get canvas 2D context');
            return;
        }

        analyserNode.fftSize = 2048;
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const resizeCanvas = (): void => {
            if (!canvas || !container) return;

            const width = container.clientWidth;
            const pixelRatio = window.devicePixelRatio || 1;

            canvas.width = width * pixelRatio;
            canvas.height = height * pixelRatio;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
            canvasContext.scale(pixelRatio, pixelRatio);
        };

        resizeCanvas();
        setIsAnimating(true);

        const drawWaveform = (): void => {
            if (!canvas || !container) return;

            const width = container.clientWidth;

            // Get current audio data
            analyserNode.getByteTimeDomainData(dataArray);

            // Sample multiple points from the current buffer to capture waveform detail
            // We'll take every Nth sample to get good detail without overwhelming memory
            const samplingRate = Math.max(1, Math.floor(bufferLength / 100)); // ~100 samples per frame

            for (let i = 0; i < bufferLength; i += samplingRate) {
                const normalized = ((dataArray[i] ?? 128) - 128) / 128.0;
                waveformHistoryRef.current.push(normalized);
            }

            // Calculate max history length based on canvas width and timeline duration
            // We want enough samples to fill the width with detail
            const samplesPerSecond = (60 * 100) / samplingRate; // ~60fps * samples per frame
            const maxSamples = Math.floor(timelineSeconds * samplesPerSecond);

            // Keep only the most recent samples
            if (waveformHistoryRef.current.length > maxSamples) {
                waveformHistoryRef.current = waveformHistoryRef.current.slice(-maxSamples);
            }

            // Clear canvas
            canvasContext.fillStyle = backgroundColor;
            canvasContext.fillRect(0, 0, width, height);

            // Draw timeline grid
            canvasContext.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            canvasContext.lineWidth = 1;

            // Vertical grid lines (time markers)
            const pixelsPerSecond = width / timelineSeconds;

            for (let i = 0; i <= timelineSeconds; i++) {
                const x = width - (i * pixelsPerSecond);
                canvasContext.beginPath();
                canvasContext.moveTo(x, 0);
                canvasContext.lineTo(x, height);
                canvasContext.stroke();

                // Time labels
                if (i > 0) {
                    canvasContext.fillStyle = 'rgba(255, 255, 255, 0.3)';
                    canvasContext.font = '10px monospace';
                    canvasContext.fillText(`-${i}s`, x + 2, 12);
                }
            }

            // Horizontal center line
            canvasContext.beginPath();
            canvasContext.moveTo(0, height / 2);
            canvasContext.lineTo(width, height / 2);
            canvasContext.stroke();

            // Draw scrolling waveform with detail
            if (waveformHistoryRef.current.length > 1) {
                canvasContext.strokeStyle = waveColor;
                canvasContext.lineWidth = lineWidth;
                canvasContext.beginPath();

                const pixelsPerSample = width / maxSamples;
                const historyLength = waveformHistoryRef.current.length;

                for (let i = 0; i < historyLength; i++) {
                    const amplitude = waveformHistoryRef.current[i] ?? 0;
                    const x = width - (historyLength - i) * pixelsPerSample;
                    const y = height / 2 - (amplitude * height * 0.45);

                    if (i === 0) {
                        canvasContext.moveTo(x, y);
                    } else {
                        canvasContext.lineTo(x, y);
                    }
                }

                canvasContext.stroke();
            }

            // Draw "now" indicator line
            canvasContext.strokeStyle = 'rgba(255, 0, 0, 0.6)';
            canvasContext.lineWidth = 2;
            canvasContext.beginPath();
            canvasContext.moveTo(width - 1, 0);
            canvasContext.lineTo(width - 1, height);
            canvasContext.stroke();

            animationFrameRef.current = requestAnimationFrame(drawWaveform);
        };

        drawWaveform();

        // Handle window resize
        const handleResize = (): void => {
            resizeCanvas();
        };

        window.addEventListener('resize', handleResize);

        return (): void => {
            if (animationFrameRef.current !== null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            window.removeEventListener('resize', handleResize);
            setIsAnimating(false);
            waveformHistoryRef.current = [];
        };
    }, [analyserNode, height, backgroundColor, waveColor, lineWidth, timelineSeconds]);

    return (
        <div ref={containerRef} style={{ position: 'relative', width: '100%' }}>
            <canvas
                ref={canvasRef}
                style={{
                    display: 'block',
                    border: '1px solid #444',
                    borderRadius: '8px',
                    width: '100%',
                }}
            />
            {!analyserNode && (
                <div
                    style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        color: '#888',
                        fontSize: '14px',
                    }}
                >
                    No audio playing
                </div>
            )}
            {isAnimating && (
                <div
                    style={{
                        position: 'absolute',
                        top: '8px',
                        right: '8px',
                        color: waveColor,
                        fontSize: '10px',
                        opacity: 0.6,
                        backgroundColor: 'rgba(0, 0, 0, 0.5)',
                        padding: '4px 8px',
                        borderRadius: '4px',
                    }}
                >
                    🎵 Recording
                </div>
            )}
        </div>
    );
}
