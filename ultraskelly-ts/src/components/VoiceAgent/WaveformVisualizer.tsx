import { useEffect, useRef, useState } from 'react';

interface AudioWaveformVisualizerProps {
    analyserNode: AnalyserNode | null;
    height?: number;
    backgroundColor?: string;
    waveColor?: string;
    lineWidth?: number;
}

export function AudioWaveformVisualizer({
                                            analyserNode,
                                            height = 150,
                                            backgroundColor = '#1a1a2e',
                                            waveColor = '#00ff88',
                                            lineWidth = 2,
                                        }: AudioWaveformVisualizerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const animationFrameRef = useRef<number | null>(null);
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

            analyserNode.getByteTimeDomainData(dataArray);

            canvasContext.fillStyle = backgroundColor;
            canvasContext.fillRect(0, 0, width, height);

            canvasContext.lineWidth = lineWidth;
            canvasContext.strokeStyle = waveColor;
            canvasContext.beginPath();

            const sliceWidth = width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = (dataArray[i] ?? 128) / 128.0;
                const y = (v * height) / 2;

                if (i === 0) {
                    canvasContext.moveTo(x, y);
                } else {
                    canvasContext.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasContext.lineTo(width, height / 2);
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
        };
    }, [analyserNode, height, backgroundColor, waveColor, lineWidth]);

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
                        bottom: '8px',
                        right: '8px',
                        color: waveColor,
                        fontSize: '10px',
                        opacity: 0.6,
                    }}
                >
                    🎵 Visualizing
                </div>
            )}
        </div>
    );
}
