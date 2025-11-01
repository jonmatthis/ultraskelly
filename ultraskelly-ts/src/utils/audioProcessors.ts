import type { AudioProcessor } from '../services/audioService';

export class VolumeProcessor implements AudioProcessor {
    constructor(private gain: number) {}

    process(audioData: Int16Array): Int16Array {
        const output = new Int16Array(audioData.length);

        for (let i = 0; i < audioData.length; i++) {
            const amplified = audioData[i] * this.gain;
            // Clamp to prevent overflow
            output[i] = Math.max(-32768, Math.min(32767, amplified));
        }

        return output;
    }
}

export class NoiseGateProcessor implements AudioProcessor {
    constructor(private threshold: number) {}

    process(audioData: Int16Array): Int16Array {
        const output = new Int16Array(audioData.length);

        for (let i = 0; i < audioData.length; i++) {
            const absValue = Math.abs(audioData[i]);
            // If below threshold, silence it; otherwise pass through
            output[i] = absValue < this.threshold ? 0 : audioData[i];
        }

        return output;
    }
}

export class CompressorProcessor implements AudioProcessor {
    constructor(
        private threshold: number,
        private ratio: number
    ) {}

    process(audioData: Int16Array): Int16Array {
        const output = new Int16Array(audioData.length);

        for (let i = 0; i < audioData.length; i++) {
            const sample = audioData[i];
            const absValue = Math.abs(sample);

            if (absValue > this.threshold) {
                // Apply compression above threshold
                const excess = absValue - this.threshold;
                const compressed = this.threshold + (excess / this.ratio);
                output[i] = sample >= 0 ? compressed : -compressed;
            } else {
                output[i] = sample;
            }
        }

        return output;
    }
}

export class BandpassFilterProcessor implements AudioProcessor {
    private lowCutoff: number;
    private highCutoff: number;
    private sampleRate: number;

    constructor(lowCutoff: number, highCutoff: number, sampleRate: number = 24000) {
        this.lowCutoff = lowCutoff;
        this.highCutoff = highCutoff;
        this.sampleRate = sampleRate;
    }

    process(audioData: Int16Array): Int16Array {
        // Simple FFT-free approximation using a combination of filters
        // For production, you'd want a proper FFT-based bandpass filter

        // High-pass filter (removes frequencies below lowCutoff)
        const highPassed = this.highPassFilter(audioData, this.lowCutoff);

        // Low-pass filter (removes frequencies above highCutoff)
        const bandPassed = this.lowPassFilter(highPassed, this.highCutoff);

        return bandPassed;
    }

    private highPassFilter(data: Int16Array, cutoff: number): Int16Array {
        const output = new Int16Array(data.length);
        const RC = 1.0 / (cutoff * 2 * Math.PI);
        const dt = 1.0 / this.sampleRate;
        const alpha = RC / (RC + dt);

        output[0] = data[0];

        for (let i = 1; i < data.length; i++) {
            output[i] = alpha * (output[i - 1] + data[i] - data[i - 1]);
        }

        return output;
    }

    private lowPassFilter(data: Int16Array, cutoff: number): Int16Array {
        const output = new Int16Array(data.length);
        const RC = 1.0 / (cutoff * 2 * Math.PI);
        const dt = 1.0 / this.sampleRate;
        const alpha = dt / (RC + dt);

        output[0] = data[0];

        for (let i = 1; i < data.length; i++) {
            output[i] = output[i - 1] + alpha * (data[i] - output[i - 1]);
        }

        return output;
    }
}

export class EchoProcessor implements AudioProcessor {
    private delayMs: number;
    private decay: number;
    private sampleRate: number;
    private buffer: Int16Array;
    private writeIndex: number;

    constructor(delayMs: number, decay: number, sampleRate: number = 24000) {
        this.delayMs = delayMs;
        this.decay = decay;
        this.sampleRate = sampleRate;

        const bufferSize = Math.floor((delayMs / 1000) * sampleRate);
        this.buffer = new Int16Array(bufferSize);
        this.writeIndex = 0;
    }

    process(audioData: Int16Array): Int16Array {
        const output = new Int16Array(audioData.length);

        for (let i = 0; i < audioData.length; i++) {
            // Read delayed sample
            const delayedSample = this.buffer[this.writeIndex];

            // Mix original with delayed signal
            const mixed = audioData[i] + (delayedSample * this.decay);
            output[i] = Math.max(-32768, Math.min(32767, mixed));

            // Write current output to delay buffer
            this.buffer[this.writeIndex] = output[i];

            // Advance circular buffer index
            this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        }

        return output;
    }
}
