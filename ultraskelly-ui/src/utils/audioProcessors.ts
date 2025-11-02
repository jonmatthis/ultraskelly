// Base interface for audio processors
interface AudioProcessor {
    process: (data: Float32Array) => Float32Array;
}

/**
 * Noise Gate Processor
 * Silences audio below a threshold to reduce noise
 */
export class NoiseGateProcessor implements AudioProcessor {
    private threshold: number;

    constructor(threshold: number) {
        // Convert from linear amplitude to decibels threshold
        this.threshold = threshold / 32768.0; // Normalize to Float32 range
    }

    process(data: Float32Array): Float32Array {
        const output = new Float32Array(data.length);

        for (let i = 0; i < data.length; i++) {
            const sample = data[i] ?? 0;
            const absValue = Math.abs(sample);

            if (absValue > this.threshold) {
                output[i] = sample;
            } else {
                output[i] = 0; // Gate closed - silence
            }
        }

        return output;
    }
}

/**
 * Bandpass Filter Processor
 * Simple IIR bandpass filter using biquad coefficients
 */
export class BandpassFilterProcessor implements AudioProcessor {
    private lowFreq: number;
    private highFreq: number;
    private sampleRate: number = 24000;

    // Filter state variables
    private x1: number = 0;
    private x2: number = 0;
    private y1: number = 0;
    private y2: number = 0;

    // Biquad coefficients
    private b0: number = 1;
    private b1: number = 0;
    private b2: number = -1;
    private a1: number = 0;
    private a2: number = 0;

    constructor(lowFreq: number, highFreq: number) {
        this.lowFreq = lowFreq;
        this.highFreq = highFreq;
        this.calculateCoefficients();
    }

    private calculateCoefficients(): void {
        const centerFreq = Math.sqrt(this.lowFreq * this.highFreq);
        const bandwidth = this.highFreq - this.lowFreq;

        const omega = 2 * Math.PI * centerFreq / this.sampleRate;
        const sn = Math.sin(omega);
        const cs = Math.cos(omega);
        const alpha = sn * Math.sinh((Math.log(2) / 2) * bandwidth * omega / sn);

        const a0 = 1 + alpha;
        this.b0 = alpha / a0;
        this.b1 = 0;
        this.b2 = -alpha / a0;
        this.a1 = -2 * cs / a0;
        this.a2 = (1 - alpha) / a0;
    }

    process(data: Float32Array): Float32Array {
        const output = new Float32Array(data.length);

        for (let i = 0; i < data.length; i++) {
            const x0 = data[i] ?? 0;

            // Biquad filter implementation (Direct Form II)
            const y0 = this.b0 * x0 + this.b1 * this.x1 + this.b2 * this.x2
                - this.a1 * this.y1 - this.a2 * this.y2;

            // Update state variables
            this.x2 = this.x1;
            this.x1 = x0;
            this.y2 = this.y1;
            this.y1 = y0;

            output[i] = y0;
        }

        return output;
    }
}

/**
 * Compressor Processor
 * Dynamic range compression with threshold and ratio
 */
export class CompressorProcessor implements AudioProcessor {
    private threshold: number;
    private ratio: number;
    private attackTime: number = 0.003; // 3ms
    private releaseTime: number = 0.05; // 50ms
    private envelope: number = 0;

    constructor(threshold: number, ratio: number) {
        // Normalize threshold to Float32 range (0-1)
        this.threshold = threshold / 32768.0;
        this.ratio = ratio;
    }

    process(data: Float32Array): Float32Array {
        const output = new Float32Array(data.length);

        for (let i = 0; i < data.length; i++) {
            const sample = data[i] ?? 0;
            const absValue = Math.abs(sample);

            // Envelope follower
            if (absValue > this.envelope) {
                this.envelope += (absValue - this.envelope) * this.attackTime;
            } else {
                this.envelope += (absValue - this.envelope) * this.releaseTime;
            }

            // Calculate gain reduction
            let gain = 1.0;
            if (this.envelope > this.threshold) {
                const excess = this.envelope - this.threshold;
                const compressed = excess / this.ratio;
                gain = (this.threshold + compressed) / this.envelope;
            }

            output[i] = sample * gain;
        }

        return output;
    }
}

/**
 * Volume Processor
 * Simple gain/volume adjustment
 */
export class VolumeProcessor implements AudioProcessor {
    private gain: number;

    constructor(gain: number) {
        this.gain = gain;
    }

    process(data: Float32Array): Float32Array {
        const output = new Float32Array(data.length);

        for (let i = 0; i < data.length; i++) {
            const sample = data[i] ?? 0;
            output[i] = sample * this.gain;
        }

        return output;
    }

    setGain(gain: number): void {
        this.gain = gain;
    }
}
