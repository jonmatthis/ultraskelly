
import type { AudioProcessor } from '../services/audioService';

export class PitchShiftProcessor implements AudioProcessor {
  constructor(private pitchFactor: number = 1.0) {}

  process(audioData: Int16Array): Int16Array {
    if (this.pitchFactor === 1.0) return audioData;

    const outputLength = Math.floor(audioData.length / this.pitchFactor);
    const output = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const position = i * this.pitchFactor;
      const index = Math.floor(position);
      const fraction = position - index;

      if (index + 1 < audioData.length) {
        // Linear interpolation
        output[i] = Math.round(
          audioData[index] * (1 - fraction) +
          audioData[index + 1] * fraction
        );
      } else {
        output[i] = audioData[index];
      }
    }

    return output;
  }
}

// 2. ECHO EFFECT PROCESSOR
export class EchoProcessor implements AudioProcessor {
  private buffer: Int16Array;
  private bufferPosition: number = 0;

  constructor(
    private delayMs: number = 300,
    private decay: number = 0.5,
    private sampleRate: number = 24000
  ) {
    const delaySamples = Math.floor((delayMs / 1000) * sampleRate);
    this.buffer = new Int16Array(delaySamples);
  }

  process(audioData: Int16Array): Int16Array {
    const output = new Int16Array(audioData.length);

    for (let i = 0; i < audioData.length; i++) {
      // Mix original with delayed signal
      const delayed = this.buffer[this.bufferPosition];
      output[i] = Math.max(-32768, Math.min(32767,
        audioData[i] + delayed * this.decay
      ));

      // Store current sample in buffer
      this.buffer[this.bufferPosition] = audioData[i];
      this.bufferPosition = (this.bufferPosition + 1) % this.buffer.length;
    }

    return output;
  }
}

// 3. SIMPLE COMPRESSOR PROCESSOR
export class CompressorProcessor implements AudioProcessor {
  constructor(
    private threshold: number = 20000,
    private ratio: number = 4.0
  ) {}

  process(audioData: Int16Array): Int16Array {
    const output = new Int16Array(audioData.length);

    for (let i = 0; i < audioData.length; i++) {
      const sample = audioData[i];
      const absValue = Math.abs(sample);

      if (absValue > this.threshold) {
        // Apply compression
        const excess = absValue - this.threshold;
        const compressed = this.threshold + (excess / this.ratio);
        output[i] = sample < 0 ? -compressed : compressed;
      } else {
        output[i] = sample;
      }
    }

    return output;
  }
}

// 4. BANDPASS FILTER PROCESSOR
export class BandpassFilterProcessor implements AudioProcessor {
  private x1: number = 0;
  private x2: number = 0;
  private y1: number = 0;
  private y2: number = 0;

  constructor(
    private lowFreq: number = 300,
    private highFreq: number = 3400,
    private sampleRate: number = 24000
  ) {}

  process(audioData: Int16Array): Int16Array {
    const output = new Int16Array(audioData.length);

    for (let i = 0; i < audioData.length; i++) {
      const x0 = audioData[i];

      // Simple IIR bandpass filter (approximation)
      const y0 = x0 - this.x2 + 0.9 * this.y1 - 0.81 * this.y2;

      output[i] = Math.max(-32768, Math.min(32767, y0));

      // Update state
      this.x2 = this.x1;
      this.x1 = x0;
      this.y2 = this.y1;
      this.y1 = y0;
    }

    return output;
  }
}


// Volume adjustment processor
export class VolumeProcessor implements AudioProcessor {
  constructor(private gain: number = 1.0) {}

  process(audioData: Int16Array): Int16Array {
    const processed = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      processed[i] = Math.max(-32768, Math.min(32767, audioData[i] * this.gain));
    }
    return processed;
  }
}

// Simple noise gate processor
export class NoiseGateProcessor implements AudioProcessor {
  constructor(private threshold: number = 100) {}

  process(audioData: Int16Array): Int16Array {
    const processed = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      processed[i] = Math.abs(audioData[i]) > this.threshold ? audioData[i] : 0;
    }
    return processed;
  }
}

// Example: Convert to different format or apply effects
export class CustomEffectProcessor implements AudioProcessor {
  process(audioData: Int16Array): Int16Array {
    // Your custom audio processing here
    // E.g., apply reverb, equalization, compression, etc.
    return audioData;
  }
}
