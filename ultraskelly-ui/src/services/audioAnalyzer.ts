
export interface AudioMetrics {
  rms: number;
  peak: number;
  clipping: boolean;
  silenceRatio: number;
}

export class AudioAnalyzer {
  analyze(audioData: Int16Array): AudioMetrics {
    let sumSquares = 0;
    let peak = 0;
    let silentSamples = 0;
    const silenceThreshold = 500;

    for (let i = 0; i < audioData.length; i++) {
      const absValue = Math.abs(audioData[i]);
      sumSquares += audioData[i] * audioData[i];
      peak = Math.max(peak, absValue);

      if (absValue < silenceThreshold) {
        silentSamples++;
      }
    }

    const rms = Math.sqrt(sumSquares / audioData.length);
    const clipping = peak >= 32767;
    const silenceRatio = silentSamples / audioData.length;

    return { rms, peak, clipping, silenceRatio };
  }

  detectSpeech(audioData: Int16Array): boolean {
    const metrics = this.analyze(audioData);
    // Speech typically has RMS > 1000 and low silence ratio
    return metrics.rms > 1000 && metrics.silenceRatio < 0.7;
  }
}
