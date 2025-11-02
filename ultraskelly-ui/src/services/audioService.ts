
export interface AudioProcessor {
  process: (audioData: Int16Array) => Int16Array;
}

export class AudioService {
  private audioContext: AudioContext | null = null;
  private audioQueue: AudioBuffer[] = [];
  private isPlaying: boolean = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private processors: AudioProcessor[] = [];

  constructor() {
    if (typeof window !== 'undefined' && 'AudioContext' in window) {
      this.audioContext = new AudioContext();
    }
  }

  // Add custom audio processor (e.g., noise reduction, volume boost)
  addProcessor(processor: AudioProcessor): void {
    this.processors.push(processor);
  }

  // Process audio through all processors
  private processAudio(audioData: Int16Array): Int16Array {
    let processed = audioData;
    for (const processor of this.processors) {
      processed = processor.process(processed);
    }
    return processed;
  }

  // Convert PCM16 audio to AudioBuffer
  private pcm16ToAudioBuffer(pcm16: Int16Array, sampleRate: number = 24000): AudioBuffer {
    if (!this.audioContext) {
      throw new Error('AudioContext not available');
    }

    const audioBuffer = this.audioContext.createBuffer(
      1, // mono
      pcm16.length,
      sampleRate
    );

    const channelData = audioBuffer.getChannelData(0);

    // Convert Int16 to Float32 (-1.0 to 1.0)
    for (let i = 0; i < pcm16.length; i++) {
      channelData[i] = pcm16[i] / 32768.0;
    }

    return audioBuffer;
  }

  // Play audio chunk
  async playAudioChunk(pcm16Data: ArrayBuffer): Promise<void> {
    if (!this.audioContext) {
      throw new Error('AudioContext not available');
    }

    // Convert ArrayBuffer to Int16Array
    const int16Data = new Int16Array(pcm16Data);

    // Process the audio
    const processedData = this.processAudio(int16Data);

    // Convert to AudioBuffer
    const audioBuffer = this.pcm16ToAudioBuffer(processedData);

    // Queue for playback
    this.audioQueue.push(audioBuffer);

    // Start playback if not already playing
    if (!this.isPlaying) {
      await this.playNextInQueue();
    }
  }

  private async playNextInQueue(): Promise<void> {
    if (!this.audioContext || this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;
    const audioBuffer = this.audioQueue.shift()!;

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    this.currentSource = source;

    source.onended = () => {
      this.currentSource = null;
      this.playNextInQueue();
    };

    source.start();
  }

  // Stop playback immediately
  stopPlayback(): void {
    if (this.currentSource) {
      this.currentSource.stop();
      this.currentSource = null;
    }
    this.audioQueue = [];
    this.isPlaying = false;
  }

  // Clear all processors
  clearProcessors(): void {
    this.processors = [];
  }
}
