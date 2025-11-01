
import React, { useEffect, useState } from 'react';
import { useAudioProcessing } from '../../hooks/useAudioProcessing';
import { useConditionalUpload } from '../../hooks/useConditionalUpload';
import {
  VolumeProcessor,
  NoiseGateProcessor,
  CompressorProcessor,
  BandpassFilterProcessor,
} from '../../utils/audioProcessors';
import { SpeechOnlyStrategy } from '../../hooks/useConditionalUpload';
import { AudioAnalyzer } from '../../services/audioAnalyzer';
import {useVoiceAgentContext} from "./VoiceAgentProvider.tsx";

export function EnhancedAudioController(): JSX.Element {
  const { voiceAgent, backend } = useVoiceAgentContext();
  const audioProcessing = useAudioProcessing();
  const [metrics, setMetrics] = useState({ rms: 0, peak: 0, clipping: false });
  const analyzer = new AudioAnalyzer();

  // Only upload chunks with actual speech
  const { processChunk } = useConditionalUpload(
    backend,
    new SpeechOnlyStrategy()
  );

  // Configure audio processing pipeline
  useEffect(() => {
    // 1. Noise gate - remove background noise
    audioProcessing.addProcessor(new NoiseGateProcessor(200));

    // 2. Bandpass filter - focus on speech frequencies (300-3400 Hz)
    audioProcessing.addProcessor(new BandpassFilterProcessor(300, 3400));

    // 3. Compressor - even out volume levels
    audioProcessing.addProcessor(new CompressorProcessor(20000, 4.0));

    // 4. Volume boost - make it louder
    audioProcessing.addProcessor(new VolumeProcessor(1.5));

    // 5. Optional: Add subtle echo for richness
    // audioProcessing.addProcessor(new EchoProcessor(150, 0.2));

    return () => {
      audioProcessing.clearProcessors();
    };
  }, [audioProcessing]);

  // Handle audio streaming with analysis
  useEffect(() => {
    if (!voiceAgent.session) return;

    voiceAgent.session.on('audio', async (event: { data: ArrayBuffer; itemId: string }) => {
      const int16Data = new Int16Array(event.data);

      // Analyze audio quality
      const audioMetrics = analyzer.analyze(int16Data);
      setMetrics(audioMetrics);

      // Log warnings
      if (audioMetrics.clipping) {
        console.warn('⚠️ Audio clipping detected!');
      }

      // 1. Conditionally upload to backend (only speech)
      await processChunk(event.itemId, event.data);

      // 2. Process and play
      await audioProcessing.playAudio(event.data);
    });

    voiceAgent.session.on('audio_interrupted', () => {
      audioProcessing.stopAudio();
    });

  }, [voiceAgent.session, audioProcessing, processChunk, analyzer]);

  return (
    <div className="enhanced-audio-status">
      <div className="audio-metrics">
        <div>RMS: {Math.round(metrics.rms)}</div>
        <div>Peak: {Math.round(metrics.peak)}</div>
        {metrics.clipping && <div className="warning">⚠️ Clipping!</div>}
      </div>
    </div>
  );
}
