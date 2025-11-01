
import OpenAI from 'openai';
import { convertPcmToWav } from '../utils/audioConverter';
import fs from "fs/promises";

const openai = new OpenAI({ apiKey: process.env["OPENAI_API_KEY"] });

export async function transcribeAudioFile(pcmPath: string): Promise<string> {
  // Convert PCM to WAV first
  const wavPath = pcmPath.replace('.pcm', '.wav');
  await convertPcmToWav(pcmPath, wavPath);

  // Transcribe with Whisper
  const transcription = await openai.audio.transcriptions.create({
    file: await fs.readFile(wavPath),
    model: 'whisper-1',
  });

  return transcription.text;
}
