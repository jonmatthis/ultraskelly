import fs from 'fs/promises';

export async function convertPcmToWav(
  pcmPath: string,
  wavPath: string,
  sampleRate: number = 24000,
  channels: number = 1,
  bitDepth: number = 16
): Promise<void> {
  const pcmData = await fs.readFile(pcmPath);
  const wavBuffer = createWavHeader(pcmData.length, sampleRate, channels, bitDepth);

  const finalBuffer = Buffer.concat([wavBuffer, pcmData]);
  await fs.writeFile(wavPath, finalBuffer);
}

function createWavHeader(
  dataLength: number,
  sampleRate: number,
  channels: number,
  bitDepth: number
): Buffer {
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * channels * (bitDepth / 8);
  const blockAlign = channels * (bitDepth / 8);

  // RIFF header
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + dataLength, 4);
  header.write('WAVE', 8);

  // fmt chunk
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16); // Subchunk1Size
  header.writeUInt16LE(1, 20); // AudioFormat (PCM)
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitDepth, 34);

  // data chunk
  header.write('data', 36);
  header.writeUInt32LE(dataLength, 40);

  return header;
}
