import {EphemeralKeyResponse, HealthCheckResponse, SendMessageRequest, SendMessageResponse} from "../types/types.ts";

export interface AudioUploadRequest {
  itemId: string;
  audioData: ArrayBuffer;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export class BackendService {
  constructor(private baseUrl: string) {}

  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await fetch(`${this.baseUrl}/health`);

    if (!response.ok) {
      throw new Error('Backend health check failed');
    }

    return response.json();
  }

  async getEphemeralKey(): Promise<string> {
    const response = await fetch(`${this.baseUrl}/ephemeral-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`Failed to get ephemeral key: ${response.statusText}`);
    }

    const data: EphemeralKeyResponse = await response.json();
    return data.value;
  }

  async sendMessage(request: SendMessageRequest): Promise<SendMessageResponse> {
    const response = await fetch(`${this.baseUrl}/api/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to send message: ${response.statusText}`);
    }

    return response.json();
  }

  async saveConversationHistory(
    conversationId: string,
    history: unknown[]
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/conversations/${conversationId}/history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history }),
    });

    if (!response.ok) {
      throw new Error(`Failed to save conversation: ${response.statusText}`);
    }
  }

  async uploadAudioChunk(request: AudioUploadRequest): Promise<void> {
    const formData = new FormData();

    // Convert ArrayBuffer to Blob
    const audioBlob = new Blob([request.audioData], { type: 'audio/pcm' });

    formData.append('audio', audioBlob, `${request.itemId}.pcm`);
    formData.append('itemId', request.itemId);
    formData.append('timestamp', request.timestamp);
    formData.append('metadata', JSON.stringify(request.metadata || {}));

    const response = await fetch(`${this.baseUrl}/api/audio/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to upload audio chunk');
    }
  }

  async uploadFullAudioStream(
    itemId: string,
    audioChunks: ArrayBuffer[]
  ): Promise<void> {
    // Concatenate all chunks
    const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
    const fullAudio = new Uint8Array(totalLength);

    let offset = 0;
    for (const chunk of audioChunks) {
      fullAudio.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    await this.uploadAudioChunk({
      itemId,
      audioData: fullAudio.buffer,
      timestamp: new Date().toISOString(),
    });
  }
}
