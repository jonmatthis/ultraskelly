// src/worklets/microphone-processor.worklet.ts

class MicrophoneProcessor extends AudioWorkletProcessor {
    process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean {
        const input = inputs[0];

        if (input && input.length > 0) {
            const inputChannel = input[0];
            this.port.postMessage(inputChannel);
        }

        return true;
    }
}

registerProcessor('microphone-processor', MicrophoneProcessor);