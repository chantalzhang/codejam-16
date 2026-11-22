import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

export const BACKENDS = ['webgl', 'webgpu', 'cpu'];
export const backendLabel = name => ({ webgl: 'WebGL', webgpu: 'WebGPU', cpu: 'CPU (not GPU)' })[name] ?? name;

// Tries the preferred backend, then WebGL, then CPU.
export async function initializeBackend(preferred) {
  for (const name of new Set([preferred, 'webgl', 'cpu'])) {
    try {
      if (await tf.setBackend(name)) break;
    } catch {
      // Not available in this browser; try the next one.
    }
  }
  await tf.ready();
  return tf.getBackend();
}

// After a GPU reset (WebGL context lost), pending TF.js reads never finish, so the
// caller must give up on this page instead of waiting.
export function onGpuReset(callback) {
  if (tf.getBackend() === 'webgl') {
    tf.backend().gpgpu.gl.canvas.addEventListener('webglcontextlost', callback);
  }
}

export function memoryUsage() {
  const { numTensors, numBytesInGPUAllocated, numBytesAllocatedInGPU } = tf.memory(); // WebGL, WebGPU names
  return { tensors: numTensors, gpuBytes: numBytesInGPUAllocated ?? numBytesAllocatedInGPU ?? 0 };
}
