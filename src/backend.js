import * as tf from '@tensorflow/tfjs';

export async function initializeBackend(preferred) {
  try {
    if (!await tf.setBackend(preferred)) throw new Error('Backend unavailable');
  } catch {
    await tf.setBackend('cpu');
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
  const { numTensors, numBytesInGPUAllocated = 0 } = tf.memory();
  return { tensors: numTensors, gpuBytes: numBytesInGPUAllocated };
}
