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
