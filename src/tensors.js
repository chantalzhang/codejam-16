import * as tf from '@tensorflow/tfjs';

export async function readValues(tensors) {
  return Promise.all(tensors.map(async tensor => Array.from(await tensor.data())));
}

export function validateValues(values, variables) {
  if (!Array.isArray(values) || values.length !== variables.length ||
      values.some((value, i) => !Array.isArray(value) || value.length !== variables[i].size ||
        !value.every(Number.isFinite))) {
    throw new Error('Tensor values must match the model and contain only finite numbers.');
  }
}

// Messages carry weights as raw float32 bytes in base64: ~4x smaller than JSON number
// arrays, and still plain JSON for the relay.
export function packValues(values) {
  const bytes = new Uint8Array(Float32Array.from(values.flat()).buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export function unpackValues(text, variables) {
  const binary = typeof text === 'string' ? atob(text) : '';
  if (binary.length !== 4 * variables.reduce((sum, variable) => sum + variable.size, 0)) {
    throw new Error('Packed values do not match the model.');
  }
  const floats = new Float32Array(Uint8Array.from(binary, char => char.charCodeAt(0)).buffer);
  let offset = 0;
  return variables.map(variable => Array.from(floats.subarray(offset, offset += variable.size)));
}

export function assignValues(variables, values) {
  validateValues(values, variables);
  tf.tidy(() => variables.forEach((variable, i) => {
    variable.assign(tf.tensor(values[i], variable.shape, 'float32'));
  }));
}

export function createOptimizer(name, learningRate) {
  if (name === 'adam') return tf.train.adam(learningRate);
  if (name === 'sgd') return tf.train.sgd(learningRate);
  throw new Error(`Unknown optimizer: ${name}`);
}

// Starts from the shared weights, trains for several steps, and returns the weight change.
export async function trainLocally(task, optimizer, weights, { firstStep, localSteps, offset, count }) {
  const started = performance.now();
  assignValues(task.variables, weights);
  const losses = [];
  try {
    for (let i = 0; i < localSteps; i++) {
      const batch = await task.makeBatch({ step: firstStep + i, offset, count });
      try {
        losses.push(optimizer.minimize(() => task.loss(batch), true, task.variables));
      } finally {
        tf.dispose(batch);
      }
    }
    const after = await readValues(task.variables);
    const meanLoss = tf.tidy(() => tf.addN(losses).div(localSteps));
    const loss = (await meanLoss.data())[0];
    meanLoss.dispose();
    const delta = after.map((values, i) => values.map((value, j) => value - weights[i][j]));
    return { delta, loss, computeMs: performance.now() - started };
  } finally {
    tf.dispose(losses);
  }
}
