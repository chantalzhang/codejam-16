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
