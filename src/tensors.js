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

export async function computeGradients(task, batchSpec) {
  const started = performance.now();
  const batch = await task.makeBatch(batchSpec);
  let result;
  try {
    result = tf.tidy(() => tf.variableGrads(() => task.loss(batch), task.variables));
    const gradients = await readValues(task.variables.map(variable => result.grads[variable.name]));
    const loss = (await result.value.data())[0];
    return { gradients, loss, computeMs: performance.now() - started };
  } finally {
    tf.dispose(result);
    tf.dispose(batch);
  }
}
