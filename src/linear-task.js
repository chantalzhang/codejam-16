import * as tf from '@tensorflow/tfjs';

// This is a replaceable wiring test, not the project's chosen model or workload.
export const name = 'Line fit (2 parameters)';
export const description = 'Test task: learn y = 2x + 1 from generated examples. Checks training works, not GPU speed.';
export const defaults = { rounds: 40, batchSize: 32, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 };
export const taskId = 'linear-function-v1';

export function createTask() {
  const variables = tf.tidy(() => [
    tf.variable(tf.zeros([1, 1])),
    tf.variable(tf.zeros([1])),
  ]);
  const predict = xs => xs.matMul(variables[0]).add(variables[1]);
  return {
    id: taskId,
    variables,
    makeBatch({ step, offset, count }) {
      // Global indices make the total batch identical regardless of worker count.
      return tf.tidy(() => {
        const values = Array.from({ length: count }, (_, i) =>
          ((step * 97 + (offset + i) * 37) % 1024) / 511.5 - 1);
        const xs = tf.tensor2d(values, [count, 1]);
        return { xs, ys: xs.mul(2).add(1) };
      });
    },
    loss({ xs, ys }) {
      return predict(xs).sub(ys).square().mean();
    },
    async evaluate() {
      const loss = tf.tidy(() => {
        const xs = tf.linspace(-1, 1, 65).reshape([65, 1]);
        return predict(xs).sub(xs.mul(2).add(1)).square().mean();
      });
      try { return (await loss.data())[0]; }
      finally { loss.dispose(); }
    },
    dispose() { tf.dispose(variables); },
  };
}
