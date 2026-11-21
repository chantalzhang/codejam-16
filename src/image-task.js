import * as tf from '@tensorflow/tfjs';

// Benchmark workload: enough GPU work per step to measure against network time.
// Not the final demo model.
export const name = 'Image benchmark (small CNN)';
export const description = 'Benchmark task: a small CNN classifies generated 32×32 stripe and ring images (4 classes). Measures phone compute vs. network time.';
export const defaults = { steps: 30, batchSize: 32, learningRate: 0.1 };
export const taskId = 'image-benchmark-v1';

const SIZE = 32;
const CLASSES = 4;
const EVAL_COUNT = 256;
const EVAL_START = -EVAL_COUNT; // Training indices are positive, so these never overlap.

// Integer hash to [0, 1): identical on every browser, unlike Math.sin.
function random(i, salt) {
  let h = Math.imul(i ^ Math.imul(salt, 0x9e3779b9), 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

// Each class is a different wave direction: horizontal, vertical, diagonal, rings.
// Frequency and phase vary per image, so the model must learn the pattern's shape.
function makeExamples(start, count) {
  return tf.tidy(() => {
    const labels = [], coefficients = [], frequencies = [], phases = [];
    for (let i = start; i < start + count; i++) {
      const label = Math.floor(random(i, 1) * CLASSES);
      labels.push(label);
      coefficients.push(...[[0, 1, 0], [1, 0, 0], [Math.SQRT1_2, Math.SQRT1_2, 0], [0, 0, 1]][label]);
      frequencies.push(0.3 + random(i, 2) * 0.5);
      phases.push(random(i, 3) * 2 * Math.PI);
    }
    const coords = tf.range(0, SIZE).sub((SIZE - 1) / 2);
    const x = coords.reshape([1, 1, SIZE]);
    const y = coords.reshape([1, SIZE, 1]);
    const r = x.square().add(y.square()).sqrt();
    const [a, b, c] = tf.split(tf.tensor(coefficients, [count, 3]), 3, 1).map(t => t.reshape([count, 1, 1]));
    const position = a.mul(x).add(b.mul(y)).add(c.mul(r));
    const images = position.mul(tf.tensor(frequencies, [count, 1, 1]))
      .add(tf.tensor(phases, [count, 1, 1])).sin();
    return {
      xs: images.expandDims(3),
      ys: tf.oneHot(tf.tensor1d(labels, 'int32'), CLASSES),
    };
  });
}

export function createTask() {
  const conv = (filters, seed, extra = {}) => tf.layers.conv2d({
    filters, kernelSize: 3, padding: 'same', activation: 'relu',
    kernelInitializer: tf.initializers.heNormal({ seed }), ...extra,
  });
  const model = tf.sequential({
    layers: [
      conv(8, 1, { inputShape: [SIZE, SIZE, 1] }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      conv(16, 2),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      conv(32, 3),
      tf.layers.globalAveragePooling2d({}),
      tf.layers.dense({ units: CLASSES, kernelInitializer: tf.initializers.glorotUniform({ seed: 4 }) }),
    ],
  });
  const loss = ({ xs, ys }) => tf.losses.softmaxCrossEntropy(ys, model.apply(xs));
  let evalSet;
  return {
    id: taskId,
    variables: model.trainableWeights.map(weight => weight.val),
    // Global example indices make the total batch identical regardless of worker count.
    makeBatch: ({ step, offset, count }) => makeExamples(step * 100_000 + offset, count),
    loss,
    async evaluate() {
      evalSet ??= makeExamples(EVAL_START, EVAL_COUNT);
      const value = tf.tidy(() => loss(evalSet));
      try { return (await value.data())[0]; }
      finally { value.dispose(); }
    },
    dispose() {
      tf.dispose(evalSet);
      model.dispose();
    },
  };
}
