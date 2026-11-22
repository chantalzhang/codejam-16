import * as tf from '@tensorflow/tfjs';

// Stand-in demo task until the fun one is picked. Data files: ml/export_fashion.py.
export const name = 'Fashion-MNIST (stand-in)';
export const description = 'Stand-in task: a small CNN learns 10 kinds of clothing from 28×28 Fashion-MNIST images.';
// From ml/sim.py: Adam needs far fewer steps than SGD, and syncing every 10 steps was fastest.
export const defaults = { rounds: 100, batchSize: 32, localSteps: 10, optimizer: 'adam', learningRate: 0.01 };
export const taskId = 'fashion-mnist-v1';

const SIZE = 28;
const PIXELS = SIZE * SIZE;
const CLASSES = 10;

// File layout: count × 784 uint8 pixels, then count uint8 labels.
export function decode(bytes) {
  const count = bytes.length / (PIXELS + 1);
  if (!Number.isInteger(count) || count === 0) throw new Error('Unexpected Fashion-MNIST file size.');
  return { count, pixels: bytes.subarray(0, count * PIXELS), labels: bytes.subarray(count * PIXELS) };
}

async function download(file) {
  const response = await fetch(`/data/${file}`);
  if (!response.ok) throw new Error(`Could not load ${file}. Run: uv run ml/export_fashion.py`);
  return decode(new Uint8Array(await response.arrayBuffer()));
}

function toTensors({ pixels, labels }, indices) {
  const xs = new Float32Array(indices.length * PIXELS);
  indices.forEach((index, i) => {
    for (let p = 0; p < PIXELS; p++) xs[i * PIXELS + p] = pixels[index * PIXELS + p] / 255;
  });
  return tf.tidy(() => ({
    xs: tf.tensor4d(xs, [indices.length, SIZE, SIZE, 1]),
    ys: tf.oneHot(tf.tensor1d(indices.map(index => labels[index]), 'int32'), CLASSES),
  }));
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
  // Workers download training images on their first step; the laptop only needs test images.
  let train, test;
  return {
    id: taskId,
    variables: model.trainableWeights.map(weight => weight.val),
    async makeBatch({ step, offset, count }) {
      const data = await (train ??= download('fashion-train.bin'));
      // Global example numbers keep the full batch identical however it's split across workers.
      const indices = Array.from({ length: count }, (_, i) =>
        (Math.imul(step * 100_000 + offset + i, 0x9e3779b1) >>> 0) % data.count);
      return toTensors(data, indices);
    },
    loss: ({ xs, ys }) => tf.losses.softmaxCrossEntropy(ys, model.apply(xs)),
    async evaluate() {
      const { xs, ys } = await (test ??= download('fashion-test.bin')
        .then(data => toTensors(data, Array.from({ length: data.count }, (_, i) => i))));
      const [loss, accuracy] = tf.tidy(() => {
        const logits = model.apply(xs);
        return [tf.losses.softmaxCrossEntropy(ys, logits), logits.argMax(1).equal(ys.argMax(1)).mean()];
      });
      try { return { loss: (await loss.data())[0], accuracy: (await accuracy.data())[0] }; }
      finally { tf.dispose([loss, accuracy]); }
    },
    dispose() {
      test?.then(tf.dispose);
      model.dispose();
    },
  };
}
