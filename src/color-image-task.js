import * as tf from '@tensorflow/tfjs';

// Color image classification tasks (Guess the game, Brainrot). Same model and training as
// Fashion-MNIST, with 32x32 RGB input. Data files: ml/prepare_*.py, then ml/export_images.py.
const SIZE = 32;
const DISPLAY_SIZE = 64; // Display images are bigger so people can see them.

// File layout: count images of size × size × 3 uint8 pixels, then count uint8 labels.
export function decode(bytes, size = SIZE) {
  const pixels = size * size * 3;
  const count = bytes.length / (pixels + 1);
  if (!Number.isInteger(count) || count === 0) throw new Error('Unexpected image file size.');
  return { count, size, pixels: bytes.subarray(0, count * pixels), labels: bytes.subarray(count * pixels) };
}

export function colorImageTask({ taskId, name, description, file, classes, slices, defaults }) {
  async function download(suffix, size) {
    const response = await fetch(`/data/${file}-${suffix}.bin`);
    if (!response.ok) throw new Error(`Could not load ${file}-${suffix}.bin. Run: uv run ml/export_images.py ${file}`);
    return decode(new Uint8Array(await response.arrayBuffer()), size);
  }

  // Display images are averaged down to the model's input size, the same way the training
  // images were made from the originals.
  function toTensors({ size, pixels, labels }, indices) {
    const length = size * size * 3;
    const xs = new Float32Array(indices.length * length);
    indices.forEach((index, i) => {
      for (let p = 0; p < length; p++) xs[i * length + p] = pixels[index * length + p] / 255;
    });
    return tf.tidy(() => {
      const images = tf.tensor4d(xs, [indices.length, size, size, 3]);
      return {
        xs: size === SIZE ? images : tf.avgPool(images, size / SIZE, size / SIZE, 'valid'),
        ys: tf.oneHot(tf.tensor1d(indices.map(index => labels[index]), 'int32'), classes.length),
      };
    });
  }
  const all = ({ count }) => Array.from({ length: count }, (_, i) => i);

  // A few fixed held-out images shown on screen, separate from training and from the
  // accuracy set. Small, so phones can show them too.
  let displayFile;
  async function displayExamples() {
    const { count, pixels, labels } = await (displayFile ??= download('display', DISPLAY_SIZE));
    const length = DISPLAY_SIZE * DISPLAY_SIZE * 3;
    return Array.from({ length: count }, (_, i) => ({
      label: labels[i],
      image: { width: DISPLAY_SIZE, height: DISPLAY_SIZE, channels: 3, pixels: pixels.subarray(i * length, (i + 1) * length) },
    }));
  }

  // Workers pick a slice by join number, like Fashion-MNIST. Slices are class-balanced.
  const sliceFor = joinNumber => Number.isInteger(joinNumber) ? joinNumber % slices : Math.floor(Math.random() * slices);

  function createTask({ joinNumber } = {}) {
    const conv = (filters, seed, extra = {}) => tf.layers.conv2d({
      filters, kernelSize: 3, padding: 'same', activation: 'relu',
      kernelInitializer: tf.initializers.heNormal({ seed }), ...extra,
    });
    const model = tf.sequential({
      layers: [
        conv(8, 1, { inputShape: [SIZE, SIZE, 3] }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        conv(16, 2),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        conv(32, 3),
        tf.layers.globalAveragePooling2d({}),
        tf.layers.dense({ units: classes.length, kernelInitializer: tf.initializers.glorotUniform({ seed: 4 }) }),
      ],
    });
    const slice = sliceFor(joinNumber);
    let train, test, display;
    return {
      id: taskId,
      variables: model.trainableWeights.map(weight => weight.val),
      async makeBatch({ step, offset, count }) {
        const data = await (train ??= download(`train-${slice}`));
        // Global example numbers keep the full batch identical however it's split across workers.
        const indices = Array.from({ length: count }, (_, i) =>
          (Math.imul(step * 100_000 + offset + i, 0x9e3779b1) >>> 0) % data.count);
        return toTensors(data, indices);
      },
      loss: ({ xs, ys }) => tf.losses.softmaxCrossEntropy(ys, model.apply(xs)),
      async evaluate() {
        const { xs, ys } = await (test ??= download('test').then(data => toTensors(data, all(data))));
        const [loss, accuracy] = tf.tidy(() => {
          const logits = model.apply(xs);
          return [tf.losses.softmaxCrossEntropy(ys, logits), logits.argMax(1).equal(ys.argMax(1)).mean()];
        });
        try { return { loss: (await loss.data())[0], accuracy: (await accuracy.data())[0] }; }
        finally { tf.dispose([loss, accuracy]); }
      },
      async predictDisplay() {
        const { xs } = await (display ??= (displayFile ??= download('display', DISPLAY_SIZE))
          .then(data => toTensors(data, all(data))));
        const probabilities = tf.tidy(() => tf.softmax(model.apply(xs)));
        try { return await probabilities.array(); }
        finally { probabilities.dispose(); }
      },
      dispose() {
        test?.then(tf.dispose);
        display?.then(tf.dispose);
        model.dispose();
      },
    };
  }

  return { taskId, name, description, defaults, classes, displayExamples, sliceFor, createTask };
}
