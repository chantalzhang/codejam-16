import * as tf from '@tensorflow/tfjs';

// Image classification tasks (Fashion-MNIST, Guess the game, Brainrot): a small CNN, with
// data files from ml/export_*.py. Display images can be bigger than the model input so
// people can see them; the model averages them down.

// File layout: count images of size × size × channels uint8 pixels, then count uint8 labels.
export function decode(bytes, size, channels) {
  const count = bytes.length / (size * size * channels + 1);
  if (!Number.isInteger(count) || count === 0) throw new Error('Unexpected image file size.');
  return { count, size, channels, pixels: bytes.subarray(0, bytes.length - count), labels: bytes.subarray(bytes.length - count) };
}

export function imageTask({ taskId, name, description, file, classes, size, channels, displaySize = size, slices, defaults }) {
  async function download(suffix, imageSize = size) {
    const response = await fetch(`/data/${file}-${suffix}.bin`);
    if (!response.ok) throw new Error(`Could not load ${file}-${suffix}.bin. Run its ml/export_*.py script.`);
    return decode(new Uint8Array(await response.arrayBuffer()), imageSize, channels);
  }
  const all = ({ count }) => Array.from({ length: count }, (_, i) => i);

  function toTensors({ size: imageSize, pixels, labels }, indices) {
    const length = imageSize * imageSize * channels;
    const xs = new Float32Array(indices.length * length);
    indices.forEach((index, i) => {
      for (let p = 0; p < length; p++) xs[i * length + p] = pixels[index * length + p] / 255;
    });
    return tf.tidy(() => {
      const images = tf.tensor4d(xs, [indices.length, imageSize, imageSize, channels]);
      const scale = imageSize / size;
      return {
        xs: scale === 1 ? images : tf.avgPool(images, scale, scale, 'valid'),
        ys: tf.oneHot(tf.tensor1d(indices.map(index => labels[index]), 'int32'), classes.length),
      };
    });
  }

  // A few fixed held-out images shown on screen, separate from training and from the
  // accuracy set. Small, so phones can show them too.
  let displayFile;
  async function displayExamples() {
    const { count, pixels, labels } = await (displayFile ??= download('display', displaySize));
    const length = displaySize * displaySize * channels;
    return Array.from({ length: count }, (_, i) => ({
      label: labels[i],
      image: { width: displaySize, height: displaySize, channels, pixels: pixels.subarray(i * length, (i + 1) * length) },
    }));
  }

  // Workers pick a slice by join number, so small groups get different slices and a phone
  // keeps the same one across runs. Without a join number (e.g. the laptop), pick at random.
  const sliceFor = joinNumber => Number.isInteger(joinNumber) ? joinNumber % slices : Math.floor(Math.random() * slices);

  function createTask({ joinNumber } = {}) {
    const conv = (filters, seed, extra = {}) => tf.layers.conv2d({
      filters, kernelSize: 3, padding: 'same', activation: 'relu',
      kernelInitializer: tf.initializers.heNormal({ seed }), ...extra,
    });
    const model = tf.sequential({
      layers: [
        conv(8, 1, { inputShape: [size, size, channels] }),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        conv(16, 2),
        tf.layers.maxPooling2d({ poolSize: 2 }),
        conv(32, 3),
        tf.layers.globalAveragePooling2d({}),
        tf.layers.dense({ units: classes.length, kernelInitializer: tf.initializers.glorotUniform({ seed: 4 }) }),
      ],
    });
    // Each worker downloads its slice on its first step; the laptop only needs the test images.
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
      // The model's class probabilities on the display images, for the big screen.
      async predictDisplay() {
        const { xs } = await (display ??= (displayFile ??= download('display', displaySize))
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
