import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as tf from '@tensorflow/tfjs';
import { createTask } from '../src/linear-task.js';
import * as imageTask from '../src/image-task.js';
import { decode } from '../src/fashion-task.js';
import { SharedTrainer, averageUpdates, timingSummary } from '../src/shared-trainer.js';
import { assignValues, createOptimizer, packValues, readValues, trainLocally, unpackValues } from '../src/tensors.js';

await tf.setBackend('cpu');

function closeTo(actual, expected, tolerance = 1e-5) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}
function sameWeights(a, b) {
  a.forEach((values, tensor) => values.forEach((value, i) => closeTo(value, b[tensor][i])));
}

// Runs the coordinator against in-process workers that behave like the worker page.
async function simulate(workerCount, { rounds = 40, batchSize = 17, localSteps = 1, optimizer = 'sgd', learningRate = 0.1 } = {}) {
  const task = createTask();
  const workers = Array.from({ length: workerCount }, () => ({ task: createTask() }));
  const trainer = new SharedTrainer(task, {
    send(message) {
      const worker = workers[Number(message.to)];
      void (async () => {
        if (worker.runId !== message.runId) {
          worker.optimizer?.dispose();
          Object.assign(worker, { runId: message.runId, optimizer: createOptimizer(message.optimizer, message.learningRate) });
        }
        const weights = unpackValues(message.weights, worker.task.variables);
        const result = await trainLocally(worker.task, worker.optimizer, weights, message);
        trainer.receive({
          type: 'update', workerId: message.to, runId: message.runId, round: message.round,
          count: message.count * message.localSteps, ...result, delta: packValues(result.delta),
        });
      })().catch(error => trainer.receive({ ...message, type: 'worker-error', workerId: message.to, error: error.message }));
    },
  });
  try {
    const run = await trainer.train({
      workerIds: workers.map((_, i) => String(i)), rounds, batchSize, localSteps, optimizer, learningRate,
    });
    return { run, weights: await readValues(task.variables) };
  } finally {
    task.dispose();
    workers.forEach(worker => { worker.task.dispose(); worker.optimizer?.dispose(); });
  }
}

test('image task gives identical batches however they are split, and trains to finite values', async () => {
  const before = tf.memory().numTensors;
  const task = imageTask.createTask();
  const optimizer = createOptimizer('sgd', 0.1);
  try {
    assert.equal(task.variables.reduce((sum, variable) => sum + variable.size, 0), 6020);
    const whole = task.makeBatch({ step: 3, offset: 0, count: 6 });
    const parts = [task.makeBatch({ step: 3, offset: 0, count: 2 }), task.makeBatch({ step: 3, offset: 2, count: 4 })];
    for (const key of ['xs', 'ys']) {
      const joined = tf.concat(parts.map(part => part[key]));
      assert.deepEqual(await joined.data(), await whole[key].data());
      joined.dispose();
    }
    tf.dispose([whole, parts]);
    const weights = await readValues(task.variables);
    const update = await trainLocally(task, optimizer, weights, { firstStep: 1, localSteps: 2, offset: 0, count: 4 });
    assert.ok(update.delta.flat().every(Number.isFinite));
    assert.ok(Number.isFinite(await task.evaluate()));
  } finally { task.dispose(); optimizer.dispose(); }
  assert.equal(tf.memory().numTensors, before);
});

test('Fashion-MNIST files decode into images with valid labels', async () => {
  for (const [file, count] of [['fashion-train.bin', 10_000], ['fashion-test.bin', 1_000]]) {
    const data = decode(new Uint8Array(await readFile(`public/data/${file}`)));
    assert.equal(data.count, count);
    assert.equal(data.pixels.length, count * 784);
    assert.ok(data.labels.every(label => label < 10));
  }
  assert.throws(() => decode(new Uint8Array(100)), /file size/);
});

test('timing summary skips warm-up and splits each round into compute, network, and coordinator time', () => {
  const point = (round, elapsedMs, waitMs, computeMs) => ({ round, elapsedMs, waitMs, updates: computeMs.map(ms => ({ computeMs: ms })) });
  assert.equal(timingSummary([point(0, 0, 0, []), point(1, 900, 800, [700])]), null);
  assert.deepEqual(timingSummary([
    point(0, 0, 0, []),
    point(1, 900, 800, [700]),
    point(2, 1000, 80, [50, 60]),
    point(3, 1120, 100, [70, 40]),
  ]), { rounds: 2, roundMs: 110, computeMs: 65, networkMs: 25, coordinatorMs: 20 });
});

test('updates are averaged by example count and must be finite and match the model', () => {
  const task = createTask();
  try {
    assert.deepEqual(averageUpdates([
      { count: 1, delta: [[1], [-1]] },
      { count: 3, delta: [[5], [3]] },
    ], task.variables), [[4], [2]]);
    assert.throws(() => averageUpdates([{ count: 1, delta: [[NaN], [0]] }], task.variables), /finite/);
    assert.throws(() => averageUpdates([{ count: 0, delta: [[1], [0]] }], task.variables), /count/);
    assert.throws(() => assignValues(task.variables, [[1], [1, 2]]), /match/);
  } finally { task.dispose(); }
});

test('packed weights round-trip exactly, are ~4x smaller than JSON, and reject the wrong size', async () => {
  const task = imageTask.createTask();
  try {
    const weights = await readValues(task.variables);
    const packed = packValues(weights);
    assert.deepEqual(unpackValues(packed, task.variables), weights);
    const ratio = JSON.stringify(weights).length / packed.length;
    assert.ok(ratio > 3, `only ${ratio.toFixed(1)}x smaller`);
    assert.throws(() => unpackValues(packed.slice(8), task.variables), /do not match/);
    assert.throws(() => unpackValues([[1]], task.variables), /do not match/);
  } finally { task.dispose(); }
});

test('local training returns the weight change and does not leak tensors', async () => {
  const before = tf.memory().numTensors;
  const task = createTask();
  const optimizer = createOptimizer('adam', 0.01);
  const weights = [[0.5], [-0.5]];
  let afterFirst;
  for (let round = 1; round <= 3; round++) {
    const update = await trainLocally(task, optimizer, weights, { firstStep: round, localSteps: 3, offset: 0, count: 8 });
    assert.ok(Number.isFinite(update.loss));
    const trained = await readValues(task.variables);
    sameWeights(update.delta.map((values, i) => values.map((value, j) => weights[i][j] + value)), trained);
    afterFirst ??= tf.memory().numTensors; // Adam allocates its state once.
    assert.equal(tf.memory().numTensors, afterFirst);
  }
  task.dispose();
  optimizer.dispose();
  assert.equal(tf.memory().numTensors, before);
});

test('one step on three workers equals one SGD step on their combined batch', async () => {
  const task = createTask();
  const optimizer = tf.train.sgd(0.1);
  try {
    const batch = task.makeBatch({ step: 1, offset: 0, count: 3 * 17 });
    optimizer.minimize(() => task.loss(batch));
    tf.dispose(batch);
    const distributed = await simulate(3, { rounds: 1 });
    sameWeights(await readValues(task.variables), distributed.weights);
  } finally {
    task.dispose();
    optimizer.dispose();
  }
});

test('with one local step, three workers learn exactly like one worker on the same data', async () => {
  const before = tf.memory().numTensors;
  const one = await simulate(1, { batchSize: 3 * 17 });
  const three = await simulate(3);
  assert.equal(three.run.outcome, 'completed');
  assert.equal(three.run.history.length, 41);
  assert.ok(three.run.history.at(-1).loss < 0.01);
  sameWeights(one.weights, three.weights);
  for (const point of three.run.history.slice(1)) {
    assert.equal(point.updates.reduce((sum, update) => sum + update.count, 0), 3 * 17);
  }
  assert.equal(tf.memory().numTensors, before);
});

test('one worker with K local steps matches K ordinary SGD steps', async () => {
  const task = createTask();
  const optimizer = tf.train.sgd(0.1);
  try {
    for (let step = 1; step <= 10; step++) {
      const batch = task.makeBatch({ step, offset: 0, count: 17 });
      optimizer.minimize(() => task.loss(batch));
      tf.dispose(batch);
    }
    const local = await simulate(1, { rounds: 2, localSteps: 5 });
    sameWeights(await readValues(task.variables), local.weights);
  } finally {
    task.dispose();
    optimizer.dispose();
  }
});

test('three workers learn with Adam and several local steps per round', async () => {
  const { run } = await simulate(3, { rounds: 20, localSteps: 5, optimizer: 'adam', learningRate: 0.05 });
  assert.equal(run.outcome, 'completed', run.reason);
  assert.ok(run.history.at(-1).loss < run.history[0].loss / 10);
  assert.ok(run.history.slice(1).every(point => point.updates.every(update => update.count === 17 * 5)));
});

test('the outer Nesterov step follows the same formula as ml/sim.py (PyTorch)', async () => {
  const task = createTask();
  // Every worker moves each weight by d = -0.1, so the outer gradient is g = 0.1 each round.
  const trainer = new SharedTrainer(task, {
    send(message) {
      trainer.receive({ type: 'update', runId: message.runId, workerId: message.to, round: message.round,
        count: message.count, delta: packValues([[-0.1], [-0.1]]), loss: 1, computeMs: 1 });
    },
  });
  try {
    const run = await trainer.train({ workerIds: ['a', 'b'], rounds: 2, batchSize: 8, localSteps: 1,
      optimizer: 'sgd', learningRate: 0.1, outerLearningRate: 0.5, outerMomentum: 0.9 });
    assert.equal(run.outcome, 'completed', run.reason);
    // PyTorch Nesterov: buf = 0.9 * buf + g; w -= 0.5 * (g + 0.9 * buf).
    // Round 1: buf = 0.1, step 0.5 * 0.19 = 0.095. Round 2: buf = 0.19, step 0.5 * 0.271 = 0.1355.
    sameWeights(await readValues(task.variables), [[-0.2305], [-0.2305]]);
  } finally { task.dispose(); }
});

test('stale, duplicate, and unassigned updates cannot enter a round', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, {
    send(message) {
      const update = {
        type: 'update', runId: message.runId, workerId: message.to, round: message.round,
        count: message.count, delta: packValues([[-0.1], [-0.1]]), loss: 1, computeMs: 1,
      };
      trainer.receive({ ...update, runId: 'old-run' });
      trainer.receive({ ...update, round: message.round - 1 });
      trainer.receive({ ...update, workerId: 'unknown' });
      trainer.receive(update);
      trainer.receive({ ...update, delta: packValues([[999], [999]]) });
    },
  });
  try {
    const run = await trainer.train({ workerIds: ['a', 'b'], rounds: 1, batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 });
    assert.equal(run.outcome, 'completed');
    sameWeights(await readValues(task.variables), [[-0.1], [-0.1]]);
  } finally { task.dispose(); }
});

test('a round with no usable updates fails without changing the model', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, { send() {} });
  try {
    const run = await trainer.train({ workerIds: ['absent'], rounds: 1, batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1, deadlineMs: 5 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /No worker sent a usable update.*deadline/);
    assert.equal(run.history.length, 1);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});

test('workers that miss the deadline or fail are dropped, and the run continues with the rest', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, {
    send(message) {
      const update = { type: 'update', runId: message.runId, workerId: message.to, round: message.round,
        count: message.count, delta: packValues([[-0.1], [-0.1]]), loss: 1, computeMs: 1 };
      if (message.to === 'fast') trainer.receive(update);
      if (message.to === 'broken') trainer.receive({ ...update, type: 'worker-error', error: 'GPU reset' });
      // 'silent' never answers.
    },
  });
  try {
    const run = await trainer.train({ workerIds: ['fast', 'silent', 'broken'], rounds: 3, batchSize: 8, localSteps: 1,
      optimizer: 'sgd', learningRate: 0.1, deadlineMs: 5 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.deepEqual(run.history[1].dropped.map(item => item.workerId).sort(), ['broken', 'silent']);
    assert.match(run.history[1].dropped.find(item => item.workerId === 'silent').reason, /deadline/);
    assert.match(run.history[1].dropped.find(item => item.workerId === 'broken').reason, /GPU reset/);
    assert.ok(run.history.slice(2).every(point => point.updates.length === 1 && point.dropped.length === 0));
    sameWeights(await readValues(task.variables), [[-0.3], [-0.3]]); // Three rounds of the fast worker's update.
  } finally { task.dispose(); }
});

test('stop invalidates in-flight work and a new run can start cleanly', async () => {
  const task = createTask();
  let old;
  let shouldStop = true;
  const trainer = new SharedTrainer(task, {
    send(message) {
      const update = { type: 'update', workerId: 'a', runId: message.runId, round: message.round,
        count: message.count, delta: packValues([[-0.1], [-0.1]]), loss: 1, computeMs: 1 };
      if (shouldStop) { old = update; trainer.stop(); }
      else { trainer.receive(old); trainer.receive(update); }
    },
  });
  try {
    const settings = { workerIds: ['a'], rounds: 1, batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 };
    const stopped = await trainer.train(settings);
    assert.equal(stopped.outcome, 'stopped');
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
    shouldStop = false;
    const completed = await trainer.train(settings);
    assert.equal(completed.outcome, 'completed');
    assert.notEqual(completed.id, stopped.id);
    closeTo((await readValues(task.variables))[0][0], -0.1);
  } finally { task.dispose(); }
});

test('a worker that leaves mid-round is not waited for, and later rounds go on without it', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, {
    send(message) {
      if (message.to === 'leaving') return queueMicrotask(() => trainer.removeWorker('leaving', 'left'));
      trainer.receive({ type: 'update', runId: message.runId, workerId: message.to, round: message.round,
        count: message.count, delta: packValues([[-0.1], [-0.1]]), loss: 1, computeMs: 1 });
    },
  });
  try {
    // A long deadline: the run only finishes quickly if it stops waiting for the leaver.
    const started = Date.now();
    const run = await trainer.train({ workerIds: ['staying', 'leaving'], rounds: 3, batchSize: 8, localSteps: 1,
      optimizer: 'sgd', learningRate: 0.1, deadlineMs: 60_000 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.ok(Date.now() - started < 1000);
    assert.deepEqual(run.history[1].dropped, [{ workerId: 'leaving', reason: 'left' }]);
    assert.deepEqual(run.workerIds, ['staying']);
    assert.ok(run.history.slice(2).every(point => point.updates.length === 1));
  } finally { task.dispose(); }
});

test('a worker that joins mid-run gets the current model from the next round', async () => {
  const task = createTask();
  const firstWeights = {};
  const trainer = new SharedTrainer(task, {
    send(message) {
      firstWeights[message.to] ??= unpackValues(message.weights, task.variables);
      trainer.receive({ type: 'update', runId: message.runId, workerId: message.to, round: message.round,
        count: message.count, delta: packValues([[-0.1], [-0.1]]), loss: 1, computeMs: 1 });
    },
    onProgress(point) {
      if (point.round === 1) trainer.addWorker('late');
      if (point.round === 2) trainer.addWorker('early'); // Already in the run: ignored.
    },
  });
  try {
    const run = await trainer.train({ workerIds: ['early'], rounds: 3, batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.deepEqual(run.history.slice(1).map(point => point.updates.length), [1, 2, 2]);
    sameWeights(firstWeights.late, [[-0.1], [-0.1]]); // The model after round 1.
    assert.deepEqual(run.participants, ['early', 'late']);
  } finally { task.dispose(); }
});

test('the run fails once every worker has left', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, { send: message => queueMicrotask(() => trainer.removeWorker(message.to, 'left')) });
  try {
    const run = await trainer.train({ workerIds: ['a'], rounds: 3, batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /left/);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});

test('a worker with a wrong example count is dropped rather than skewing the average', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, {
    send(message) {
      trainer.receive({ type: 'update', runId: message.runId, workerId: message.to, round: message.round,
        count: message.count, delta: packValues([[1], [1]]), loss: 1, computeMs: 1 });
    },
  });
  try {
    // 3 local steps of 8 examples, but the worker claims only 8.
    const run = await trainer.train({ workerIds: ['a'], rounds: 1, batchSize: 8, localSteps: 3, optimizer: 'sgd', learningRate: 0.1 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /invalid update/);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});
