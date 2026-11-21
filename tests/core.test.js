import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { createTask } from '../src/task.js';
import { SharedTrainer, splitBatch, averageGradients } from '../src/shared-trainer.js';
import { assignValues, computeGradients, readValues } from '../src/tensors.js';

await tf.setBackend('cpu');

function closeTo(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}

async function simulate(workerCount, steps = 40, batchSize = 17) {
  const task = createTask();
  const workers = Array.from({ length: workerCount }, createTask);
  const trainer = new SharedTrainer(task, {
    send(message) {
      const worker = workers[Number(message.to)];
      void (async () => {
        assignValues(worker.variables, message.weights);
        const result = await computeGradients(worker, message);
        trainer.receive({
          type: 'gradient', workerId: message.to, runId: message.runId,
          step: message.step, count: message.count, ...result,
        });
      })().catch(error => trainer.receive({ ...message, type: 'worker-error', workerId: message.to, error: error.message }));
    },
  });
  try {
    const run = await trainer.train({
      workerIds: workers.map((_, i) => String(i)), steps, batchSize, learningRate: 0.1,
    });
    return { run, weights: await readValues(task.variables) };
  } finally {
    task.dispose();
    workers.forEach(worker => worker.dispose());
  }
}

test('fixed global batches are divided without gaps or repeated examples', () => {
  assert.deepEqual(splitBatch(10, ['a', 'b', 'c']), [
    { workerId: 'a', offset: 0, count: 4 },
    { workerId: 'b', offset: 4, count: 3 },
    { workerId: 'c', offset: 7, count: 3 },
  ]);
  assert.throws(() => splitBatch(1, ['a', 'b']), /batch size/);
  assert.throws(() => splitBatch(10, ['a', 'a']), /distinct/);
  assert.throws(() => splitBatch(10, []), /workers/);
});

test('averaging weights mean gradients by actual assigned example count', () => {
  const task = createTask();
  try {
    assert.deepEqual(averageGradients([
      { count: 1, gradients: [[1], [-1]] },
      { count: 3, gradients: [[5], [3]] },
    ], task.variables), [[4], [2]]);
    assert.throws(() => averageGradients([{ count: 1, gradients: [[NaN], [0]] }], task.variables), /finite/);
    assert.throws(() => averageGradients([{ count: 0, gradients: [[1], [0]] }], task.variables), /count/);
    assert.throws(() => assignValues(task.variables, [[1], [1, 2]]), /match/);
  } finally { task.dispose(); }
});

test('gradient computation does not update worker weights or leak tensors', async () => {
  const before = tf.memory().numTensors;
  const task = createTask();
  const initial = await readValues(task.variables);
  const allocated = tf.memory().numTensors;
  for (let step = 1; step <= 3; step++) {
    const update = await computeGradients(task, { step, offset: 0, count: 8 });
    assert.ok(Number.isFinite(update.loss));
    assert.equal(update.gradients.length, task.variables.length);
  }
  assert.deepEqual(await readValues(task.variables), initial);
  assert.equal(tf.memory().numTensors, allocated);
  task.dispose();
  assert.equal(tf.memory().numTensors, before);
});

test('one and three workers produce the same learned model on the same global batches', async () => {
  const before = tf.memory().numTensors;
  const one = await simulate(1);
  const three = await simulate(3);
  assert.equal(one.run.outcome, 'completed');
  assert.equal(three.run.outcome, 'completed');
  assert.equal(three.run.history.length, 41);
  assert.ok(three.run.history.at(-1).loss < 0.01);
  one.weights.forEach((values, tensor) => values.forEach((value, i) => closeTo(value, three.weights[tensor][i])));
  for (const point of three.run.history.slice(1)) {
    assert.equal(point.updates.reduce((sum, update) => sum + update.count, 0), 17);
  }
  assert.equal(tf.memory().numTensors, before);
});

test('distributed update matches an ordinary SGD update on the full batch', async () => {
  const task = createTask();
  const optimizer = tf.train.sgd(0.1);
  try {
    const batch = task.makeBatch({ step: 1, offset: 0, count: 17 });
    optimizer.minimize(() => task.loss(batch));
    tf.dispose(batch);
    const distributed = await simulate(3, 1);
    const expected = await readValues(task.variables);
    expected.forEach((values, tensor) => values.forEach((value, i) => closeTo(value, distributed.weights[tensor][i])));
  } finally {
    task.dispose();
    optimizer.dispose();
  }
});

test('stale, duplicate, and unassigned updates cannot enter a step', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, {
    send(message) {
      const update = {
        type: 'gradient', runId: message.runId, workerId: message.to, step: message.step,
        count: message.count, gradients: [[1], [1]], loss: 1, computeMs: 1,
      };
      trainer.receive({ ...update, runId: 'old-run' });
      trainer.receive({ ...update, step: message.step - 1 });
      trainer.receive({ ...update, workerId: 'unknown' });
      trainer.receive(update);
      trainer.receive({ ...update, gradients: [[999], [999]] });
    },
  });
  try {
    const run = await trainer.train({ workerIds: ['a', 'b'], steps: 1, batchSize: 8, learningRate: 0.1 });
    assert.equal(run.outcome, 'completed');
    const weights = await readValues(task.variables);
    closeTo(weights[0][0], -0.1);
    closeTo(weights[1][0], -0.1);
  } finally { task.dispose(); }
});

test('a missing worker times out without applying a partial update', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, { send() {} });
  try {
    const run = await trainer.train({ workerIds: ['absent'], steps: 1, batchSize: 8, learningRate: 0.1, timeoutMs: 20 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /timed out/);
    assert.equal(run.history.length, 1);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});

test('stop invalidates in-flight work and a new run can start cleanly', async () => {
  const task = createTask();
  let old;
  let shouldStop = true;
  const trainer = new SharedTrainer(task, {
    send(message) {
      const update = { type: 'gradient', workerId: 'a', runId: message.runId, step: message.step,
        count: message.count, gradients: [[1], [1]], loss: 1, computeMs: 1 };
      if (shouldStop) { old = update; trainer.stop(); }
      else { trainer.receive(old); trainer.receive(update); }
    },
  });
  try {
    const settings = { workerIds: ['a'], steps: 1, batchSize: 8, learningRate: 0.1 };
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

test('invalid worker counts fail rather than changing the effective batch', async () => {
  const task = createTask();
  const trainer = new SharedTrainer(task, {
    send(message) {
      trainer.receive({ type: 'gradient', runId: message.runId, workerId: message.to, step: message.step,
        count: message.count + 1, gradients: [[1], [1]], loss: 1, computeMs: 1 });
    },
  });
  try {
    const run = await trainer.train({ workerIds: ['a'], steps: 1, batchSize: 8, learningRate: 0.1 });
    assert.equal(run.outcome, 'failed');
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});
