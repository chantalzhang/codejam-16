import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as tf from '@tensorflow/tfjs';
import { createTask } from '../src/linear-task.js';
import * as imageTask from '../src/image-task.js';
import { classes, decode, sliceFor } from '../src/fashion-task.js';
import { decode as decodeColor } from '../src/color-image-task.js';
import games from '../src/games-task.js';
import brainrot from '../src/brainrot-task.js';
import { SharedTrainer, timingSummary } from '../src/shared-trainer.js';
import { createOptimizer, packValues, readValues, trainLocally, unpackValues } from '../src/tensors.js';
import { createRelay } from '../server/relay-core.js';

await tf.setBackend('cpu');

function closeTo(actual, expected, tolerance = 1e-5) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
}
function sameWeights(a, b) {
  a.forEach((values, tensor) => values.forEach((value, i) => closeTo(value, b[tensor][i])));
}

// Wires a trainer to the real relay logic in-process, the way the laptop page and phones
// connect over WebSockets. `behaviors[name](round, reply, worker)` decides how each worker
// answers a round. Like the laptop page, workers that disconnect leave the run.
function swarm(task, behaviors, options = {}) {
  const relay = createRelay();
  let trainer;
  const laptop = relay.connect(text => {
    const message = JSON.parse(text);
    if (message.type === 'presence') {
      for (const id of trainer.active?.workerIds ?? []) {
        if (!message.workers.some(worker => worker.id === id)) trainer.removeWorker(id);
      }
    }
    trainer.receive(message);
  });
  trainer = new SharedTrainer(task, { ...options, send: message => laptop.receive(JSON.stringify(message)) });
  laptop.receive(JSON.stringify({ type: 'join', role: 'coordinator' }));
  const ids = {};
  const connections = {};
  for (const [name, behavior] of Object.entries(behaviors)) {
    const worker = { name };
    const connection = relay.connect(text => {
      const message = JSON.parse(text);
      if (message.type === 'welcome') ids[name] = message.id;
      if (message.type !== 'round') return;
      const reply = answer => connection.receive(JSON.stringify({ runId: message.runId, round: message.round, ...answer }));
      behavior(message, reply, worker);
    });
    connection.receive(JSON.stringify({ type: 'join', role: 'worker' }));
    connection.receive(JSON.stringify({ type: 'ready', backend: 'cpu', tasks: [task.id] }));
    connections[name] = connection;
  }
  const names = id => Object.keys(ids).find(name => ids[name] === id);
  const settings = { batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 };
  return {
    trainer, ids, connections, names,
    train: overrides => trainer.train({ workerIds: Object.values(ids), rounds: 1, ...settings, ...overrides }),
  };
}

// Answers every round with the same fake change, the right size for the linear task.
const steady = (delta = [[-0.1], [-0.1]]) => (round, reply) =>
  reply({ type: 'update', count: round.count * round.localSteps, delta: packValues(delta), loss: 1, computeMs: 1 });

// Trains for real, like the worker page: own task copy, own optimizer for each run.
const realWorker = (round, reply, worker) => void (async () => {
  worker.task ??= createTask();
  if (worker.runId !== round.runId) {
    worker.optimizer?.dispose();
    Object.assign(worker, { runId: round.runId, optimizer: createOptimizer(round.optimizer, round.learningRate) });
  }
  const weights = unpackValues(round.weights, worker.task.variables);
  const result = await trainLocally(worker.task, worker.optimizer, weights, round);
  reply({ type: 'update', count: round.count * round.localSteps, ...result, delta: packValues(result.delta) });
})();

async function simulate(workerCount, { rounds = 40, batchSize = 17, localSteps = 1, optimizer = 'sgd', learningRate = 0.1 } = {}) {
  const task = createTask();
  const workers = [];
  const behaviors = Object.fromEntries(Array.from({ length: workerCount }, (_, i) => [`w${i}`, (round, reply, worker) => {
    if (!workers.includes(worker)) workers.push(worker);
    realWorker(round, reply, worker);
  }]));
  try {
    const run = await swarm(task, behaviors).train({ rounds, batchSize, localSteps, optimizer, learningRate });
    return { run, weights: await readValues(task.variables) };
  } finally {
    task.dispose();
    workers.forEach(worker => { worker.task?.dispose(); worker.optimizer?.dispose(); });
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
  const slices = [0, 1, 2, 3, 4].map(i => [`fashion-train-${i}.bin`, 2_000]);
  for (const [file, count] of [...slices, ['fashion-test.bin', 1_000]]) {
    const data = decode(new Uint8Array(await readFile(`public/data/${file}`)));
    assert.equal(data.count, count);
    assert.equal(data.pixels.length, count * 784);
    assert.ok(data.labels.every(label => label < 10));
  }
  assert.throws(() => decode(new Uint8Array(100)), /file size/);
  // Workers take slices in join order; the 6th worker wraps back to slice 0.
  assert.deepEqual([0, 1, 2, 3, 4, 5, 6].map(sliceFor), [0, 1, 2, 3, 4, 0, 1]);
  // Four held-out images to show on screen: ankle boot, trouser, bag, shirt.
  const display = decode(new Uint8Array(await readFile('public/data/fashion-display.bin')));
  assert.deepEqual(Array.from(display.labels, label => classes[label]), ['Ankle boot', 'Trouser', 'Bag', 'Shirt']);
});

test('Guess the game files decode, and every training slice has the same number of each game', async () => {
  for (let slice = 0; slice < 10; slice++) {
    const data = decodeColor(new Uint8Array(await readFile(`public/data/games-train-${slice}.bin`)));
    const counts = games.classes.map((_, c) => data.labels.filter(label => label === c).length);
    assert.deepEqual(counts, games.classes.map(() => 111));
  }
  const test = decodeColor(new Uint8Array(await readFile('public/data/games-test.bin')));
  assert.ok(test.count > 800 && test.labels.every(label => label < games.classes.length));
  const display = decodeColor(new Uint8Array(await readFile('public/data/games-display.bin')), 64);
  assert.deepEqual(Array.from(display.labels, label => games.classes[label]), ['Minecraft', 'Fortnite', 'Among Us', 'CS2']);
  assert.deepEqual([0, 9, 10].map(games.sliceFor), [0, 9, 0]);
});

test('brainrot files decode into one balanced slice, held-out test images, and 4 display images', async () => {
  const train = decodeColor(new Uint8Array(await readFile('public/data/brainrot-train-0.bin')));
  assert.deepEqual(brainrot.classes.map((_, c) => train.labels.filter(label => label === c).length), [122, 122, 122, 122, 122]);
  const test = decodeColor(new Uint8Array(await readFile('public/data/brainrot-test.bin')));
  assert.ok(test.count > 150 && test.labels.every(label => label < brainrot.classes.length));
  const display = decodeColor(new Uint8Array(await readFile('public/data/brainrot-display.bin')), 64);
  assert.equal(display.count, 4);
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
  try {
    // Every worker moves each weight by d = -0.1, so the outer gradient is g = 0.1 each round.
    const run = await swarm(task, { a: steady(), b: steady() }).train({ rounds: 2, outerLearningRate: 0.5, outerMomentum: 0.9 });
    assert.equal(run.outcome, 'completed', run.reason);
    // PyTorch Nesterov: buf = 0.9 * buf + g; w -= 0.5 * (g + 0.9 * buf).
    // Round 1: buf = 0.1, step 0.5 * 0.19 = 0.095. Round 2: buf = 0.19, step 0.5 * 0.271 = 0.1355.
    sameWeights(await readValues(task.variables), [[-0.2305], [-0.2305]]);
  } finally { task.dispose(); }
});

test('a run stops as soon as it reaches the target accuracy, and says when', async () => {
  const task = createTask();
  task.evaluate = async () => ({ loss: 1, accuracy: [0.1, 0.4, 0.7, 0.9][(evaluations++)] ?? 0.9 }); // Fake climb.
  let evaluations = 0;
  try {
    const run = await swarm(task, { a: steady(), b: steady() }).train({ rounds: 10, targetAccuracy: 0.7 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.equal(run.history.length, 3); // Rounds 0, 1, 2: stopped at 70%, not after 10 rounds.
    assert.deepEqual([run.reachedTarget.round, run.reachedTarget.accuracy, run.reachedTarget.workers], [2, 0.7, 2]);
    assert.ok(run.reachedTarget.elapsedMs >= 0);
  } finally { task.dispose(); }
});

test('each round records the model probabilities on the display images, when the task has them', async () => {
  const task = createTask();
  task.predictDisplay = async () => [[0.25, 0.75]];
  try {
    const run = await swarm(task, { a: steady() }).train({ rounds: 2 });
    assert.ok(run.history.every(point => point.display?.[0][1] === 0.75));
  } finally { task.dispose(); }
});

test('a run stops once its time limit has passed', async () => {
  const task = createTask();
  try {
    const run = await swarm(task, { a: steady() }).train({ rounds: 100_000, timeLimitMs: 20 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.ok(run.timeUp && run.timeUp.elapsedMs >= 20 && run.history.length < 100_001);
    assert.equal(run.reachedTarget, undefined);
  } finally { task.dispose(); }
});

test('a round with no usable updates fails without changing the model', async () => {
  const task = createTask();
  try {
    const run = await swarm(task, { silent: () => {} }).train({ deadlineMs: 5 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /No worker sent a usable update.*deadline/);
    assert.equal(run.history.length, 1);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});

test('workers that miss the deadline or fail are dropped, and the run continues with the rest', async () => {
  const task = createTask();
  try {
    const swarmed = swarm(task, {
      fast: steady(),
      silent: () => {},
      broken: (round, reply) => reply({ type: 'worker-error', error: 'GPU reset' }),
    });
    const run = await swarmed.train({ rounds: 3, deadlineMs: 5 });
    assert.equal(run.outcome, 'completed', run.reason);
    const dropped = Object.fromEntries(run.history[1].dropped.map(({ workerId, reason }) => [swarmed.names(workerId), reason]));
    assert.deepEqual(Object.keys(dropped).sort(), ['broken', 'silent']);
    assert.match(dropped.silent, /deadline/);
    assert.match(dropped.broken, /GPU reset/);
    assert.ok(run.history.slice(2).every(point => point.updates.length === 1 && point.dropped.length === 0));
    sameWeights(await readValues(task.variables), [[-0.3], [-0.3]]); // Three rounds of the fast worker's update.
  } finally { task.dispose(); }
});

test('stop invalidates in-flight work and a new run can start cleanly', async () => {
  const task = createTask();
  let stopNext = true;
  try {
    const swarmed = swarm(task, {
      a: (round, reply) => (stopNext ? swarmed.trainer.stop() : steady()(round, reply)),
    });
    const stopped = await swarmed.train();
    assert.equal(stopped.outcome, 'stopped');
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
    stopNext = false;
    const completed = await swarmed.train();
    assert.equal(completed.outcome, 'completed', completed.reason);
    assert.notEqual(completed.id, stopped.id);
    closeTo((await readValues(task.variables))[0][0], -0.1);
  } finally { task.dispose(); }
});

test('a worker that leaves mid-round is not waited for, and later rounds go on without it', async () => {
  const task = createTask();
  try {
    const swarmed = swarm(task, {
      staying: steady(),
      leaving: () => queueMicrotask(() => swarmed.connections.leaving.disconnect()),
    });
    // A long deadline: the run only finishes quickly if it stops waiting for the leaver.
    const started = Date.now();
    const run = await swarmed.train({ rounds: 3, deadlineMs: 60_000 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.ok(Date.now() - started < 1000);
    assert.deepEqual(run.history[1].dropped, [{ workerId: swarmed.ids.leaving, reason: 'left' }]);
    assert.deepEqual(run.workerIds, [swarmed.ids.staying]);
    assert.ok(run.history.slice(2).every(point => point.updates.length === 1));
  } finally { task.dispose(); }
});

test('a worker that joins mid-run gets the current model from the next round', async () => {
  const task = createTask();
  let lateWeights;
  try {
    const swarmed = swarm(task, {
      early: steady(),
      late: (round, reply) => { lateWeights ??= unpackValues(round.weights, task.variables); steady()(round, reply); },
    }, {
      onProgress(point) {
        if (point.round === 1) swarmed.trainer.addWorker(swarmed.ids.late);
        if (point.round === 2) swarmed.trainer.addWorker(swarmed.ids.early); // Already in the run: ignored.
      },
    });
    const run = await swarmed.trainer.train({ workerIds: [swarmed.ids.early], rounds: 3, batchSize: 8, localSteps: 1, optimizer: 'sgd', learningRate: 0.1 });
    assert.equal(run.outcome, 'completed', run.reason);
    assert.deepEqual(run.history.slice(1).map(point => point.updates.length), [1, 2, 2]);
    sameWeights(lateWeights, [[-0.1], [-0.1]]); // The model after round 1.
    assert.deepEqual(run.participants, [swarmed.ids.early, swarmed.ids.late]);
  } finally { task.dispose(); }
});

test('with more workers than the per-round cap, workers take turns', async () => {
  const task = createTask();
  try {
    const swarmed = swarm(task, Object.fromEntries(['a', 'b', 'c', 'd', 'e'].map(name => [name, steady()])));
    const run = await swarmed.train({ rounds: 5, maxWorkersPerRound: 2 });
    assert.equal(run.outcome, 'completed', run.reason);
    const picks = run.history.slice(1).map(point => point.updates.map(update => swarmed.names(update.workerId)).sort().join(''));
    // Longest-waiting first, ties in join order: everyone trains within every 3 rounds.
    assert.deepEqual(picks, ['ab', 'cd', 'ae', 'bc', 'ad']);
    assert.ok(run.history.slice(1).every(point => point.dropped.length === 0), 'Resting workers are not dropped');
  } finally { task.dispose(); }
});

test('the run fails once every worker has left', async () => {
  const task = createTask();
  try {
    const swarmed = swarm(task, { a: () => queueMicrotask(() => swarmed.connections.a.disconnect()) });
    const run = await swarmed.train({ rounds: 3 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /left/);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});

test('a worker with a wrong example count is dropped rather than skewing the average', async () => {
  const task = createTask();
  try {
    // 3 local steps of 8 examples, but the worker claims only 8.
    const run = await swarm(task, {
      a: (round, reply) => reply({ type: 'update', count: round.count, delta: packValues([[1], [1]]), loss: 1, computeMs: 1 }),
    }).train({ localSteps: 3 });
    assert.equal(run.outcome, 'failed');
    assert.match(run.reason, /invalid update/);
    assert.deepEqual(await readValues(task.variables), [[0], [0]]);
  } finally { task.dispose(); }
});
