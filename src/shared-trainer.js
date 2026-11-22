import * as tf from '@tensorflow/tfjs';
import { packValues, readValues, unpackValues, validateValues } from './tensors.js';

// Average time per round, skipping round 1 (GPU warm-up). Network time is approximate:
// the wait for all updates minus the slowest worker's compute time.
export function timingSummary(history) {
  const rounds = history.slice(2).map((point, i) => {
    const roundMs = point.elapsedMs - history[i + 1].elapsedMs;
    const computeMs = Math.max(...point.updates.map(update => update.computeMs));
    return { roundMs, computeMs, networkMs: point.waitMs - computeMs, coordinatorMs: roundMs - point.waitMs };
  });
  if (!rounds.length) return null;
  const mean = key => rounds.reduce((sum, round) => sum + round[key], 0) / rounds.length;
  return {
    rounds: rounds.length, roundMs: mean('roundMs'), computeMs: mean('computeMs'),
    networkMs: mean('networkMs'), coordinatorMs: mean('coordinatorMs'),
  };
}

// Local SGD: each round, every worker trains its own copy for `localSteps` steps on
// `batchSize` examples per step. The laptop then treats minus the average change as a
// gradient for an outer optimizer (DiLoCo-style); outer lr 1 with no momentum is plain
// averaging. Nesterov momentum was ~2x faster in ml/sim.py.
export class SharedTrainer {
  constructor(task, { send, onProgress = () => {} }) {
    this.task = task;
    this.send = send;
    this.onProgress = onProgress;
    this.active = null;
    this.pending = null;
  }

  // The relay sends one result per round: the average change and who was dropped.
  receive(message) {
    const pending = this.pending;
    if (message.type !== 'round-result' || !pending || this.active?.cancelReason ||
        message.runId !== this.active?.id || message.round !== pending.round) return;
    if (!message.delta) {
      const reasons = message.dropped.map(({ reason }) => reason).join('; ');
      pending.reject(new Error(`No worker sent a usable update (${reasons}).`));
      return;
    }
    try {
      const average = unpackValues(message.delta, this.task.variables);
      validateValues(average, this.task.variables);
      pending.resolve({ average, updates: message.updates, dropped: message.dropped });
    } catch (error) {
      pending.reject(error);
    }
  }

  // A worker connected mid-run: it gets the current model and trains from the next round.
  // Workers already in this run (including dropped ones) are not re-added.
  addWorker(workerId) {
    const run = this.active;
    if (!run || run.cancelReason || run.participants.includes(workerId)) return;
    run.participants.push(workerId);
    run.workerIds.push(workerId);
  }

  // A worker left: leave it out of later rounds. The relay stops waiting for it this round.
  removeWorker(workerId) {
    const run = this.active;
    if (run) run.workerIds = run.workerIds.filter(id => id !== workerId);
  }

  stop(reason = 'Stopped by coordinator') {
    if (!this.active) return;
    this.active.cancelReason = reason;
    this.pending?.reject(new Error(reason));
  }

  // The relay fans the round out, waits for every worker or the deadline, and replies once.
  async requestUpdates(run, round, assignments, weights, deadlineMs) {
    const pending = { round };
    let timer;
    try {
      return await new Promise((resolve, reject) => {
        Object.assign(pending, { resolve, reject });
        this.pending = pending;
        timer = setTimeout(() => reject(new Error(`The relay never finished round ${round}.`)), deadlineMs + 15_000);
        this.send({
          type: 'round', taskId: this.task.id, runId: run.id, round, deadlineMs,
          firstStep: (round - 1) * run.localSteps + 1, localSteps: run.localSteps,
          optimizer: run.optimizer, learningRate: run.learningRate, weights: packValues(weights), assignments,
        });
      });
    } finally {
      clearTimeout(timer);
      if (this.pending === pending) this.pending = null;
    }
  }

  async train({
    workerIds, rounds, batchSize, localSteps, optimizer, learningRate,
    outerLearningRate = 1, outerMomentum = 0, deadlineMs = 10_000, maxWorkersPerRound = 500,
  }) {
    if (this.active) throw new Error('A run is already active.');
    if (!workerIds.length || new Set(workerIds).size !== workerIds.length ||
        ![rounds, batchSize, localSteps].every(value => Number.isInteger(value) && value > 0) ||
        ![learningRate, outerLearningRate].every(value => Number.isFinite(value) && value > 0) ||
        !(outerMomentum >= 0 && outerMomentum < 1) ||
        !Number.isFinite(deadlineMs) || deadlineMs <= 0 ||
        !(Number.isInteger(maxWorkersPerRound) && maxWorkersPerRound > 0)) throw new Error('Invalid training settings.');
    const run = {
      id: crypto.randomUUID(), taskId: this.task.id, workerIds: [...workerIds], participants: [...workerIds],
      rounds, batchSize, localSteps,
      optimizer, learningRate, outerLearningRate, outerMomentum,
      startedAt: new Date().toISOString(), history: [], outcome: 'running',
    };
    this.active = run;
    const lastPicked = new Map(); // workerId → last round it trained in
    const buffers = this.task.variables.map(variable => tf.tidy(() => tf.variable(tf.zerosLike(variable), false)));
    const started = performance.now();
    const checkCancelled = () => { if (run.cancelReason) throw new Error(run.cancelReason); };
    const record = async (round, updates = [], waitMs = 0, dropped = []) => {
      // Tasks return a loss, or { loss, accuracy }.
      const evaluation = await this.task.evaluate();
      const { loss, accuracy } = typeof evaluation === 'number' ? { loss: evaluation } : evaluation;
      checkCancelled();
      if (!Number.isFinite(loss)) throw new Error('Evaluation loss is not finite.');
      const point = { round, loss, accuracy, elapsedMs: performance.now() - started, waitMs, updates, dropped };
      run.history.push(point);
      this.onProgress(point, run.history);
    };
    try {
      await record(0);
      for (let round = 1; round <= rounds; round++) {
        const weights = await readValues(this.task.variables);
        checkCancelled();
        if (!run.workerIds.length) throw new Error('Every worker left.');
        // With a huge crowd, each round uses the workers that have waited longest, so everyone
        // takes turns. Since the relay adds up updates, 500 per round took ~1 s per round in
        // scripts/load-test.js (2.5x one phone). Sort is stable: ties keep join order.
        const picked = [...run.workerIds]
          .sort((a, b) => (lastPicked.get(a) ?? 0) - (lastPicked.get(b) ?? 0))
          .slice(0, maxWorkersPerRound);
        for (const id of picked) lastPicked.set(id, round);
        // Worker i gets examples [i × batchSize, (i + 1) × batchSize) of each global step.
        const assignments = picked.map((workerId, i) => ({ workerId, offset: i * batchSize, count: batchSize }));
        const waitStarted = performance.now();
        // Round 1 also downloads data and warms up each worker's GPU.
        const { average, updates, dropped } = await this.requestUpdates(
          run, round, assignments, weights, round === 1 ? 4 * deadlineMs : deadlineMs);
        const waitMs = performance.now() - waitStarted;
        checkCancelled();
        run.workerIds = run.workerIds.filter(id => !dropped.some(item => item.workerId === id));
        // Nesterov, same formula as torch.optim.SGD: buf = m * buf + g; w -= lr * (g + m * buf).
        tf.tidy(() => this.task.variables.forEach((variable, i) => {
          const gradient = tf.tensor(average[i], variable.shape, 'float32').neg();
          buffers[i].assign(buffers[i].mul(outerMomentum).add(gradient));
          variable.assign(variable.sub(gradient.add(buffers[i].mul(outerMomentum)).mul(outerLearningRate)));
        }));
        await record(round, updates.map(({ workerId, count, loss, computeMs }) => ({
          workerId, count, loss, computeMs,
        })), waitMs, dropped);
      }
      run.outcome = 'completed';
    } catch (error) {
      run.outcome = run.cancelReason ? 'stopped' : 'failed';
      run.reason = error.message;
    } finally {
      tf.dispose(buffers);
      this.active = null;
    }
    run.elapsedMs = performance.now() - started;
    return run;
  }
}
