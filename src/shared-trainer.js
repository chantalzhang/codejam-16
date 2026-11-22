import * as tf from '@tensorflow/tfjs';
import { readValues, validateValues } from './tensors.js';

// Average of the workers' weight changes, weighted by how many examples each trained on.
export function averageUpdates(updates, variables) {
  if (!updates.length) throw new Error('No updates received.');
  for (const update of updates) {
    if (!Number.isInteger(update.count) || update.count <= 0) throw new Error('Invalid example count.');
    validateValues(update.delta, variables);
  }
  const total = updates.reduce((sum, update) => sum + update.count, 0);
  return variables.map((variable, tensor) => Array.from({ length: variable.size }, (_, element) =>
    updates.reduce((sum, update) => sum + update.delta[tensor][element] * update.count, 0) / total));
}

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
// `batchSize` examples per step, then the shared model moves by the average change.
export class SharedTrainer {
  constructor(task, { send, onProgress = () => {} }) {
    this.task = task;
    this.send = send;
    this.onProgress = onProgress;
    this.active = null;
    this.pending = null;
  }

  receive(message) {
    const pending = this.pending;
    const run = this.active;
    if (!pending || run?.cancelReason || message.runId !== run?.id ||
        message.round !== pending.round || pending.results.has(message.workerId)) return;
    const assignment = pending.assignments.find(item => item.workerId === message.workerId);
    if (!assignment) return;
    if (message.type === 'worker-error') {
      pending.reject(new Error(`Worker failed: ${message.error}`));
      return;
    }
    try {
      if (message.count !== assignment.count * run.localSteps || !Number.isFinite(message.loss) ||
          !Number.isFinite(message.computeMs) || message.computeMs < 0) {
        throw new Error('Worker returned invalid round statistics.');
      }
      validateValues(message.delta, this.task.variables);
      pending.results.set(message.workerId, message);
      if (pending.results.size === pending.assignments.length) {
        pending.resolve([...pending.results.values()]);
      }
    } catch (error) {
      pending.reject(error);
    }
  }

  stop(reason = 'Stopped by coordinator') {
    if (!this.active) return;
    this.active.cancelReason = reason;
    this.pending?.reject(new Error(reason));
  }

  async requestUpdates(run, round, assignments, weights, timeoutMs) {
    const pending = { round, assignments, results: new Map() };
    let timer;
    try {
      return await new Promise((resolve, reject) => {
        Object.assign(pending, { resolve, reject });
        this.pending = pending;
        timer = setTimeout(() => reject(new Error('A worker timed out; no partial update was applied.')), timeoutMs);
        for (const { workerId, offset, count } of assignments) {
          this.send({
            type: 'round', to: workerId, taskId: this.task.id, runId: run.id, round,
            firstStep: (round - 1) * run.localSteps + 1, localSteps: run.localSteps,
            optimizer: run.optimizer, learningRate: run.learningRate, offset, count, weights,
          });
        }
      });
    } finally {
      clearTimeout(timer);
      if (this.pending === pending) this.pending = null;
    }
  }

  async train({ workerIds, rounds, batchSize, localSteps, optimizer, learningRate, timeoutMs = 30_000 }) {
    if (this.active) throw new Error('A run is already active.');
    if (!workerIds.length || new Set(workerIds).size !== workerIds.length ||
        ![rounds, batchSize, localSteps].every(value => Number.isInteger(value) && value > 0) ||
        !Number.isFinite(learningRate) || learningRate <= 0 ||
        !Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid training settings.');
    // Worker i always gets examples [i × batchSize, (i + 1) × batchSize) of each global step.
    const assignments = workerIds.map((workerId, i) => ({ workerId, offset: i * batchSize, count: batchSize }));
    const run = {
      id: crypto.randomUUID(), taskId: this.task.id, workerIds, rounds, batchSize, localSteps,
      optimizer, learningRate, startedAt: new Date().toISOString(), history: [], outcome: 'running',
    };
    this.active = run;
    const started = performance.now();
    const checkCancelled = () => { if (run.cancelReason) throw new Error(run.cancelReason); };
    const record = async (round, updates = [], waitMs = 0) => {
      // Tasks return a loss, or { loss, accuracy }.
      const evaluation = await this.task.evaluate();
      const { loss, accuracy } = typeof evaluation === 'number' ? { loss: evaluation } : evaluation;
      checkCancelled();
      if (!Number.isFinite(loss)) throw new Error('Evaluation loss is not finite.');
      const point = { round, loss, accuracy, elapsedMs: performance.now() - started, waitMs, updates };
      run.history.push(point);
      this.onProgress(point, run.history);
    };
    try {
      await record(0);
      for (let round = 1; round <= rounds; round++) {
        const weights = await readValues(this.task.variables);
        checkCancelled();
        const waitStarted = performance.now();
        const updates = await this.requestUpdates(run, round, assignments, weights, timeoutMs);
        const waitMs = performance.now() - waitStarted;
        checkCancelled();
        const average = averageUpdates(updates, this.task.variables);
        tf.tidy(() => this.task.variables.forEach((variable, i) => {
          variable.assign(variable.add(tf.tensor(average[i], variable.shape, 'float32')));
        }));
        await record(round, updates.map(({ workerId, count, loss, computeMs }) => ({
          workerId, count, loss, computeMs,
        })), waitMs);
      }
      run.outcome = 'completed';
    } catch (error) {
      run.outcome = run.cancelReason ? 'stopped' : 'failed';
      run.reason = error.message;
    } finally {
      this.active = null;
    }
    run.elapsedMs = performance.now() - started;
    return run;
  }
}
