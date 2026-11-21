import * as tf from '@tensorflow/tfjs';
import { readValues, validateValues } from './tensors.js';

export function splitBatch(batchSize, workerIds) {
  if (!Number.isInteger(batchSize) || batchSize < workerIds.length || !workerIds.length ||
      new Set(workerIds).size !== workerIds.length) {
    throw new Error('Use distinct workers and a batch size at least as large as the worker count.');
  }
  let offset = 0;
  return workerIds.map((workerId, index) => {
    const count = Math.floor(batchSize / workerIds.length) + (index < batchSize % workerIds.length ? 1 : 0);
    const assignment = { workerId, offset, count };
    offset += count;
    return assignment;
  });
}

export function averageGradients(updates, variables) {
  if (!updates.length) throw new Error('No gradients received.');
  for (const update of updates) {
    if (!Number.isInteger(update.count) || update.count <= 0) throw new Error('Invalid example count.');
    validateValues(update.gradients, variables);
  }
  const total = updates.reduce((sum, update) => sum + update.count, 0);
  return variables.map((variable, tensor) => Array.from({ length: variable.size }, (_, element) =>
    updates.reduce((sum, update) => sum + update.gradients[tensor][element] * update.count, 0) / total));
}

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
    if (!pending || this.active?.cancelReason || message.runId !== this.active?.id ||
        message.step !== pending.step || pending.results.has(message.workerId)) return;
    const assignment = pending.assignments.find(item => item.workerId === message.workerId);
    if (!assignment) return;
    if (message.type === 'worker-error') {
      pending.reject(new Error(`Worker failed: ${message.error}`));
      return;
    }
    try {
      if (message.count !== assignment.count || !Number.isFinite(message.loss) ||
          !Number.isFinite(message.computeMs) || message.computeMs < 0) {
        throw new Error('Worker returned invalid step statistics.');
      }
      validateValues(message.gradients, this.task.variables);
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

  async requestGradients(run, step, assignments, weights, timeoutMs) {
    const pending = { step, assignments, results: new Map() };
    let timer;
    try {
      return await new Promise((resolve, reject) => {
        Object.assign(pending, { resolve, reject });
        this.pending = pending;
        timer = setTimeout(() => reject(new Error('A worker timed out; no partial update was applied.')), timeoutMs);
        for (const { workerId, offset, count } of assignments) {
          this.send({
            type: 'step', to: workerId, taskId: this.task.id,
            runId: run.id, step, offset, count, weights,
          });
        }
      });
    } finally {
      clearTimeout(timer);
      if (this.pending === pending) this.pending = null;
    }
  }

  async train({ workerIds, steps, batchSize, learningRate, timeoutMs = 10_000 }) {
    if (this.active) throw new Error('A run is already active.');
    if (!Number.isInteger(steps) || steps <= 0 ||
        !Number.isFinite(learningRate) || learningRate <= 0 ||
        !Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid training settings.');
    const assignments = splitBatch(batchSize, workerIds);
    const run = {
      id: crypto.randomUUID(), taskId: this.task.id, workerIds,
      steps, batchSize, learningRate, startedAt: new Date().toISOString(),
      history: [], outcome: 'running',
    };
    this.active = run;
    const optimizer = tf.train.sgd(learningRate);
    const started = performance.now();
    const checkCancelled = () => { if (run.cancelReason) throw new Error(run.cancelReason); };
    const record = async (step, updates = []) => {
      const loss = await this.task.evaluate();
      checkCancelled();
      if (!Number.isFinite(loss)) throw new Error('Evaluation loss is not finite.');
      const point = { step, loss, elapsedMs: performance.now() - started, updates };
      run.history.push(point);
      this.onProgress(point);
    };
    try {
      await record(0);
      for (let step = 1; step <= steps; step++) {
        const weights = await readValues(this.task.variables);
        checkCancelled();
        const updates = await this.requestGradients(run, step, assignments, weights, timeoutMs);
        checkCancelled();
        const gradients = averageGradients(updates, this.task.variables);
        tf.tidy(() => optimizer.applyGradients(Object.fromEntries(
          this.task.variables.map((variable, i) => [
            variable.name, tf.tensor(gradients[i], variable.shape, 'float32'),
          ]),
        )));
        await record(step, updates.map(({ workerId, count, loss, computeMs }) => ({
          workerId, count, loss, computeMs,
        })));
      }
      run.outcome = 'completed';
    } catch (error) {
      run.outcome = run.cancelReason ? 'stopped' : 'failed';
      run.reason = error.message;
    } finally {
      optimizer.dispose();
      this.active = null;
    }
    run.elapsedMs = performance.now() - started;
    return run;
  }
}
