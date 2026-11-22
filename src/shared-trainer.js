import * as tf from '@tensorflow/tfjs';
import { packValues, readValues, unpackValues, validateValues } from './tensors.js';

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

  receive(message) {
    const pending = this.pending;
    const run = this.active;
    if (!pending || run?.cancelReason || message.runId !== run?.id || message.round !== pending.round ||
        pending.results.has(message.workerId) || pending.dropped.has(message.workerId)) return;
    const assignment = pending.assignments.find(item => item.workerId === message.workerId);
    if (!assignment) return;
    if (message.type === 'worker-error') {
      this.drop(message.workerId, `failed: ${message.error}`);
      return;
    }
    let delta;
    try {
      if (message.count !== assignment.count * run.localSteps || !Number.isFinite(message.loss) ||
          !Number.isFinite(message.computeMs) || message.computeMs < 0) throw new Error();
      delta = unpackValues(message.delta, this.task.variables);
      validateValues(delta, this.task.variables);
    } catch {
      this.drop(message.workerId, 'sent an invalid update');
      return;
    }
    pending.results.set(message.workerId, { ...message, delta });
    pending.settle();
  }

  // Removes a worker from the current round; the round goes on without it.
  drop(workerId, reason) {
    const pending = this.pending;
    if (!pending?.assignments.some(item => item.workerId === workerId) || pending.results.has(workerId)) return;
    pending.dropped.set(workerId, reason);
    pending.settle();
  }

  // A worker connected mid-run: it gets the current model and trains from the next round.
  // Workers already in this run (including dropped ones) are not re-added.
  addWorker(workerId) {
    const run = this.active;
    if (!run || run.cancelReason || run.participants.includes(workerId)) return;
    run.participants.push(workerId);
    run.workerIds.push(workerId);
  }

  // A worker left: stop waiting for it this round and leave it out of later rounds.
  removeWorker(workerId, reason) {
    const run = this.active;
    if (!run?.workerIds.includes(workerId)) return;
    run.workerIds = run.workerIds.filter(id => id !== workerId);
    this.drop(workerId, reason);
  }

  stop(reason = 'Stopped by coordinator') {
    if (!this.active) return;
    this.active.cancelReason = reason;
    this.pending?.reject(new Error(reason));
  }

  // Resolves once every worker has answered or dropped out, or at the deadline, with
  // whatever updates arrived. Fails only if no worker sent a usable update.
  async requestUpdates(run, round, assignments, weights, deadlineMs) {
    const pending = { round, assignments, results: new Map(), dropped: new Map() };
    const packedWeights = packValues(weights);
    let timer;
    try {
      return await new Promise((resolve, reject) => {
        const finish = () => {
          if (pending.results.size) resolve({ updates: [...pending.results.values()], dropped: pending.dropped });
          else reject(new Error(`No worker sent a usable update (${[...pending.dropped.values()].join('; ')}).`));
        };
        Object.assign(pending, {
          reject,
          settle: () => { if (pending.results.size + pending.dropped.size === assignments.length) finish(); },
        });
        this.pending = pending;
        timer = setTimeout(() => {
          for (const { workerId } of assignments) {
            if (!pending.results.has(workerId) && !pending.dropped.has(workerId)) {
              pending.dropped.set(workerId, `missed the round ${round} deadline`);
            }
          }
          finish();
        }, deadlineMs);
        for (const { workerId, offset, count } of assignments) {
          this.send({
            type: 'round', to: workerId, taskId: this.task.id, runId: run.id, round,
            firstStep: (round - 1) * run.localSteps + 1, localSteps: run.localSteps,
            optimizer: run.optimizer, learningRate: run.learningRate, offset, count, weights: packedWeights,
          });
        }
      });
    } finally {
      clearTimeout(timer);
      if (this.pending === pending) this.pending = null;
    }
  }

  async train({
    workerIds, rounds, batchSize, localSteps, optimizer, learningRate,
    outerLearningRate = 1, outerMomentum = 0, deadlineMs = 10_000, maxWorkersPerRound = 100,
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
        // With a crowd, each round uses the workers that have waited longest, so everyone
        // takes turns. 100 per round kept rounds under 1.7x slower in scripts/load-test.js;
        // 200 was 3.2x. Sort is stable: ties keep join order.
        const picked = [...run.workerIds]
          .sort((a, b) => (lastPicked.get(a) ?? 0) - (lastPicked.get(b) ?? 0))
          .slice(0, maxWorkersPerRound);
        for (const id of picked) lastPicked.set(id, round);
        // Worker i gets examples [i × batchSize, (i + 1) × batchSize) of each global step.
        const assignments = picked.map((workerId, i) => ({ workerId, offset: i * batchSize, count: batchSize }));
        const waitStarted = performance.now();
        // Round 1 also downloads data and warms up each worker's GPU.
        const { updates, dropped } = await this.requestUpdates(
          run, round, assignments, weights, round === 1 ? 4 * deadlineMs : deadlineMs);
        const waitMs = performance.now() - waitStarted;
        checkCancelled();
        run.workerIds = run.workerIds.filter(id => !dropped.has(id));
        const average = averageUpdates(updates, this.task.variables);
        // Nesterov, same formula as torch.optim.SGD: buf = m * buf + g; w -= lr * (g + m * buf).
        tf.tidy(() => this.task.variables.forEach((variable, i) => {
          const gradient = tf.tensor(average[i], variable.shape, 'float32').neg();
          buffers[i].assign(buffers[i].mul(outerMomentum).add(gradient));
          variable.assign(variable.sub(gradient.add(buffers[i].mul(outerMomentum)).mul(outerLearningRate)));
        }));
        await record(round, updates.map(({ workerId, count, loss, computeMs }) => ({
          workerId, count, loss, computeMs,
        })), waitMs, [...dropped].map(([workerId, reason]) => ({ workerId, reason })));
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
