import './style.css';
import { BACKENDS, backendLabel, initializeBackend, memoryUsage, onGpuReset } from './backend.js';
import { SharedTrainer, timingSummary } from './shared-trainer.js';
import { createOptimizer, packValues, trainLocally, unpackValues } from './tensors.js';
import { tasks } from './tasks.js';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const role = params.get('role') === 'worker' ? 'worker' : 'coordinator';
// Workers use WebGPU when the browser has it: WebGL froze iPhones mid-training.
// Override for testing, e.g. ?role=worker&backend=webgl.
const preferredBackend = BACKENDS.includes(params.get('backend')) ? params.get('backend')
  : navigator.gpu ? 'webgpu' : 'webgl';
$('role-title').textContent = role === 'coordinator' ? 'Shared training · laptop' : 'Shared training · worker';
$(role).hidden = false;
for (const { taskId, name } of tasks) $('task').add(new Option(name, taskId));

let selected = tasks[0];
let task, trainer, socket, backend;
let connected = false, running = false;
let workers = [];

const formatMs = ms => `${Math.round(ms)} ms`;
const formatBytes = bytes => bytes < 2 ** 20 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / 2 ** 20).toFixed(1)} MB`;

function deviceName() {
  const names = [['iPhone', /iPhone/], ['iPad', /iPad/], ['Android', /Android/], ['Mac', /Macintosh/], ['Windows', /Windows/], ['Linux', /Linux/]];
  return names.find(([, pattern]) => pattern.test(navigator.userAgent))?.[0] ?? 'Browser';
}

function send(message) {
  if (socket.readyState !== WebSocket.OPEN) throw new Error('Relay is disconnected. Reload to reconnect.');
  socket.send(JSON.stringify(message));
  if (message.type === 'round') {
    workerInfo(message.to).activity = `training round ${message.round}`;
    showWorkers();
  }
}

// ---- Coordinator (laptop) ----

function status(message) { $('status').textContent = message; }
const canRun = worker => worker.ready && worker.tasks.includes(selected.taskId);
function refreshControls() {
  $('start').disabled = !connected || running || !workers.some(canRun);
  $('stop').disabled = !running;
  for (const id of ['task', 'rounds', 'batch-size', 'local-steps']) $(id).disabled = running;
}
const TIMING_IDS = { roundMs: 'round-time', computeMs: 'compute-time', networkMs: 'network-time', coordinatorMs: 'coordinator-time' };
function showProgress(point, history) {
  $('round').textContent = `${point.round} / ${trainer.active.rounds}`;
  $('loss').textContent = point.loss.toPrecision(4);
  $('accuracy').textContent = point.accuracy === undefined ? '—' : `${(point.accuracy * 100).toFixed(1)}%`;
  $('elapsed').textContent = `${(point.elapsedMs / 1000).toFixed(1)} s`;
  // Workers that missed the deadline or failed this round are out of the run.
  for (const { workerId, reason } of point.dropped) workerInfo(workerId).error ??= `dropped: ${reason}`;
  if (point.dropped.length) showWorkers();
  const timing = timingSummary(history);
  for (const [key, id] of Object.entries(TIMING_IDS)) $(id).textContent = timing ? formatMs(timing[key]) : '—';
}
function showTask() {
  $('task-note').textContent = selected.description;
  $('rounds').value = selected.defaults.rounds;
  $('batch-size').value = selected.defaults.batchSize;
  $('local-steps').value = selected.defaults.localSteps;
}
function resetModel() {
  task?.dispose();
  task = selected.createTask();
  trainer = new SharedTrainer(task, { send, onProgress: showProgress });
  for (const id of ['round', 'loss', 'accuracy', 'elapsed', ...Object.values(TIMING_IDS)]) $(id).textContent = '—';
}

// What the laptop knows about each worker, kept after it leaves so failures stay visible.
const infos = new Map();
function workerInfo(id) {
  if (!infos.has(id)) infos.set(id, { number: infos.size + 1, examples: 0 });
  return infos.get(id);
}
function workerState(info, worker) {
  if (!worker) return ['left', info.error ? `left · ${info.error}` : 'left'];
  if (info.error) return ['failed', info.error];
  if (!worker.ready) return ['busy', 'joining…'];
  if (!canRun(worker)) return ['failed', 'different task version: reload this worker'];
  if (info.activity) return [info.activity.startsWith('computing') ? 'busy' : 'ready', info.activity];
  return ['ready', 'ready'];
}
function card(id, info) {
  const worker = workers.find(candidate => candidate.id === id);
  const [tone, label] = workerState(info, worker);
  const element = document.createElement('article');
  element.className = `card ${tone}`;
  const title = document.createElement('h3');
  title.textContent = `Worker ${info.number} · ${info.device ?? '…'} · ${backendLabel(info.backend)}`;
  const state = document.createElement('p');
  state.className = 'state';
  state.textContent = label;
  const metrics = document.createElement('dl');
  metrics.className = 'metrics';
  for (const [name, value] of [
    ['Examples', info.examples],
    ['Last compute', info.computeMs === undefined ? '—' : formatMs(info.computeMs)],
    ['GPU memory', info.memory ? formatBytes(info.memory.gpuBytes) : '—'],
    ['Tensors', info.memory?.tensors ?? '—'],
  ]) {
    const item = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = name;
    dd.textContent = value;
    item.append(dt, dd);
    metrics.append(item);
  }
  element.append(title, state, metrics);
  return element;
}
function showWorkers() {
  for (const worker of workers) Object.assign(workerInfo(worker.id), { device: worker.device, backend: worker.backend });
  $('worker-count').textContent = workers.length;
  $('workers').replaceChildren(...[...infos].map(([id, info]) => card(id, info)));
  if (!infos.size) $('workers').textContent = 'No workers yet. Open a worker tab or the phone link.';
  // Workers that disconnect mid-run just leave it; new ones join from the next round.
  for (const id of [...trainer?.active?.workerIds ?? []]) {
    if (!workers.some(worker => worker.id === id && worker.ready)) trainer.removeWorker(id, 'left');
  }
  for (const worker of workers.filter(canRun)) trainer?.addWorker(worker.id);
  refreshControls();
}
function snapshot(id) {
  const { number, device, backend, examples, computeMs, memory, error } = workerInfo(id);
  return { id, number, device, backend, examples, computeMs, memory, error };
}

async function saveRun(run) {
  try {
    const response = await fetch('/api/runs', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(run),
    });
    if (!response.ok) throw new Error();
    $('saved').textContent = `Saved: ${(await response.json()).file}`;
  } catch {
    $('saved').textContent = 'Not saved. Open the laptop page at http://127.0.0.1 to save runs.';
  }
}

// ---- Worker (phone) ----

// Workers create each task the first time the coordinator asks for it.
const workerTasks = new Map();
function workerTask(taskId) {
  if (!workerTasks.has(taskId)) {
    const module = tasks.find(candidate => candidate.taskId === taskId);
    if (!module) throw new Error('Unknown task. Reload this page.');
    workerTasks.set(taskId, module.createTask());
  }
  return workerTasks.get(taskId);
}

// The optimizer (and its state, e.g. Adam's moments) lives for one run.
let workerRun, currentRound;
async function work(message) {
  currentRound = message;
  const task = workerTask(message.taskId);
  const { round, firstStep, localSteps, offset, count } = message;
  if (![round, firstStep, localSteps, count].every(value => Number.isInteger(value) && value > 0) ||
      !Number.isInteger(offset) || offset < 0) throw new Error('Invalid round assignment.');
  if (workerRun?.id !== message.runId) {
    workerRun?.optimizer.dispose();
    workerRun = { id: message.runId, optimizer: createOptimizer(message.optimizer, message.learningRate), examples: 0 };
  }
  $('worker-status').textContent = `Round ${round}: training ${localSteps} step${localSteps > 1 ? 's' : ''}…`;
  const weights = unpackValues(message.weights, task.variables);
  const result = await trainLocally(task, workerRun.optimizer, weights, message);
  const memory = memoryUsage();
  workerRun.examples += count * localSteps;
  $('examples').textContent = workerRun.examples;
  $('compute').textContent = formatMs(result.computeMs);
  $('gpu-memory').textContent = formatBytes(memory.gpuBytes);
  $('tensors').textContent = memory.tensors;
  send({ type: 'update', runId: message.runId, round, count: count * localSteps, memory, ...result, delta: packValues(result.delta) });
  currentRound = undefined;
  $('worker-status').textContent = `Round ${round} sent. Waiting for the next one.`;
}

// Tell the laptop, then leave: after a GPU reset this page can't compute anymore.
function handleGpuReset() {
  const error = 'GPU reset (WebGL context lost). Reload this page to rejoin.';
  $(role === 'worker' ? 'worker-status' : 'status').textContent = error;
  if (socket.readyState === WebSocket.OPEN) {
    send({ type: 'worker-error', runId: currentRound?.runId, round: currentRound?.round, error });
  }
  socket.close();
}

// ---- Connection ----

function receive(message) {
  if (message.type === 'update') {
    const info = workerInfo(message.workerId);
    Object.assign(info, { activity: `round ${message.round} done`, computeMs: message.computeMs, memory: message.memory });
    info.examples += message.count;
    trainer?.receive(message);
    showWorkers();
  } else if (message.type === 'worker-error') {
    workerInfo(message.workerId).error = message.error;
    trainer?.receive(message);
    showWorkers();
  }
}

async function initialize() {
  backend = await initializeBackend(role === 'worker' ? preferredBackend : 'webgl');
  onGpuReset(handleGpuReset);
  if (role === 'coordinator') {
    showTask();
    resetModel();
  }
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/swarm`);
  socket.addEventListener('open', () => send({ type: 'join', role }));
  let queue = Promise.resolve();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.type === 'welcome') {
      connected = true;
      $('connection').textContent = `Connected · ${deviceName()} · ${backendLabel(backend)}`;
      if (role === 'worker') {
        send({ type: 'ready', backend, device: deviceName(), tasks: tasks.map(({ taskId }) => taskId) });
      }
      refreshControls();
    } else if (message.type === 'presence') {
      workers = message.workers;
      showWorkers();
    } else if (role === 'worker' && message.type === 'round') {
      // Serialize assignments so a late old round cannot mutate a newer round's model.
      queue = queue.then(() => work(message)).catch(error => {
        $('worker-status').textContent = `Failed: ${error.message}`;
        if (socket.readyState === WebSocket.OPEN) {
          send({ type: 'worker-error', runId: message.runId, round: message.round, error: error.message });
        }
      });
    } else if (message.type === 'coordinator-left') {
      $('worker-status').textContent = 'Laptop disconnected. Waiting for it to come back.';
    } else if (message.type === 'error') {
      trainer?.stop(message.error);
      $('connection').textContent = `Relay error: ${message.error}`;
    } else {
      receive(message);
    }
  });
  socket.addEventListener('close', () => {
    connected = false;
    trainer?.stop('Relay disconnected.');
    $('connection').textContent = 'Disconnected. Reload this page to reconnect.';
    refreshControls();
  });
  socket.addEventListener('error', () => {
    $('connection').textContent = 'Connection failed. Is npm start running?';
  });
  if (role === 'coordinator') showWorkers();
}

$('start').addEventListener('click', async () => {
  const [rounds, batchSize, localSteps] = ['rounds', 'batch-size', 'local-steps'].map(id => Number($(id).value));
  if (![[rounds, 10_000], [batchSize, 1_000], [localSteps, 100]].every(([value, max]) =>
    Number.isInteger(value) && value >= 1 && value <= max)) {
    status('Choose whole numbers: rounds ≤ 10,000, images per step ≤ 1,000, local steps ≤ 100.');
    return;
  }
  // Fresh run: forget workers that left and clear last run's numbers.
  for (const [id, info] of infos) {
    if (!workers.some(worker => worker.id === id)) infos.delete(id);
    else Object.assign(info, { examples: 0, activity: undefined, error: undefined });
  }
  const participants = workers.filter(canRun);
  running = true;
  resetModel();
  $('saved').textContent = '';
  showWorkers();
  status(`Training one shared model with ${participants.length} worker(s)…`);
  try {
    const run = await trainer.train({
      workerIds: participants.map(worker => worker.id), rounds, batchSize, localSteps,
      optimizer: selected.defaults.optimizer, learningRate: selected.defaults.learningRate,
      outerLearningRate: selected.defaults.outerLearningRate, outerMomentum: selected.defaults.outerMomentum,
    });
    const outcome = { completed: 'Completed', stopped: 'Stopped', failed: 'Failed' }[run.outcome];
    status(`${outcome}: ${run.history.at(-1)?.round ?? 0} of ${rounds} rounds.${run.reason ? ' ' + run.reason : ''}`);
    await saveRun({
      ...run, timing: timingSummary(run.history), workers: run.participants.map(snapshot),
      coordinatorBackend: backend, userAgent: navigator.userAgent,
    });
  } catch (error) {
    status(`Cannot start: ${error.message}`);
  } finally {
    running = false;
    refreshControls();
  }
});
$('task').addEventListener('change', () => {
  selected = tasks.find(({ taskId }) => taskId === $('task').value);
  showTask();
  resetModel();
  showWorkers();
  status(`Selected: ${selected.name}.`);
});
$('stop').addEventListener('click', () => trainer.stop());
void initialize().catch(error => {
  $('connection').textContent = `Initialization failed: ${error.message}`;
});
