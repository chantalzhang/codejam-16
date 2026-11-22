import './style.css';
import { initializeBackend, memoryUsage, onGpuReset } from './backend.js';
import { SharedTrainer, timingSummary } from './shared-trainer.js';
import { assignValues, computeGradients } from './tensors.js';
import { tasks } from './tasks.js';

const $ = id => document.getElementById(id);
const role = new URLSearchParams(location.search).get('role') === 'worker' ? 'worker' : 'coordinator';
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
  if (message.type === 'step') {
    workerInfo(message.to).activity = `computing step ${message.step}`;
    showWorkers();
  }
}

// ---- Coordinator (laptop) ----

function status(message) { $('status').textContent = message; }
const canRun = worker => worker.ready && worker.tasks.includes(selected.taskId);
function refreshControls() {
  $('start').disabled = !connected || running || !workers.some(canRun);
  $('stop').disabled = !running;
  $('task').disabled = $('steps').disabled = $('batch-size').disabled = running;
}
const TIMING_IDS = { stepMs: 'step-time', computeMs: 'compute-time', networkMs: 'network-time', coordinatorMs: 'coordinator-time' };
function showProgress(point, history) {
  $('step').textContent = `${point.step} / ${trainer.active.steps}`;
  $('loss').textContent = point.loss.toPrecision(4);
  $('accuracy').textContent = point.accuracy === undefined ? '—' : `${(point.accuracy * 100).toFixed(1)}%`;
  $('elapsed').textContent = `${(point.elapsedMs / 1000).toFixed(1)} s`;
  const timing = timingSummary(history);
  for (const [key, id] of Object.entries(TIMING_IDS)) $(id).textContent = timing ? formatMs(timing[key]) : '—';
}
function showTask() {
  $('task-note').textContent = selected.description;
  $('steps').value = selected.defaults.steps;
  $('batch-size').value = selected.defaults.batchSize;
}
function resetModel() {
  task?.dispose();
  task = selected.createTask();
  trainer = new SharedTrainer(task, { send, onProgress: showProgress });
  for (const id of ['step', 'loss', 'accuracy', 'elapsed', ...Object.values(TIMING_IDS)]) $(id).textContent = '—';
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
  const backendLabel = info.backend === 'cpu' ? 'CPU fallback' : 'WebGL';
  title.textContent = `Worker ${info.number} · ${info.device ?? '…'} · ${backendLabel}`;
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
  const active = trainer?.active;
  if (active && active.workerIds.some(id => !workers.some(worker => worker.id === id && worker.ready))) {
    trainer.stop('A worker disconnected. No incomplete step was applied.');
  }
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

let workerRunId, examplesProcessed = 0, currentStep;
async function work(message) {
  currentStep = message;
  const task = workerTask(message.taskId);
  if (!Number.isInteger(message.count) || message.count <= 0 ||
      !Number.isInteger(message.offset) || message.offset < 0 ||
      !Number.isInteger(message.step) || message.step <= 0) throw new Error('Invalid batch assignment.');
  if (workerRunId !== message.runId) {
    workerRunId = message.runId;
    examplesProcessed = 0;
  }
  $('worker-status').textContent = `Step ${message.step}: computing gradients…`;
  assignValues(task.variables, message.weights);
  const result = await computeGradients(task, message);
  const memory = memoryUsage();
  examplesProcessed += message.count;
  $('examples').textContent = examplesProcessed;
  $('compute').textContent = formatMs(result.computeMs);
  $('gpu-memory').textContent = formatBytes(memory.gpuBytes);
  $('tensors').textContent = memory.tensors;
  send({ type: 'gradient', runId: message.runId, step: message.step, count: message.count, memory, ...result });
  currentStep = undefined;
  $('worker-status').textContent = `Step ${message.step} sent. Waiting for the next one.`;
}

// Tell the laptop, then leave: after a GPU reset this page can't compute anymore.
function handleGpuReset() {
  const error = 'GPU reset (WebGL context lost). Reload this page to rejoin.';
  $(role === 'worker' ? 'worker-status' : 'status').textContent = error;
  if (socket.readyState === WebSocket.OPEN) {
    send({ type: 'worker-error', runId: currentStep?.runId, step: currentStep?.step, error });
  }
  socket.close();
}

// ---- Connection ----

function receive(message) {
  if (message.type === 'gradient') {
    const info = workerInfo(message.workerId);
    Object.assign(info, { activity: `step ${message.step} done`, computeMs: message.computeMs, memory: message.memory });
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
  backend = await initializeBackend('webgl');
  const backendLabel = backend === 'webgl' ? 'WebGL' : 'CPU fallback (not GPU)';
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
      $('connection').textContent = `Connected · ${deviceName()} · ${backendLabel}`;
      if (role === 'worker') {
        send({ type: 'ready', backend, device: deviceName(), tasks: tasks.map(({ taskId }) => taskId) });
      }
      refreshControls();
    } else if (message.type === 'presence') {
      workers = message.workers;
      showWorkers();
    } else if (role === 'worker' && message.type === 'step') {
      // Serialize assignments so a late old step cannot mutate a newer step's model.
      queue = queue.then(() => work(message)).catch(error => {
        $('worker-status').textContent = `Failed: ${error.message}`;
        if (socket.readyState === WebSocket.OPEN) {
          send({ type: 'worker-error', runId: message.runId, step: message.step, error: error.message });
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
  const steps = Number($('steps').value);
  const batchSize = Number($('batch-size').value);
  if (!Number.isInteger(steps) || steps < 1 || steps > 10_000 ||
      !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 100_000) {
    status('Choose a whole-number batch size (≤100,000) and step count (≤10,000).');
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
      workerIds: participants.map(worker => worker.id), steps, batchSize,
      learningRate: selected.defaults.learningRate,
    });
    const outcome = { completed: 'Completed', stopped: 'Stopped', failed: 'Failed' }[run.outcome];
    status(`${outcome}: ${run.history.at(-1)?.step ?? 0} of ${steps} steps.${run.reason ? ' ' + run.reason : ''}`);
    await saveRun({
      ...run, timing: timingSummary(run.history), workers: participants.map(worker => snapshot(worker.id)),
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
