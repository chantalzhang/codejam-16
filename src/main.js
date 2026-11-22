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
// Connect to this page's own server, or a hosted relay: ?relay=wss://….workers.dev/swarm
// (lets the laptop page run locally, so runs still save, while phones use the hosted pages).
const relayUrl = params.get('relay') ?? `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/swarm`;
const relayPage = new URL(relayUrl.replace(/^ws/, 'http'));
$('worker-link').href = `${relayPage.origin}/?role=worker`;
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
  // iPads report themselves as Macs; the touchscreen gives them away.
  if (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1) return 'iPad';
  const names = [['iPhone', /iPhone/], ['iPad', /iPad/], ['Android', /Android/], ['Mac', /Macintosh/], ['Windows', /Windows/], ['Linux', /Linux/]];
  return names.find(([, pattern]) => pattern.test(navigator.userAgent))?.[0] ?? 'Browser';
}

// Phones without WebGPU sit out: WebGL froze iPhones mid-training, and CPU is far too slow.
// Laptops are fine on WebGL. Forcing a backend in the link (?backend=) skips this check.
function sitOutReason() {
  if (backend === 'webgpu' || params.has('backend') || !['iPhone', 'iPad', 'Android'].includes(deviceName())) return null;
  return 'sitting out: no WebGPU in this browser';
}

function send(message) {
  if (socket.readyState !== WebSocket.OPEN) throw new Error('Relay is disconnected. Reload to reconnect.');
  socket.send(JSON.stringify(message));
  if (message.type === 'round') {
    for (const { workerId } of message.assignments) workerInfo(workerId).activity = `training round ${message.round}`;
    showWorkers();
  }
}

// ---- Coordinator (laptop) ----

function status(message) { $('status').textContent = message; }
const canRun = worker => worker.ready && worker.tasks.includes(selected.taskId);
function refreshControls() {
  $('start').disabled = !connected || running || !workers.some(canRun);
  $('stop').disabled = !running;
  for (const id of ['task', 'rounds', 'batch-size', 'local-steps', 'target']) $(id).disabled = running;
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
  $('target').value = selected.defaults.targetAccuracy ? Math.round(selected.defaults.targetAccuracy * 100) : '';
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
  if (worker.note) return ['failed', worker.note];
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
    if (!workers.some(worker => worker.id === id && worker.ready)) trainer.removeWorker(id);
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
let joinNumber; // From the relay: this worker's place in join order.
function workerTask(taskId) {
  if (!workerTasks.has(taskId)) {
    const module = tasks.find(candidate => candidate.taskId === taskId);
    if (!module) throw new Error('Unknown task. Reload this page.');
    workerTasks.set(taskId, module.createTask({ joinNumber }));
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

// A sleeping screen pauses the page, so workers ask the browser to keep it on. The browser
// releases the lock whenever the page is hidden, and some only grant it after a tap.
let screenLock;
async function keepScreenOn() {
  if (screenLock && !screenLock.released) return;
  try {
    screenLock = await navigator.wakeLock.request('screen');
    $('screen').textContent = 'Screen stays on while this tab is open.';
    screenLock.addEventListener('release', () => { $('screen').textContent = 'Tap anywhere to keep the screen on.'; });
  } catch {
    $('screen').textContent = navigator.wakeLock
      ? 'Tap anywhere to keep the screen on.'
      : "This browser can't keep the screen on. Keep it on yourself.";
  }
}

// ---- Connection ----

function receive(message) {
  if (message.type === 'round-result') {
    for (const { workerId, count, computeMs, memory } of message.updates) {
      const info = workerInfo(workerId);
      Object.assign(info, { activity: `round ${message.round} done`, computeMs, memory });
      info.examples += count;
    }
    trainer?.receive(message);
    showWorkers();
  } else if (message.type === 'worker-error') {
    workerInfo(message.workerId).error = message.error;
    showWorkers();
  }
}

async function initialize() {
  backend = await initializeBackend(role === 'worker' ? preferredBackend : 'webgl');
  onGpuReset(handleGpuReset);
  if (role === 'worker') {
    void keepScreenOn();
    document.addEventListener('visibilitychange', () => { if (!document.hidden) void keepScreenOn(); });
    document.addEventListener('pointerdown', () => void keepScreenOn());
  }
  if (role === 'coordinator') {
    showTask();
    resetModel();
  }
  socket = new WebSocket(relayUrl);
  socket.addEventListener('open', () => send({ type: 'join', role }));
  let queue = Promise.resolve();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.type === 'welcome') {
      joinNumber = message.joinNumber;
      connected = true;
      $('connection').textContent = `Connected · ${deviceName()} · ${backendLabel(backend)}`;
      if (role === 'worker') {
        const note = sitOutReason();
        // A worker that sits out offers no tasks, so the laptop never gives it work.
        send({ type: 'ready', backend, device: deviceName(), note, tasks: note ? [] : tasks.map(({ taskId }) => taskId) });
        if (note) {
          $('worker-status').textContent = "This browser doesn't support WebGPU, so this phone is sitting out " +
            '(older GPU paths freeze during training). Updating iOS or using a newer browser should fix it.';
        }
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
  // Empty means no target: run every round.
  const target = $('target').value === '' ? null : Number($('target').value) / 100;
  if (target !== null && !(target > 0 && target <= 1)) {
    status('Stop at accuracy must be between 1 and 100, or empty.');
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
  $('result').hidden = true;
  showWorkers();
  status(`Training one shared model with ${participants.length} worker(s)…`);
  try {
    const run = await trainer.train({
      workerIds: participants.map(worker => worker.id), rounds, batchSize, localSteps,
      optimizer: selected.defaults.optimizer, learningRate: selected.defaults.learningRate,
      outerLearningRate: selected.defaults.outerLearningRate, outerMomentum: selected.defaults.outerMomentum,
      targetAccuracy: target,
    });
    const outcome = { completed: 'Completed', stopped: 'Stopped', failed: 'Failed' }[run.outcome];
    status(`${outcome}: ${run.history.at(-1)?.round ?? 0} of ${rounds} rounds.${run.reason ? ' ' + run.reason : ''}`);
    if (run.reachedTarget) {
      const { accuracy, elapsedMs, workers: count } = run.reachedTarget;
      $('result').textContent = `Reached ${(accuracy * 100).toFixed(1)}% accuracy in ${(elapsedMs / 1000).toFixed(1)} s ` +
        `with ${count} phone${count === 1 ? '' : 's'}`;
      $('result').hidden = false;
    }
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
