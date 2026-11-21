import './style.css';
import { initializeBackend, memoryUsage, onGpuReset } from './backend.js';
import { SharedTrainer, timingSummary } from './shared-trainer.js';
import { assignValues, computeGradients } from './tensors.js';
import { tasks } from './tasks.js';

const $ = id => document.getElementById(id);
const role = new URLSearchParams(location.search).get('role') === 'worker' ? 'worker' : 'coordinator';
$('role-title').textContent = role === 'coordinator' ? 'Shared training / Coordinator' : 'Shared training / Worker';
$('role-note').textContent = role === 'coordinator'
  ? 'This tab combines worker gradients to update one shared model.'
  : 'This tab computes gradients for its share of the training examples.';
$(role).hidden = false;
for (const { taskId, name } of tasks) $('task').add(new Option(name, taskId));

let selected = tasks[0];
let task, trainer, socket, backend;
let connected = false, running = false;
let workers = [], lastRun;

function send(message) {
  if (socket.readyState !== WebSocket.OPEN) throw new Error('Relay is disconnected. Reload to reconnect.');
  socket.send(JSON.stringify(message));
}
function status(message) { $('status').textContent = message; }
const canRun = worker => worker.ready && worker.tasks.includes(selected.taskId);
function refreshControls() {
  $('start').disabled = !connected || running || !workers.some(canRun);
  $('stop').disabled = !running;
  $('reset').disabled = running || !task;
  $('export').disabled = running || !lastRun;
  $('task').disabled = $('steps').disabled = $('batch-size').disabled = running;
}
const TIMING_IDS = { stepMs: 'step-time', computeMs: 'compute-time', networkMs: 'network-time', coordinatorMs: 'coordinator-time' };
function showProgress(point, history) {
  $('step').textContent = point.step;
  $('loss').textContent = point.loss.toPrecision(5);
  $('elapsed').textContent = `${(point.elapsedMs / 1000).toFixed(2)}s`;
  const timing = timingSummary(history);
  for (const [key, id] of Object.entries(TIMING_IDS)) {
    $(id).textContent = timing ? `${Math.round(timing[key])} ms` : '—';
  }
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
  lastRun = undefined;
  $('step').textContent = '0';
  for (const id of ['loss', 'elapsed', ...Object.values(TIMING_IDS)]) $(id).textContent = '—';
}
function showWorkers() {
  $('workers').replaceChildren();
  $('worker-count').textContent = workers.length;
  for (const [index, worker] of workers.entries()) {
    const li = document.createElement('li');
    const backendLabel = worker.ready ? (worker.backend === 'webgl' ? 'WebGL' : 'CPU fallback') : 'preparing';
    const state = !worker.ready ? 'waiting' : canRun(worker) ? 'ready' : 'different task version';
    li.textContent = `Worker ${index + 1} (${worker.id.slice(0, 8)}) · ${backendLabel} · ${state}`;
    $('workers').append(li);
  }
  if (!workers.length) {
    const li = document.createElement('li');
    li.textContent = 'No workers. Open a worker tab.';
    $('workers').append(li);
  }
  const active = trainer?.active;
  if (active && active.workerIds.some(id => !workers.some(worker => worker.id === id && worker.ready))) {
    trainer.stop('A participating worker disconnected. No incomplete step was applied.');
  }
  refreshControls();
}

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
    $('examples').textContent = '0';
  }
  $('worker-status').textContent = `Step ${message.step}: computing gradients…`;
  assignValues(task.variables, message.weights);
  const result = await computeGradients(task, message);
  examplesProcessed += message.count;
  $('examples').textContent = examplesProcessed;
  $('compute').textContent = `${result.computeMs.toFixed(1)}ms`;
  const { tensors, gpuBytes } = memoryUsage();
  $('gpu-memory').textContent = gpuBytes < 2 ** 20 ? `${Math.round(gpuBytes / 1024)} KB` : `${(gpuBytes / 2 ** 20).toFixed(1)} MB`;
  $('tensors').textContent = tensors;
  send({ type: 'gradient', runId: message.runId, step: message.step, count: message.count, ...result });
  currentStep = undefined;
  $('worker-status').textContent = `Step ${message.step}: gradients sent. Waiting.`;
}

// Report the reset for any in-flight step, then leave: this page can't compute anymore.
function handleGpuReset() {
  const error = 'GPU reset (WebGL context lost). Reload this page to rejoin.';
  $(role === 'worker' ? 'worker-status' : 'status').textContent = error;
  if (currentStep && socket.readyState === WebSocket.OPEN) {
    send({ type: 'worker-error', runId: currentStep.runId, step: currentStep.step, error });
  }
  socket.close();
}

async function initialize() {
  backend = await initializeBackend('webgl');
  const backendLabel = backend === 'webgl' ? 'WebGL (browser graphics)' : 'CPU fallback (not GPU)';
  $('backend').textContent = role === 'coordinator'
    ? `Backend: ${backendLabel}. Combines updates and checks loss; worker tabs compute gradients.`
    : `Training backend: ${backendLabel}.`;
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
      $('connection').textContent = `Connected as ${role} · ${message.id.slice(0, 8)}`;
      if (role === 'worker') send({ type: 'ready', backend, tasks: tasks.map(({ taskId }) => taskId) });
      refreshControls();
    } else if (message.type === 'presence') {
      workers = message.workers;
      showWorkers();
    } else if (role === 'worker' && message.type === 'step') {
      // Serialize assignments so a late old step cannot mutate a newer step's model.
      queue = queue.then(() => work(message)).catch(error => {
        $('worker-status').textContent = `Worker failed: ${error.message}`;
        if (socket.readyState === WebSocket.OPEN) {
          send({ type: 'worker-error', runId: message.runId, step: message.step, error: error.message });
        }
      });
    } else if (message.type === 'gradient' || message.type === 'worker-error') {
      trainer?.receive(message);
    } else if (message.type === 'coordinator-left') {
      $('worker-status').textContent = 'Coordinator disconnected. Waiting for another coordinator.';
    } else if (message.type === 'error') {
      trainer?.stop(message.error);
      $('connection').textContent = `Relay error: ${message.error}`;
    }
  });
  socket.addEventListener('close', () => {
    connected = false;
    trainer?.stop('Relay disconnected.');
    $('connection').textContent = 'Disconnected. Reload this page to reconnect.';
    refreshControls();
  });
  socket.addEventListener('error', () => {
    $('connection').textContent = 'Connection failed. Is npm run dev running?';
  });
  showWorkers();
}

$('start').addEventListener('click', async () => {
  const steps = Number($('steps').value);
  const batchSize = Number($('batch-size').value);
  if (!Number.isInteger(steps) || steps < 1 || steps > 10_000 ||
      !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 100_000) {
    status('Choose a positive whole-number batch size (≤100,000) and step count (≤10,000).');
    return;
  }
  const participants = workers.filter(canRun);
  running = true;
  resetModel();
  refreshControls();
  status(`Training one shared model with ${participants.length} worker(s)…`);
  try {
    const run = await trainer.train({
      workerIds: participants.map(worker => worker.id), steps, batchSize,
      learningRate: selected.defaults.learningRate,
    });
    lastRun = {
      ...run, timing: timingSummary(run.history), workers: participants,
      coordinatorBackend: backend, userAgent: navigator.userAgent,
    };
    const outcome = { completed: 'Completed', stopped: 'Stopped', failed: 'Failed' }[run.outcome];
    status(`${outcome}: ${run.history.at(-1)?.step ?? 0} of ${steps} steps.${run.reason ? ' ' + run.reason : ''}`);
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
$('reset').addEventListener('click', () => {
  resetModel();
  refreshControls();
  status('Model reset.');
});
$('export').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastRun, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `shared-training-${Date.now()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
void initialize().catch(error => {
  $('connection').textContent = `Initialization failed: ${error.message}`;
});
