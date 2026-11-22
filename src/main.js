import './app.css';
import QRCode from 'qrcode';
import { BACKENDS, backendLabel, initializeBackend, memoryUsage, onGpuReset } from './backend.js';
import { SharedTrainer, timingSummary } from './shared-trainer.js';
import { createOptimizer, packValues, trainLocally, unpackValues } from './tensors.js';
import { tasks } from './tasks.js';
import { clock, drawLossChart, drawProbabilityHistory, drawRateSparkline } from './ui/charts.js';
import { imageUrl } from './ui/images.js';
import { bump, countTo } from './ui/motion.js';
import { createCard, guessChanges, resultSummary, updateCard, winner } from './ui/predictions.js';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
// The plain link is the phone page, so the link you share can't open (or take over) the
// laptop page. The laptop page is ?role=laptop.
const role = params.get('role') === 'laptop' ? 'coordinator' : 'worker';
// Workers use WebGPU when the browser has it: WebGL froze iPhones mid-training.
// Override for testing, e.g. ?role=worker&backend=webgl.
const preferredBackend = BACKENDS.includes(params.get('backend')) ? params.get('backend')
  : navigator.gpu ? 'webgpu' : 'webgl';
// Connect to this page's own server, or a hosted relay: ?relay=wss://….workers.dev/swarm
// (lets the laptop page run locally, so runs still save, while phones use the hosted pages).
const relayUrl = params.get('relay') ?? `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/swarm`;
const relayPage = new URL(relayUrl.replace(/^ws/, 'http'));
$('worker-link').href = `${relayPage.origin}/`;
$('worker-link').textContent = $('dialog-url').textContent = relayPage.host;
if (role === 'coordinator') {
  void QRCode.toDataURL(`${relayPage.origin}/`, { margin: 1, width: 640 }).then(url => { $('join-qr').src = $('dialog-qr').src = url; });
}
document.body.classList.toggle('phone-view', role === 'worker');
$(role).hidden = false;
// The phone and laptop pages each have their own connection line.
const connectionId = role === 'worker' ? 'worker-connection' : 'connection';
for (const { taskId, name } of tasks) $('task').add(new Option(name, taskId));
// Dev controls are behind the Dev button, or open from the start with ?dev.
function showDev(open) {
  $('dev-panel').hidden = !open;
  $('dev-toggle').setAttribute('aria-expanded', open);
}
showDev(params.has('dev'));
$('dev-toggle').addEventListener('click', () => showDev($('dev-panel').hidden));
$('enlarge-qr').addEventListener('click', () => $('join-dialog').showModal());
// Copies the phone link, e.g. to send to people on laptops.
for (const button of document.querySelectorAll('.copy-link')) {
  const label = button.textContent;
  button.addEventListener('click', async () => {
    await navigator.clipboard.writeText(`${relayPage.origin}/`);
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = label; }, 1500);
  });
}

let selected = tasks[0];
let task, trainer, socket, backend;
let connected = false;
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

// What to tell someone whose browser can't train. Apps' built-in browsers (e.g. after
// scanning the QR code inside Instagram) usually lack WebGPU, so point to the real browser.
function sitOutAdvice() {
  const app = [['Instagram', /Instagram/], ['Facebook', /FBAN|FBAV/], ['Snapchat', /Snapchat/], ['TikTok', /musical_ly|BytedanceWebview/],
    ['LinkedIn', /LinkedInApp/], ['Discord', /Discord/]].find(([, pattern]) => pattern.test(navigator.userAgent))?.[0];
  const browser = deviceName() === 'Android' ? 'Chrome' : 'Safari';
  if (app) return `You're in the ${app} browser, which can't train. Copy the link and open it in ${browser} to join.`;
  return deviceName() === 'Android'
    ? "This browser can't train. Open the link in Chrome to join."
    : "This browser can't train. Open the link in Safari to join (needs iOS 26 or newer).";
}

function send(message) {
  if (socket.readyState !== WebSocket.OPEN) throw new Error('Relay is disconnected. Reload to reconnect.');
  socket.send(JSON.stringify(message));
  if (message.type === 'round') {
    for (const { workerId } of message.assignments) {
      workerInfo(workerId).activity = `training round ${message.round}`;
      cells.get(workerId)?.classList.add('training');
    }
    showWorkers();
  }
}

// ---- Coordinator (laptop) ----

// Big-screen state for the current run. display holds the task's fixed held-out images and
// the model's probabilities on them at each evaluation, starting with the untrained model.
let screen = { points: [], display: null, speed: null, total: 0 };
let roomState = 'waiting';

function status(message) { $('status').textContent = message; }
const canRun = worker => worker.ready && worker.tasks.includes(selected.taskId);
function refreshControls() {
  const running = roomState === 'training';
  $('start').disabled = !connected || running || !workers.some(canRun);
  $('stop').disabled = !running;
  for (const id of ['task', 'rounds', 'batch-size', 'local-steps', 'target', 'time-limit']) $(id).disabled = running;
}
const TIMING_IDS = { roundMs: 'round-time', computeMs: 'compute-time', networkMs: 'network-time', coordinatorMs: 'coordinator-time' };
const readyPhones = () => workers.filter(canRun).length;
const correctCount = evaluation => evaluation.filter((probabilities, i) => winner(probabilities) === screen.display.examples[i].label).length;

// The room's measured training speed: examples per second over the last 3 rounds,
// skipping round 1 (downloads and GPU warm-up).
function roomSpeed(history) {
  const recent = history.filter(point => point.round > 1).slice(-4);
  if (recent.length < 2) return null;
  const examples = recent.slice(1).reduce((sum, point) => sum + point.updates.reduce((total, update) => total + update.count, 0), 0);
  return examples / ((recent.at(-1).elapsedMs - recent[0].elapsedMs) / 1000);
}

// One square per phone that can train, in join order. Phones that leave fade out.
const cells = new Map();
function showGrid() {
  const ids = workers.filter(canRun).map(worker => worker.id);
  for (const [id, cell] of cells) {
    if (ids.includes(id)) continue;
    cells.delete(id);
    cell.classList.add('gone');
    setTimeout(() => cell.remove(), 600);
  }
  for (const id of ids) {
    if (cells.has(id)) continue;
    const cell = document.createElement('i');
    cells.set(id, cell);
    $('phone-grid').append(cell);
  }
  // Smaller squares for bigger rooms, so 500 phones still fit in a few rows.
  $('phone-grid').style.setProperty('--cell', `${ids.length > 150 ? 4 : ids.length > 50 ? 6 : 9}px`);
}
// Each phone's square flashes when its update lands, earlier for phones that computed faster.
function flashGrid({ round, updates }) {
  const slowest = Math.max(...updates.map(update => update.computeMs));
  for (const { workerId, computeMs } of updates) {
    const cell = cells.get(workerId);
    if (!cell) continue;
    cell.classList.remove('training', 'dropped', 'landed-a', 'landed-b');
    cell.style.animationDelay = `${Math.round(computeMs / slowest * 300)}ms`;
    cell.classList.add(round % 2 ? 'landed-a' : 'landed-b'); // Alternating restarts the animation.
  }
}

function showRoom() {
  const phones = readyPhones();
  // Phones that joined but can't train (no WebGPU, or an outdated page).
  const cantJoin = workers.filter(worker => worker.ready && !canRun(worker)).length;
  $('cant-join').textContent = cantJoin ? `${cantJoin} more can't join (browser)` : '';
  if (String(phones) !== $('room-workers').textContent) bump($('room-workers'));
  $('header-workers').textContent = $('room-workers').textContent = phones;
  $('premise-text').textContent = `phone${phones === 1 ? '' : 's'} ${roomState === 'training' ? 'training one shared model' : 'ready'}`;
  $('room-status').textContent = { waiting: phones ? 'Ready' : 'Waiting for phones', training: 'Training', done: 'Done' }[roomState];
  scheduleStatus();
}

function showScreen({ done = false } = {}) {
  const latest = screen.points.at(-1);
  $('room-loss').textContent = latest ? latest.loss.toFixed(3) : '—';
  $('room-accuracy').textContent = latest?.accuracy === undefined ? '—' : `${(latest.accuracy * 100).toFixed(1)}%`;
  $('room-elapsed').textContent = latest ? clock(latest.seconds) : '—';
  $('room-step').textContent = latest ? latest.round : '—';
  $('comparison-time').textContent = latest ? `round ${latest.round} · t=${latest.seconds.toFixed(1)}s` : '';
  const display = screen.display;
  $('coordinator').classList.toggle('run-complete', done);
  if (display?.history.length) {
    const { examples, history, classes } = display;
    const total = examples.length;
    const correct = correctCount(history.at(-1));
    [...$('predictions').children].forEach((card, image) =>
      updateCard(card, history.at(-1)[image], guessChanges(history, image), classes, { done }));
    $('predictions-panel').classList.toggle('has-errors', correct < total);
    $('correct-count').textContent = `${correct} / ${total} correct`;
    if (done) $('prediction-title').replaceChildren(resultSummary(correctCount(history[0]), correct, total));
    $('evaluation-count').textContent = `${history.length} evals`;
  }
  // Fit the charts above the control bar (after the cards above them are filled in).
  // Positions are in screen pixels, sizes in zoomed ones.
  const zoom = Number(getComputedStyle(document.body).zoom) || 1;
  const space = (innerHeight - $('loss-chart').getBoundingClientRect().top - scrollY) / zoom - 72;
  const lossHeight = drawLossChart($('loss-chart'), screen.points, display?.examples.length, space);
  if (display?.history.length) {
    drawProbabilityHistory($('probability-chart'), display.history, display.examples.map(example => example.label), display.classes, lossHeight);
  }
}

// Shows the task's fixed held-out images with the (untrained) model's guesses, or hides the
// prediction panels for tasks without images.
async function resetScreen() {
  screen = { points: [], display: null, speed: null, total: 0 };
  $('room-speed').textContent = '—';
  delete $('room-speed').dataset.value;
  $('room-total').textContent = 'examples per second, all phones together';
  for (const cell of cells.values()) cell.className = '';
  $('room-task').textContent = selected.classes ? `${selected.name} · ${selected.classes.length} classes` : selected.name;
  $('prediction-title').replaceChildren('Test predictions ', Object.assign(document.createElement('span'), {
    className: 'prediction-note', textContent: 'Fixed held-out images · class probabilities',
  }));
  $('loss-note').textContent = selected.displayExamples ? '+ accuracy' : 'held-out test set';
  const current = task;
  const examples = await selected.displayExamples?.();
  const initial = await task.predictDisplay?.();
  if (task !== current) return;
  if (examples && initial) {
    screen.display = { examples, classes: selected.classes, history: [initial] };
    $('predictions').replaceChildren(...examples.map(example => createCard({ ...example, url: imageUrl(example.image) }, selected.classes)));
  }
  $('predictions-panel').hidden = $('probability-panel').hidden = !screen.display;
  showScreen();
  scheduleStatus();
}

// Phones show the room's state and the shared model's guesses. Sent at most twice a second,
// since a wave of joins would otherwise send one per join to every phone.
let statusTimer, phoneResult;
function scheduleStatus() {
  statusTimer ??= setTimeout(() => {
    statusTimer = undefined;
    if (socket?.readyState !== WebSocket.OPEN) return;
    const history = screen.display?.history;
    send({
      type: 'status', state: roomState, phones: readyPhones(), taskId: selected.taskId, result: phoneResult,
      runId: screen.runId, speed: screen.speed, total: screen.total,
      display: history?.at(-1).map((probabilities, image) => ({
        probabilities: probabilities.map(value => Math.round(value * 1000) / 1000),
        guesses: guessChanges(history, image),
      })),
    });
  }, 500);
}

function showProgress(point, history) {
  screen.runId = trainer.active.id;
  screen.speed = roomSpeed(history);
  screen.total += point.updates.reduce((sum, update) => sum + update.count, 0);
  if (screen.speed) countTo($('room-speed'), screen.speed);
  $('room-total').textContent = `examples/s · ${screen.total.toLocaleString('en-US')} examples trained`;
  screen.points.push({ round: point.round, seconds: point.elapsedMs / 1000, loss: point.loss, accuracy: point.accuracy });
  if (screen.display && point.display) {
    // Round 0 is the untrained model, which replaces the preview shown before the run.
    if (point.round === 0) screen.display.history = [];
    screen.display.history.push(point.display);
    screen.points.at(-1).shown = correctCount(point.display);
  }
  showScreen();
  scheduleStatus();
  $('round').textContent = `${point.round} / ${trainer.active.rounds}`;
  $('loss').textContent = point.loss.toPrecision(4);
  $('accuracy').textContent = point.accuracy === undefined ? '—' : `${(point.accuracy * 100).toFixed(1)}%`;
  $('elapsed').textContent = `${(point.elapsedMs / 1000).toFixed(1)} s`;
  // Workers that missed the deadline or failed sit out a few rounds, then try again.
  for (const { workerId, reason } of point.dropped) {
    workerInfo(workerId).error = `sitting out a round: ${reason}`;
    cells.get(workerId)?.classList.add('dropped');
  }
  if (point.dropped.length) showWorkers();
  const timing = timingSummary(history);
  for (const [key, id] of Object.entries(TIMING_IDS)) $(id).textContent = timing ? formatMs(timing[key]) : '—';
}
function showTask() {
  $('task-note').textContent = selected.description;
  $('rounds').value = selected.defaults.rounds;
  $('target').value = selected.defaults.targetAccuracy ? Math.round(selected.defaults.targetAccuracy * 100) : '';
  $('time-limit').value = selected.defaults.timeLimitSeconds ?? '';
  $('batch-size').value = selected.defaults.batchSize;
  $('local-steps').value = selected.defaults.localSteps;
}
function resetModel() {
  task?.dispose();
  task = selected.createTask();
  trainer = new SharedTrainer(task, { send, onProgress: showProgress });
  for (const id of ['round', 'loss', 'accuracy', 'elapsed', ...Object.values(TIMING_IDS)]) $(id).textContent = '—';
  return resetScreen();
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
  return ['ready', info.activity ?? 'ready'];
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
  for (const worker of workers) {
    const info = workerInfo(worker.id);
    if (canRun(worker) && !info.announced) {
      info.announced = true;
      $('latest-join').textContent = `Phone ${String(info.number).padStart(2, '0')} joined`;
    }
    Object.assign(info, { device: worker.device, backend: worker.backend });
  }
  $('worker-count').textContent = workers.length;
  $('workers').replaceChildren(...[...infos].map(([id, info]) => card(id, info)));
  if (!infos.size) $('workers').textContent = 'No workers yet. Open a worker tab or the phone link.';
  // Workers that disconnect mid-run just leave it; new ones join from the next round.
  for (const id of [...trainer?.active?.workerIds ?? []]) {
    if (!workers.some(worker => worker.id === id && worker.ready)) trainer.removeWorker(id);
  }
  for (const worker of workers.filter(canRun)) trainer?.addWorker(worker.id);
  refreshControls();
  showGrid();
  showRoom();
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
    $('saved').textContent = 'Not saved. Open the laptop page at http://127.0.0.1:4173/?role=laptop to save runs.';
  }
}

// ---- Worker (phone) ----

function phoneState(label, { inactive = false } = {}) {
  $('worker-state').textContent = label;
  $('worker').classList.toggle('worker-inactive', inactive);
}

// The laptop's status: room state, phone count, and the model's guess on one shown image
// (picked by join order, so neighbors see different ones).
let phoneExample;
async function showStatus({ state, phones, taskId, result, display, runId, speed, total }) {
  if (!socket || socket.readyState !== WebSocket.OPEN || sittingOut) return;
  phoneState({ waiting: 'Waiting', training: 'Training', done: 'Done' }[state], { inactive: state !== 'training' });
  $('phone-team').hidden = false;
  if (String(phones) !== $('phone-workers').textContent) bump($('phone-workers'));
  $('phone-workers').textContent = phones;
  $('phone-room-speed').textContent = speed ? ` · ${Math.round(speed).toLocaleString('en-US')} examples/s together` : '';
  const mine = runId && workerRun?.id === runId ? workerRun.examples : 0;
  $('phone-share').textContent = mine && total
    ? `${(Math.min(mine / total, 1) * 100).toFixed(1)}% of the room's ${total.toLocaleString('en-US')} examples` : '';
  $('phone-team-text').textContent = `phone${phones === 1 ? '' : 's'} · one shared model`;
  $('phone-summary').hidden = state !== 'done' || !result;
  $('phone-summary').textContent = result ?? '';
  if (state === 'training') $('worker-status').textContent = 'Keep this page open and your screen on.';
  else if (state === 'waiting') $('worker-status').textContent = 'Waiting for the laptop to start a run.';
  const module = tasks.find(candidate => candidate.taskId === taskId);
  if (!display || !module?.displayExamples) {
    $('phone-prediction').hidden = true;
    return;
  }
  const index = joinNumber % display.length;
  if (phoneExample?.taskId !== taskId) {
    const example = (await module.displayExamples())[index];
    phoneExample = { taskId, label: example.label };
    $('phone-image').src = imageUrl(example.image);
    $('phone-image').alt = `Held-out test image: ${module.classes[example.label]}`;
    $('phone-actual').textContent = `Actual: ${module.classes[example.label]}`;
    $('phone-prediction').dataset.label = example.label;
  }
  updateCard($('phone-prediction'), display[index].probabilities, display[index].guesses, module.classes, { done: state === 'done' });
  $('phone-prediction').hidden = false;
}

// This phone's measured speed: examples per second of GPU compute, each round.
const rateSamples = [];
function showRate(examples, computeMs) {
  const seconds = performance.now() / 1000;
  const rate = examples / (computeMs / 1000);
  rateSamples.push({ seconds, rate });
  while (rateSamples[0].seconds < seconds - 30) rateSamples.shift();
  $('phone-rate').hidden = false;
  $('phone-rate-value').textContent = `${Math.round(rate).toLocaleString('en-US')} examples/s`;
  drawRateSparkline($('phone-rate-chart'), rateSamples, seconds);
}

// The current round's stage on the phone: receive → train → send.
function showPhase(phase, detail) {
  $('phone-live').hidden = false;
  for (const item of $('phone-live').querySelectorAll('[data-phase]')) {
    item.classList.toggle('active', item.dataset.phase === phase);
  }
  $('phone-detail').textContent = detail;
}

// This phone's GPU, e.g. "iPhone · Apple GPU · WebGPU" (browsers may hide the vendor).
async function gpuLabel() {
  const vendor = backend === 'webgpu' ? (await navigator.gpu.requestAdapter())?.info?.vendor : null;
  return [deviceName(), vendor && `${vendor[0].toUpperCase()}${vendor.slice(1)} GPU`, backendLabel(backend)].filter(Boolean).join(' · ');
}

// Workers create each task the first time the coordinator asks for it.
const workerTasks = new Map();
let joinNumber; // From the relay: this worker's place in join order.
let sittingOut = false;
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
  $('phone-round').textContent = `Round ${round}`;
  showPhase('receive', 'Got the shared model');
  const weights = unpackValues(message.weights, task.variables);
  showPhase('train', `Training ${localSteps} steps on ${count * localSteps} images…`);
  const result = await trainLocally(task, workerRun.optimizer, weights, message);
  const memory = memoryUsage();
  workerRun.examples += count * localSteps;
  countTo($('examples'), workerRun.examples);
  showRate(count * localSteps, result.computeMs);
  $('compute').textContent = formatMs(result.computeMs);
  $('gpu-memory').textContent = formatBytes(memory.gpuBytes);
  $('tensors').textContent = memory.tensors;
  send({ type: 'update', runId: message.runId, round, count: count * localSteps, memory, ...result, delta: packValues(result.delta) });
  showPhase('send', `Trained ${count * localSteps} images in ${formatMs(result.computeMs)}, sent the change`);
  currentRound = undefined;
}

// Tell the laptop, then leave: after a GPU reset this page can't compute anymore.
function handleGpuReset() {
  const error = 'GPU reset (WebGL context lost). Reload this page to rejoin.';
  $(role === 'worker' ? 'worker-status' : 'status').textContent = error;
  if (role === 'worker') phoneState('Stopped', { inactive: true });
  if (socket.readyState === WebSocket.OPEN) {
    send({ type: 'worker-error', runId: currentRound?.runId, round: currentRound?.round, error });
  }
  stayClosed = true;
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
      Object.assign(info, { activity: `round ${message.round} done`, computeMs, memory, error: undefined });
      info.examples += count;
    }
    flashGrid(message);
    trainer?.receive(message);
    showWorkers();
  } else if (message.type === 'worker-error') {
    workerInfo(message.workerId).error = message.error;
    showWorkers();
  }
}

// Connects to the relay. Phones call this again after losing the connection.
let queue = Promise.resolve(); // Round assignments, one at a time.
let stayClosed = false, retryMs = 1000, reconnectTimer;
function connect() {
  socket = new WebSocket(relayUrl);
  socket.addEventListener('open', () => {
    retryMs = 1000;
    send({ type: 'join', role });
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.type === 'welcome') {
      joinNumber = message.joinNumber;
      connected = true;
      $(connectionId).textContent = `Connected · ${deviceName()} · ${backendLabel(backend)}`;
      if (role === 'worker') {
        const note = sitOutReason();
        sittingOut = Boolean(note);
        phoneState(note ? 'Sitting out' : 'Waiting', { inactive: true });
        void gpuLabel().then(label => { $('phone-gpu').textContent = label; });
        $('worker').querySelector('[data-phase=train]').textContent = `Training on ${backend === 'cpu' ? 'CPU' : 'GPU'}`;
        // A worker that sits out offers no tasks, so the laptop never gives it work.
        send({ type: 'ready', backend, device: deviceName(), note, tasks: note ? [] : tasks.map(({ taskId }) => taskId) });
        $('sit-out').hidden = !note;
        $('worker').classList.toggle('sitting-out', Boolean(note));
        if (note) {
          $('sit-out-text').textContent = sitOutAdvice();
          $('worker-status').textContent = 'Sitting out: this browser has no WebGPU.';
        }
      }
      refreshControls();
      if (role === 'coordinator') scheduleStatus();
    } else if (message.type === 'presence') {
      workers = message.workers;
      showWorkers();
    } else if (role === 'worker' && message.type === 'round') {
      // Serialize assignments so a late old round cannot mutate a newer round's model.
      queue = queue.then(() => work(message)).catch(error => {
        if (socket.readyState !== WebSocket.OPEN) return; // Reconnecting: this round is lost anyway.
        $('worker-status').textContent = `Failed: ${error.message}`;
        send({ type: 'worker-error', runId: message.runId, round: message.round, error: error.message });
      });
    } else if (role === 'worker' && message.type === 'status') {
      void showStatus(message);
    } else if (message.type === 'coordinator-left') {
      if (!sittingOut) phoneState('Waiting', { inactive: true });
      $('worker-status').textContent = 'Laptop reconnecting… Keep this page open.';
    } else if (message.type === 'replaced') {
      // Another laptop tab took over (e.g. this page was opened again). Don't fight it.
      stayClosed = true;
      trainer?.stop('Another laptop tab took over.');
      status('Another laptop tab took over. Close this one.');
      $(connectionId).textContent = 'Another laptop tab took over. Close this one.';
      socket.close();
    } else if (message.type === 'error') {
      trainer?.stop(message.error);
      $(connectionId).textContent = `Relay error: ${message.error}`;
    } else {
      receive(message);
    }
  });
  socket.addEventListener('close', () => {
    connected = false;
    trainer?.stop('Relay disconnected.');
    refreshControls();
    // Reconnect by ourselves (wifi blips, locked screens): phones rejoin from the next round,
    // and the laptop can start a new run. Not after Disconnect, a GPU reset, or a takeover.
    if (!stayClosed) {
      if (role === 'worker') phoneState('Reconnecting', { inactive: true });
      else {
        status('Connection lost. Reconnecting…');
        $('room-status').textContent = 'Reconnecting…';
      }
      $(connectionId).textContent = 'Connection lost. Reconnecting…';
      reconnectTimer = setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, 10_000);
      return;
    }
    if (role === 'worker') {
      $(connectionId).textContent = 'Disconnected. Reload this page to reconnect.';
      phoneState('Disconnected', { inactive: true });
      $('disconnect').disabled = true;
    }
  });
  socket.addEventListener('error', () => {
    $(connectionId).textContent = 'Connection failed. Is npm start running?';
  });
}

async function initialize() {
  backend = await initializeBackend(role === 'worker' ? preferredBackend : 'webgl');
  onGpuReset(handleGpuReset);
  if (role === 'worker') {
    void keepScreenOn();
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) return;
      void keepScreenOn();
      if (reconnectTimer && socket.readyState === WebSocket.CLOSED) {
        clearTimeout(reconnectTimer);
        connect();
      }
    });
    document.addEventListener('pointerdown', () => void keepScreenOn());
  }
  if (role === 'coordinator') {
    showTask();
    void resetModel();
  }
  connect();
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
  const timeLimitMs = $('time-limit').value === '' ? null : Number($('time-limit').value) * 1000;
  if (timeLimitMs !== null && !(timeLimitMs > 0)) {
    status('Stop after must be a positive number of seconds, or empty.');
    return;
  }
  // Fresh run: forget workers that left and clear last run's numbers.
  for (const [id, info] of infos) {
    if (!workers.some(worker => worker.id === id)) infos.delete(id);
    else Object.assign(info, { examples: 0, activity: undefined, error: undefined });
  }
  const participants = workers.filter(canRun);
  roomState = 'training';
  phoneResult = undefined;
  await resetModel();
  $('saved').textContent = '';
  $('result').hidden = $('run-result').hidden = true;
  showWorkers();
  status(`Training one shared model with ${participants.length} worker(s)…`);
  try {
    const run = await trainer.train({
      workerIds: participants.map(worker => worker.id), rounds, batchSize, localSteps,
      optimizer: selected.defaults.optimizer, learningRate: selected.defaults.learningRate,
      outerLearningRate: selected.defaults.outerLearningRate, outerMomentum: selected.defaults.outerMomentum,
      targetAccuracy: target, timeLimitMs,
    });
    const outcome = { completed: 'Completed', stopped: 'Stopped', failed: 'Failed' }[run.outcome];
    status(`${outcome}: ${run.history.at(-1)?.round ?? 0} of ${rounds} rounds.${run.reason ? ' ' + run.reason : ''}`);
    const ending = run.reachedTarget ?? run.timeUp;
    if (ending) {
      const { accuracy, elapsedMs, workers: count } = ending;
      const [accuracyText, seconds, phones] = [(accuracy * 100).toFixed(1), (elapsedMs / 1000).toFixed(1), `${count} phone${count === 1 ? '' : 's'}`];
      $('result').textContent = $('run-result').textContent = run.reachedTarget
        ? `Reached ${accuracyText}% held-out accuracy in ${seconds} s with ${phones}`
        : `Time's up: ${accuracyText}% held-out accuracy after ${seconds} s with ${phones}`;
      // Phones get a short version.
      phoneResult = run.reachedTarget ? `${accuracyText}% accuracy in ${seconds} s` : `Time's up: ${accuracyText}% accuracy`;
      $('result').hidden = $('run-result').hidden = false;
    }
    roomState = 'done';
    showScreen({ done: true });
    await saveRun({
      ...run, timing: timingSummary(run.history), workers: run.participants.map(snapshot),
      coordinatorBackend: backend, userAgent: navigator.userAgent,
    });
  } catch (error) {
    status(`Cannot start: ${error.message}`);
  } finally {
    if (roomState === 'training') roomState = 'waiting';
    refreshControls();
    showRoom();
  }
});
$('task').addEventListener('change', () => {
  selected = tasks.find(({ taskId }) => taskId === $('task').value);
  roomState = 'waiting';
  $('run-result').hidden = true;
  showTask();
  void resetModel();
  showWorkers();
  status(`Selected: ${selected.name}.`);
});
$('stop').addEventListener('click', () => trainer.stop());
$('disconnect').addEventListener('click', () => {
  $('worker-status').textContent = 'Disconnected. Reload this page to rejoin.';
  stayClosed = true;
  socket?.close();
});
// Redraw when the layout changes: window resized, or web fonts finished loading.
const redraw = () => { if (role === 'coordinator') showScreen({ done: roomState === 'done' }); };
addEventListener('resize', redraw);
void document.fonts.ready.then(redraw);
void initialize().catch(error => {
  $(connectionId).textContent = `Initialization failed: ${error.message}`;
});
