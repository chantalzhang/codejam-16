import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { createServer } from 'vite';

// Saved runs go to a temporary folder, not the project's runs/.
process.env.RUNS_DIR = await mkdtemp(join(tmpdir(), 'swarm-runs-'));
const server = await createServer({ server: { host: '127.0.0.1', port: 0 } });
await server.listen();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1200, height: 1100 } });
  page.on('pageerror', error => errors.push(error.message));
  const base = server.resolvedUrls.local[0];
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent.startsWith('Connected'));
  assert.equal(await page.locator('#role-title').textContent(), 'Shared training · laptop');
  assert.equal(await page.locator('#start').isDisabled(), true);
  assert.equal(await page.locator('#worker-count').textContent(), '0');
  assert.equal(await page.locator('label').filter({ has: page.locator('#batch-size') }).textContent(), 'Examples per step ');
  assert.equal(await page.locator('#task').inputValue(), 'image-benchmark-v1');
  await page.locator('#task').selectOption('linear-function-v1');

  async function addWorker() {
    const worker = await browser.newPage();
    worker.on('pageerror', error => errors.push(error.message));
    await worker.goto(base + '?role=worker');
    await worker.waitForFunction(() => document.querySelector('#connection').textContent.startsWith('Connected'));
    assert.equal(await worker.locator('#role-title').textContent(), 'Shared training · worker');
    return worker;
  }
  // Cards of workers that left stay visible (faded), so only count connected ones.
  async function readyWorkers(count) {
    await page.waitForFunction(expected => {
      const cards = [...document.querySelectorAll('#workers .card:not(.left)')];
      return cards.length === expected && cards.every(card => card.classList.contains('ready'));
    }, count);
  }
  // Every finished run is saved automatically; read the newest one.
  let lastSaved = '';
  async function savedRun() {
    await page.waitForFunction(previous => {
      const text = document.querySelector('#saved').textContent;
      return text.startsWith('Saved: ') && text !== previous;
    }, lastSaved, { timeout: 60_000 });
    lastSaved = await page.locator('#saved').textContent();
    return JSON.parse(await readFile(lastSaved.slice('Saved: '.length), 'utf8'));
  }
  async function run(steps = 40, batchSize = 17) {
    await page.locator('#steps').fill(String(steps));
    await page.locator('#batch-size').fill(String(batchSize));
    await page.locator('#start').click();
    const result = await savedRun();
    assert.equal(result.outcome, 'completed', result.reason);
    return result;
  }

  const first = await addWorker();
  await readyWorkers(1);
  const one = await run();
  assert.match(await first.locator('#gpu-memory').textContent(), /^\d+ KB$|^\d+\.\d MB$/);
  const tensorsAfterOne = await first.locator('#tensors').textContent();
  await addWorker();
  const third = await addWorker();
  await readyWorkers(3);
  assert.equal(await page.locator('#worker-count').textContent(), '3');
  const three = await run();
  assert.equal(await first.locator('#tensors').textContent(), tensorsAfterOne, 'Worker leaked tensors between runs');

  // The laptop shows each worker's own numbers, matching what the worker page shows.
  const cards = page.locator('#workers .card');
  assert.equal(await cards.count(), 3);
  assert.match(await cards.first().locator('h3').textContent(), /^Worker 1 · Mac · WebGL$/);
  const cardTensors = await cards.first().locator('dt:text("Tensors") + dd').textContent();
  assert.equal(cardTensors, tensorsAfterOne);
  assert.equal(three.workers.reduce((sum, worker) => sum + worker.examples, 0), 40 * 17);
  assert.ok(three.workers.every(worker => worker.backend === 'webgl' && worker.device === 'Mac'));

  assert.equal(three.history.length, 41);
  assert.equal(three.timing.steps, 39);
  assert.ok(Object.values(three.timing).every(Number.isFinite));
  assert.match(await page.locator('#network-time').textContent(), /^-?\d+ ms$/);
  assert.ok(three.history.at(-1).loss < 0.01);
  for (const point of three.history.slice(1)) {
    assert.equal(point.updates.length, 3);
    assert.equal(point.updates.reduce((sum, update) => sum + update.count, 0), 17);
  }
  for (let i = 0; i < one.history.length; i++) {
    assert.ok(Math.abs(one.history[i].loss - three.history[i].loss) < 1e-5, 'Worker count changed the training trajectory');
  }

  await page.locator('#steps').fill('10000');
  await page.locator('#start').click();
  await page.waitForFunction(() => /^\d+ \//.test(document.querySelector('#step').textContent) && parseInt(document.querySelector('#step').textContent) >= 1);
  await page.locator('#stop').click();
  assert.equal((await savedRun()).outcome, 'stopped');

  await page.locator('#start').click();
  await page.waitForFunction(() => parseInt(document.querySelector('#step').textContent) >= 1);
  await third.close();
  const disconnected = await savedRun();
  assert.equal(disconnected.outcome, 'stopped');
  assert.match(disconnected.reason, /disconnected/);
  await readyWorkers(2);
  assert.equal(await page.locator('#workers .card.left').count(), 1, 'A worker that left stays visible until the next run');
  await run(3); // Old in-flight gradients must not contaminate this new run.
  assert.equal(await page.locator('#workers .card.left').count(), 0);

  // Small image-task run on software WebGL: checks the CNN path and timing breakdown.
  await page.locator('#task').selectOption('image-benchmark-v1');
  const image = await run(3, 8);
  assert.equal(image.taskId, 'image-benchmark-v1');
  assert.equal(image.timing.steps, 2);
  assert.ok(Object.values(image.timing).every(Number.isFinite));

  // Simulate a phone GPU reset mid-run: the run must end quickly, and the laptop must say why.
  await page.locator('#task').selectOption('linear-function-v1');
  await page.locator('#steps').fill('10000');
  await page.locator('#start').click();
  await page.waitForFunction(() => parseInt(document.querySelector('#step').textContent) >= 1);
  const resetAt = Date.now();
  await first.evaluate(async () => {
    // Import the exact module URL the page loaded, so this is the page's TF.js instance.
    const url = performance.getEntriesByType('resource').map(entry => entry.name)
      .find(name => name.includes('@tensorflow_tfjs.js'));
    const tf = await import(url);
    tf.backend().gpgpu.gl.getExtension('WEBGL_lose_context').loseContext();
  });
  await first.waitForFunction(() => /GPU reset/.test(document.querySelector('#worker-status').textContent));
  const gpuReset = await savedRun();
  assert.ok(Date.now() - resetAt < 5_000, 'Run did not end promptly after a GPU reset');
  // Mid-step resets report "GPU reset"; resets between steps show up as a disconnect.
  assert.notEqual(gpuReset.outcome, 'completed');
  assert.match(gpuReset.reason, /GPU reset|disconnected/);
  await page.waitForFunction(() => /GPU reset/.test(document.querySelector('#workers .card.left .state')?.textContent));
  await readyWorkers(1);
  await page.screenshot({ path: join(process.env.RUNS_DIR, 'laptop.png'), fullPage: true });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    environment: 'Headless Chromium / software WebGL. Correctness check, not a phone-speed benchmark.',
    steps: 40, fixedGlobalBatchSize: 17,
    initialLoss: three.history[0].loss,
    oneWorkerFinalLoss: one.history.at(-1).loss,
    threeWorkerFinalLoss: three.history.at(-1).loss,
    simulatedGpuReset: gpuReset.reason,
    screenshot: join(process.env.RUNS_DIR, 'laptop.png'),
    checks: 'Worker cards, saved runs, equivalence, stop, disconnect, GPU reset, and image task passed.',
  }, null, 2));
} finally {
  await browser?.close();
  await server.close();
}
