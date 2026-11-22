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
    // Software WebGL and WebGPU only: tests never touch the real GPU.
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'],
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
  assert.equal(await page.locator('label').filter({ has: page.locator('#batch-size') }).textContent(), 'Images per phone per step ');
  assert.equal(await page.locator('#task').inputValue(), 'fashion-mnist-v1');
  await page.locator('#task').selectOption('linear-function-v1');

  async function addWorker(backend = 'webgl') {
    const worker = await browser.newPage();
    worker.on('pageerror', error => errors.push(error.message));
    await worker.goto(`${base}?role=worker&backend=${backend}`);
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
  async function run(rounds = 40, batchSize = 17, localSteps = 1) {
    await page.locator('#rounds').fill(String(rounds));
    await page.locator('#batch-size').fill(String(batchSize));
    await page.locator('#local-steps').fill(String(localSteps));
    await page.locator('#start').click();
    const result = await savedRun();
    assert.equal(result.outcome, 'completed', result.reason);
    return result;
  }

  const first = await addWorker();
  await readyWorkers(1);
  const one = await run(40, 3 * 17); // Same examples per round as 3 workers × 17 below.
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
  assert.equal(three.workers.reduce((sum, worker) => sum + worker.examples, 0), 40 * 3 * 17);
  assert.ok(three.workers.every(worker => worker.backend === 'webgl' && worker.device === 'Mac'));

  assert.equal(three.history.length, 41);
  assert.equal(three.timing.rounds, 39);
  assert.ok(Object.values(three.timing).every(Number.isFinite));
  assert.match(await page.locator('#network-time').textContent(), /^-?\d+ ms$/);
  assert.ok(three.history.at(-1).loss < 0.01);
  for (const point of three.history.slice(1)) {
    assert.equal(point.updates.length, 3);
    assert.equal(point.updates.reduce((sum, update) => sum + update.count, 0), 3 * 17);
  }
  for (let i = 0; i < one.history.length; i++) {
    assert.ok(Math.abs(one.history[i].loss - three.history[i].loss) < 1e-5, 'Worker count changed the training trajectory');
  }

  await page.locator('#rounds').fill('10000');
  await page.locator('#start').click();
  await page.waitForFunction(() => /^\d+ \//.test(document.querySelector('#round').textContent) && parseInt(document.querySelector('#round').textContent) >= 1);
  await page.locator('#stop').click();
  assert.equal((await savedRun()).outcome, 'stopped');

  // A worker closing its tab mid-run just leaves; the others keep training.
  const roundNow = () => page.evaluate(() => parseInt(document.querySelector('#round').textContent));
  await page.locator('#start').click();
  await page.waitForFunction(() => parseInt(document.querySelector('#round').textContent) >= 1);
  await third.close();
  const roundAtLeave = await roundNow();
  await page.waitForFunction(start => parseInt(document.querySelector('#round').textContent) >= start + 5, roundAtLeave);
  await page.locator('#stop').click();
  const leftRun = await savedRun();
  assert.equal(leftRun.outcome, 'stopped');
  assert.ok(leftRun.history.slice(-3).every(point => point.updates.length === 2));
  await readyWorkers(2);
  assert.equal(await page.locator('#workers .card.left').count(), 1, 'A worker that left stays visible until the next run');
  await run(3); // Old in-flight updates must not contaminate this new run.
  assert.equal(await page.locator('#workers .card.left').count(), 0);

  // Small image-task run on software WebGL: checks the CNN path and timing breakdown.
  await page.locator('#task').selectOption('image-benchmark-v1');
  const image = await run(3, 8);
  assert.equal(image.taskId, 'image-benchmark-v1');

  assert.equal(image.timing.rounds, 2);
  assert.ok(Object.values(image.timing).every(Number.isFinite));

  // Short Fashion-MNIST run with local SGD: Adam, 2 local steps per round, accuracy reported.
  await page.locator('#task').selectOption('fashion-mnist-v1');
  const fashion = await run(3, 8, 2);
  assert.equal(fashion.optimizer, 'adam');
  assert.deepEqual([fashion.outerLearningRate, fashion.outerMomentum], [0.5, 0.9]);
  assert.ok(fashion.history.slice(1).every(point => point.updates.every(update => update.count === 8 * 2)));
  assert.ok(fashion.history.every(point => point.accuracy >= 0 && point.accuracy <= 1));
  assert.match(await page.locator('#accuracy').textContent(), /^\d+\.\d%$/);

  // Simulate a phone GPU reset mid-run: that worker drops out and the run keeps going.
  await page.locator('#task').selectOption('linear-function-v1');
  await page.locator('#rounds').fill('10000');
  await page.locator('#start').click();
  await page.waitForFunction(() => parseInt(document.querySelector('#round').textContent) >= 1);
  await first.evaluate(async () => {
    // Import the exact module URL the page loaded, so this is the page's TF.js instance.
    const url = performance.getEntriesByType('resource').map(entry => entry.name)
      .find(name => name.includes('@tensorflow_tfjs.js'));
    const tf = await import(url);
    tf.backend().gpgpu.gl.getExtension('WEBGL_lose_context').loseContext();
  });
  await first.waitForFunction(() => /GPU reset/.test(document.querySelector('#worker-status').textContent));
  const roundAtReset = await roundNow();
  await page.waitForFunction(start => parseInt(document.querySelector('#round').textContent) >= start + 5, roundAtReset);
  await page.locator('#stop').click();
  const gpuReset = await savedRun();
  assert.equal(gpuReset.outcome, 'stopped');
  assert.ok(gpuReset.history.slice(-3).every(point => point.updates.length === 1));
  await page.waitForFunction(() => /GPU reset/.test(document.querySelector('#workers .card.left .state')?.textContent));
  await readyWorkers(1);

  // A WebGPU worker (what iPhones use) trains alongside a WebGL one.
  const gpuWorker = await addWorker('webgpu');
  assert.match(await gpuWorker.locator('#connection').textContent(), /WebGPU$/);
  await readyWorkers(2);
  await page.locator('#task').selectOption('fashion-mnist-v1');
  const mixed = await run(2, 8, 2);
  assert.deepEqual(mixed.workers.map(worker => worker.backend).sort(), ['webgl', 'webgpu']);
  await page.screenshot({ path: join(process.env.RUNS_DIR, 'laptop.png'), fullPage: true });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    environment: 'Headless Chromium / software WebGL. Correctness check, not a phone-speed benchmark.',
    rounds: 40, imagesPerWorkerPerStep: 17,
    initialLoss: three.history[0].loss,
    oneWorkerFinalLoss: one.history.at(-1).loss,
    threeWorkerFinalLoss: three.history.at(-1).loss,
    screenshot: join(process.env.RUNS_DIR, 'laptop.png'),
    checks: 'Worker cards, saved runs, equivalence, stop, disconnect, GPU reset, and image task passed.',
  }, null, 2));
} finally {
  await browser?.close();
  await server.close();
}
