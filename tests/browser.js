import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const server = await createServer({ server: { host: '127.0.0.1', port: 0 } });
await server.listen();
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1100, height: 1100 } });
  page.on('pageerror', error => errors.push(error.message));
  const base = server.resolvedUrls.local[0];
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent.startsWith('Connected as coordinator'));
  assert.equal(await page.locator('#start').isDisabled(), true);
  assert.equal(await page.locator('#worker-count').textContent(), '0');
  assert.match(await page.locator('#role-note').textContent(), /one shared model/);
  assert.equal(await page.locator('label').filter({ has: page.locator('#batch-size') }).textContent(), 'Examples per step ');
  assert.equal(await page.locator('#task').inputValue(), 'image-benchmark-v1');
  await page.locator('#task').selectOption('linear-function-v1');

  async function addWorker() {
    const worker = await browser.newPage();
    worker.on('pageerror', error => errors.push(error.message));
    await worker.goto(base + '?role=worker');
    await worker.waitForFunction(() => document.querySelector('#connection').textContent.startsWith('Connected as worker'));
    assert.match(await worker.locator('#role-note').textContent(), /its share of the training examples/);
    return worker;
  }
  async function readyWorkers(count) {
    await page.waitForFunction(expected => {
      const workers = [...document.querySelectorAll('#workers li')];
      return workers.length === expected && workers.every(worker => worker.textContent.endsWith('ready'));
    }, count);
  }
  async function exportRun() {
    const downloaded = page.waitForEvent('download');
    await page.locator('#export').click();
    return JSON.parse(await readFile(await (await downloaded).path(), 'utf8'));
  }
  async function run(steps = 40, batchSize = 17) {
    await page.locator('#steps').fill(String(steps));
    await page.locator('#batch-size').fill(String(batchSize));
    await page.locator('#start').click();
    await page.waitForFunction(() => !document.querySelector('#export').disabled, null, { timeout: 60_000 });
    const result = await exportRun();
    assert.equal(result.outcome, 'completed', result.reason);
    return result;
  }

  await addWorker();
  await readyWorkers(1);
  const one = await run();
  await addWorker();
  const third = await addWorker();
  await readyWorkers(3);
  assert.equal(await page.locator('#worker-count').textContent(), '3');
  const three = await run();
  assert.ok(three.workers.every(worker => worker.backend === 'webgl'));
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
  await page.waitForFunction(() => Number(document.querySelector('#step').textContent) >= 1);
  await page.locator('#stop').click();
  await page.waitForFunction(() => !document.querySelector('#export').disabled);
  assert.equal((await exportRun()).outcome, 'stopped');

  await page.locator('#start').click();
  await page.waitForFunction(() => Number(document.querySelector('#step').textContent) >= 1);
  await third.close();
  await page.waitForFunction(() => !document.querySelector('#export').disabled);
  const disconnected = await exportRun();
  assert.equal(disconnected.outcome, 'stopped');
  assert.match(disconnected.reason, /disconnected/);
  await readyWorkers(2);
  await run(3); // Old in-flight gradients must not contaminate this new run.

  // Small image-task run on software WebGL: checks the CNN path and timing breakdown.
  await page.locator('#task').selectOption('image-benchmark-v1');
  const image = await run(3, 8);
  assert.equal(image.taskId, 'image-benchmark-v1');
  assert.equal(image.timing.steps, 2);
  assert.ok(Object.values(image.timing).every(Number.isFinite));
  assert.match(await page.locator('#network-time').textContent(), /^-?\d+ ms$/);
  await page.locator('#reset').click();
  assert.equal(await page.locator('#step').textContent(), '0');
  assert.equal(await page.locator('#loss').textContent(), '—');
  assert.equal(await page.locator('#export').isDisabled(), true);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    environment: 'Headless Chromium / software WebGL. Correctness check, not a phone-speed benchmark.',
    steps: 40, fixedGlobalBatchSize: 17,
    initialLoss: three.history[0].loss,
    oneWorkerFinalLoss: one.history.at(-1).loss,
    threeWorkerFinalLoss: three.history.at(-1).loss,
    checks: 'One/three-worker equivalence, relay, export, stop, disconnect, restart, and reset passed.',
  }, null, 2));
} finally {
  await browser?.close();
  await server.close();
}
