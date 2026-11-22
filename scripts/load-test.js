// Load test: fake phones against the real laptop page, to find where rounds slow down.
// Runs everything on this machine with software rendering only (never the real GPU), so it
// measures the relay and laptop page, not your internet connection. For that, run
// scripts/fake-phones.js from another computer through the tunnel.
//
// Usage: node scripts/load-test.js [counts, default 1,10,30,100,200] [--rounds 15]
import { fork } from 'node:child_process';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { rounds: { type: 'string', default: '15' } },
});
const counts = (positionals[0] ?? '1,10,30,100,200').split(',').map(Number);

process.env.RUNS_DIR = await mkdtemp(join(tmpdir(), 'load-test-'));
const server = await createServer({ server: { host: '127.0.0.1', port: 0 }, logLevel: 'error' });
await server.listen();
const base = server.resolvedUrls.local[0];
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const results = [];
try {
  const page = await browser.newPage();
  await page.goto(`${base}?role=laptop`);
  await page.waitForFunction(() => document.querySelector('#connection').textContent.startsWith('Connected'));
  await page.locator('#task').selectOption('fashion-mnist-v1');
  await page.locator('#rounds').fill(values.rounds);
  for (const count of counts) {
    const phones = fork('scripts/fake-phones.js', [String(count), `${base.replace('http', 'ws')}swarm`], { stdio: 'ignore' });
    await page.waitForFunction(n => document.querySelectorAll('#workers .card.ready').length === n, count, { timeout: 120_000 });
    const previous = await page.locator('#saved').textContent();
    await page.locator('#start').click();
    await page.waitForFunction(text => document.querySelector('#saved').textContent !== text &&
      document.querySelector('#saved').textContent.startsWith('Saved'), previous, { timeout: 600_000 });
    const run = JSON.parse(await readFile((await page.locator('#saved').textContent()).slice('Saved: '.length), 'utf8'));
    const { roundMs, networkMs, coordinatorMs } = run.timing;
    results.push({ phones: count, outcome: run.outcome, roundMs, networkMs, coordinatorMs });
    console.log(`${String(count).padStart(4)} phones: ${Math.round(roundMs)} ms per round ` +
      `(network ${Math.round(networkMs)} ms, laptop ${Math.round(coordinatorMs)} ms) ${run.outcome}`);
    phones.kill();
    await page.waitForFunction(() => document.querySelector('#worker-count').textContent === '0', null, { timeout: 60_000 });
  }
} finally {
  await browser.close();
  await server.close();
}
const baseline = results[0]?.roundMs;
console.log('\nphones | ms per round | vs 1 phone');
for (const { phones, roundMs } of results) console.log(`${String(phones).padStart(6)} | ${String(Math.round(roundMs)).padStart(12)} | ${(roundMs / baseline).toFixed(1)}x`);
