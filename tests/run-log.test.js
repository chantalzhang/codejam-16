import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer, request } from 'node:http';
import { mkdtemp, readFile, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runLog } from '../server/run-log.js';

test('run log saves runs posted by the laptop page, and nothing else', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'runs-'));
  const server = createServer(runLog(dir));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const host = `127.0.0.1:${server.address().port}`;
  // node:http, because fetch ignores a custom Host header.
  const post = (headers, body = '{"taskId":"task/../x","outcome":"completed"}') => new Promise((resolve, reject) => {
    const req = request(`http://${host}/`, { method: 'POST', headers: { host, 'content-type': 'application/json', ...headers } }, res => {
      let text = '';
      res.on('data', chunk => { text += chunk; }).on('end', () => resolve({ status: res.statusCode, json: () => JSON.parse(text) }));
    });
    req.on('error', reject);
    req.end(body);
  });

  const saved = await post({ origin: `http://${host}` });
  assert.equal(saved.status, 200);
  const { file } = saved.json();
  assert.equal(join(dir, file.split('/').at(-1)), file, 'Task ID must not escape the runs folder');
  assert.equal(JSON.parse(await readFile(file, 'utf8')).outcome, 'completed');

  // Phones reach the server through the tunnel, which sends the public hostname.
  // Origin matches its Host here, so only the hostname check can reject it.
  assert.equal((await post({ host: 'demo.trycloudflare.com', origin: 'http://demo.trycloudflare.com' })).status, 403);
  assert.equal((await post({ origin: 'https://unrelated.example' })).status, 403);
  assert.equal((await post({ origin: `http://${host}` }, 'not json')).status, 400);
  assert.equal((await readdir(dir)).length, 1);
});
