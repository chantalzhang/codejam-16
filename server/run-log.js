import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const MAX_BYTES = 10 * 1024 * 1024;

// POST /api/runs saves a finished run as JSON, so results don't need copying by hand.
// Only the laptop's own page may save: requests through the tunnel carry its public
// hostname, and the Origin check blocks other websites open on the laptop.
export function runLog(dir) {
  return (request, response) => {
    const host = request.headers.host ?? '';
    const local = /^(127\.0\.0\.1|localhost)(:\d+)?$/.test(host);
    if (request.method !== 'POST' || !local || request.headers.origin !== `http://${host}`) {
      response.statusCode = 403;
      response.end();
      return;
    }
    let body = '';
    request.setEncoding('utf8');
    request.on('data', chunk => {
      body += chunk;
      if (body.length > MAX_BYTES) request.destroy();
    });
    request.on('end', async () => {
      try {
        const run = JSON.parse(body);
        const task = String(run.taskId ?? 'run').replace(/[^\w-]/g, '');
        const file = join(dir, `${new Date().toISOString().replace(/[:.]/g, '-')}-${task}.json`);
        await mkdir(dir, { recursive: true });
        await writeFile(file, JSON.stringify(run, null, 2) + '\n');
        response.setHeader('content-type', 'application/json');
        response.end(JSON.stringify({ file }));
      } catch {
        response.statusCode = 400;
        response.end();
      }
    });
  };
}
