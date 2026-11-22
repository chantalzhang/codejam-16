// Fake phones for load testing. Each joins as a worker, waits about as long as an iPhone
// takes to train a round, then sends back a real-sized (all-zero) update. No GPU needed.
//
// Usage: node scripts/fake-phones.js <count> [url] [--compute-ms 220]
// e.g.   node scripts/fake-phones.js 200 wss://example.trycloudflare.com/swarm
import { parseArgs } from 'node:util';
import { WebSocket } from 'ws';
import { tasks } from '../src/tasks.js';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { 'compute-ms': { type: 'string', default: '220' } }, // iPhone WebGPU, 10 local steps
});
const count = Number(positionals[0] ?? 10);
const url = new URL(positionals[1] ?? 'ws://127.0.0.1:4173/swarm');
const computeMs = Number(values['compute-ms']);
// The relay only accepts its own page's origin.
const origin = `${url.protocol === 'wss:' ? 'https:' : 'http:'}//${url.host}`;
const taskIds = tasks.map(({ taskId }) => taskId);

// An all-zero update the same size as the model, so messages are real-sized.
const zeros = new Map();
const zeroUpdate = weights => {
  const bytes = Buffer.from(weights, 'base64').length;
  if (!zeros.has(bytes)) zeros.set(bytes, Buffer.alloc(bytes).toString('base64'));
  return zeros.get(bytes);
};

function fakePhone() {
  const socket = new WebSocket(url, { origin });
  const send = message => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify(message));
  socket.on('open', () => send({ type: 'join', role: 'worker' }));
  socket.on('error', error => console.error(`fake phone: ${error.message}`));
  socket.on('message', data => {
    const message = JSON.parse(data);
    if (message.type === 'welcome') send({ type: 'ready', backend: 'webgpu', device: 'Fake phone', tasks: taskIds });
    if (message.type !== 'round') return;
    setTimeout(() => send({
      type: 'update', runId: message.runId, round: message.round, count: message.count * message.localSteps,
      delta: zeroUpdate(message.weights), loss: 1, computeMs, memory: { tensors: 0, gpuBytes: 0 },
    }), computeMs);
  });
  return socket;
}

const phones = Array.from({ length: count }, fakePhone);
console.log(`${count} fake phones connecting to ${url} (${computeMs} ms per round). Ctrl+C to stop.`);
process.on('SIGINT', () => { phones.forEach(phone => phone.terminate()); process.exit(); });
process.on('SIGTERM', () => { phones.forEach(phone => phone.terminate()); process.exit(); });
