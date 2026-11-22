import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { WebSocket } from 'ws';
import { attachRelay } from '../server/relay.js';
import { packFloats, unpackFloats } from '../src/packing.js';

async function connect(url, role) {
  const socket = new WebSocket(url.replace('http:', 'ws:') + '/swarm', { origin: url });
  const inbox = [], waiters = [];
  socket.on('message', data => {
    const message = JSON.parse(data);
    const index = waiters.findIndex(waiter => waiter.type === message.type);
    if (index < 0) inbox.push(message);
    else {
      const [waiter] = waiters.splice(index, 1);
      clearTimeout(waiter.timer);
      waiter.resolve(message);
    }
  });
  const next = type => {
    const index = inbox.findIndex(message => message.type === type);
    if (index >= 0) return Promise.resolve(inbox.splice(index, 1)[0]);
    return new Promise((resolve, reject) => {
      const waiter = { type, resolve };
      waiter.timer = setTimeout(() => reject(new Error(`Missing ${type} message`)), 2000);
      waiters.push(waiter);
    });
  };
  await once(socket, 'open');
  const send = message => socket.send(JSON.stringify(message));
  send({ type: 'join', role });
  return { socket, send, next };
}

test('relay fans out rounds, adds up updates, assigns identity, and limits roles', async t => {
  const server = createServer();
  const relay = attachRelay(server);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => {
    relay.close();
    await new Promise(resolve => server.close(resolve));
  });
  const url = `http://127.0.0.1:${server.address().port}`;
  const coordinator = await connect(url, 'coordinator');
  await coordinator.next('welcome');
  assert.deepEqual((await coordinator.next('presence')).workers, []);

  const worker = await connect(url, 'worker');
  const welcome = await worker.next('welcome');
  await coordinator.next('presence');
  worker.send({ type: 'ready', backend: 'webgl', device: 'iPhone', note: 'sitting out', tasks: ['fixture'] });
  const presence = await coordinator.next('presence');
  assert.equal(presence.workers[0].id, welcome.id);
  assert.equal(presence.workers[0].ready, true);
  assert.equal(presence.workers[0].backend, 'webgl');
  assert.deepEqual(presence.workers[0].tasks, ['fixture']);
  assert.equal(presence.workers[0].device, 'iPhone');
  assert.equal(presence.workers[0].note, 'sitting out');

  // One round message fans out: each assigned worker gets the model plus its own slice.
  const weights = packFloats(new Float32Array([0, 0]));
  coordinator.send({ type: 'round', runId: 'run', round: 1, localSteps: 1, deadlineMs: 5000, weights, assignments: [
    { workerId: welcome.id, offset: 0, count: 32 },
    { workerId: 'gone', offset: 32, count: 32 },
  ] });
  const step = await worker.next('round');
  assert.deepEqual([step.runId, step.weights, step.offset, step.count, step.assignments], ['run', weights, 0, 32, undefined]);
  // The relay adds up updates and sends the laptop one result, naming who dropped out.
  worker.send({ type: 'update', runId: 'run', round: 1, count: 32, delta: packFloats(new Float32Array([2, -2])), loss: 1, computeMs: 3 });
  const result = await coordinator.next('round-result');
  assert.deepEqual(Array.from(unpackFloats(result.delta)), [2, -2]);
  assert.equal(result.updates[0].workerId, welcome.id);
  assert.deepEqual(result.dropped, [{ workerId: 'gone', reason: 'not connected' }]);

  // Worker errors are passed on for the dev page.
  worker.send({ type: 'worker-error', error: 'Computation failed' });
  const failure = await coordinator.next('worker-error');
  assert.equal(failure.workerId, welcome.id);
  assert.equal(failure.error, 'Computation failed');

  const secondCoordinator = await connect(url, 'coordinator');
  assert.match((await secondCoordinator.next('error')).error, /already connected/);
  worker.send({ type: 'round', to: welcome.id });
  assert.match((await worker.next('error')).error, /not allowed/);
  coordinator.socket.send('{broken json');
  assert.equal((await coordinator.next('error')).type, 'error');

  worker.socket.close();
  assert.deepEqual((await coordinator.next('presence')).workers, []);
  coordinator.send({ type: 'round', runId: 'run', round: 2, localSteps: 1, deadlineMs: 5000, weights, assignments: [{ workerId: welcome.id, offset: 0, count: 32 }] });
  assert.deepEqual((await coordinator.next('round-result')).dropped, [{ workerId: welcome.id, reason: 'not connected' }]);
  coordinator.send({ type: 'round' });
  assert.match((await coordinator.next('error')).error, /assignments/);

  const remainingWorker = await connect(url, 'worker');
  await remainingWorker.next('welcome');
  coordinator.socket.close();
  assert.equal((await remainingWorker.next('coordinator-left')).type, 'coordinator-left');
});

test('relay rejects cross-origin browser connections', async t => {
  const server = createServer();
  const relay = attachRelay(server);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => {
    relay.close();
    await new Promise(resolve => server.close(resolve));
  });
  const socket = new WebSocket(`ws://127.0.0.1:${server.address().port}/swarm`, { origin: 'https://unrelated.example' });
  const [error] = await once(socket, 'error');
  assert.match(error.message, /403/);
});
