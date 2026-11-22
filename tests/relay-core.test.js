import test from 'node:test';
import assert from 'node:assert/strict';
import { createRelay } from '../server/relay-core.js';
import { packFloats, unpackFloats } from '../src/packing.js';

// A relay with a laptop and named workers connected in-process. Each worker's handler
// gets the round it was sent and a `reply` function.
function setup(names) {
  const relay = createRelay();
  const toLaptop = [];
  const laptop = relay.connect(text => toLaptop.push(JSON.parse(text)));
  laptop.receive(JSON.stringify({ type: 'join', role: 'coordinator' }));
  const ids = {}, handlers = {}, connections = {};
  for (const name of names) {
    const connection = relay.connect(text => {
      const message = JSON.parse(text);
      if (message.type === 'welcome') ids[name] = message.id;
      if (message.type === 'round') {
        handlers[name]?.(message, answer => connection.receive(JSON.stringify({ runId: message.runId, round: message.round, ...answer })));
      }
    });
    connection.receive(JSON.stringify({ type: 'join', role: 'worker' }));
    connection.receive(JSON.stringify({ type: 'ready', backend: 'cpu', tasks: ['t'] }));
    connections[name] = connection;
  }
  const startRound = (assignments, extra = {}) => laptop.receive(JSON.stringify({
    type: 'round', runId: 'run', round: 1, localSteps: 1, deadlineMs: 50, weights: packFloats(new Float32Array(2)),
    assignments: assignments.map(([name, count]) => ({ workerId: ids[name] ?? name, offset: 0, count })), ...extra,
  }));
  const results = () => toLaptop.filter(message => message.type === 'round-result');
  return { ids, handlers, connections, startRound, results };
}

const update = (values, count) => ({ type: 'update', count, delta: packFloats(Float32Array.from(values)), loss: 1, computeMs: 5 });

test('the relay averages updates weighted by example count and replies once', () => {
  const swarm = setup(['a', 'b']);
  swarm.handlers.a = (round, reply) => reply(update([1, -1], 1));
  swarm.handlers.b = (round, reply) => reply(update([5, 3], 3));
  swarm.startRound([['a', 1], ['b', 3]]);
  const [result] = swarm.results();
  assert.equal(swarm.results().length, 1);
  assert.deepEqual(Array.from(unpackFloats(result.delta)), [4, 2]);
  assert.equal(result.count, 4);
  assert.deepEqual(result.updates.map(item => [item.workerId, item.count, item.computeMs]), [[swarm.ids.a, 1, 5], [swarm.ids.b, 3, 5]]);
  assert.deepEqual(result.dropped, []);
});

test('stale, duplicate, unassigned, and invalid updates never reach the average', () => {
  const swarm = setup(['good', 'stale', 'twice', 'outsider', 'nan', 'small']);
  swarm.handlers.good = (round, reply) => reply(update([1, 1], 8));
  swarm.handlers.stale = (round, reply) => { reply({ ...update([99, 99], 8), round: 0 }); reply(update([1, 1], 8)); };
  swarm.handlers.twice = (round, reply) => { reply(update([1, 1], 8)); reply(update([99, 99], 8)); };
  swarm.handlers.nan = (round, reply) => reply(update([NaN, 1], 8));
  swarm.handlers.small = (round, reply) => reply(update([1], 8));
  // 'outsider' isn't in this round; its update must be ignored.
  swarm.connections.outsider.receive(JSON.stringify({ runId: 'run', round: 1, ...update([99, 99], 8) }));
  swarm.startRound([['good', 8], ['stale', 8], ['twice', 8], ['nan', 8], ['small', 8]]);
  const [result] = swarm.results();
  assert.deepEqual(Array.from(unpackFloats(result.delta)), [1, 1]);
  assert.equal(result.updates.length, 3);
  assert.deepEqual(result.dropped.map(item => item.reason), ['sent an invalid update', 'sent an invalid update']);
});

test('the round ends at the deadline, or as soon as missing workers fail or leave', async () => {
  const swarm = setup(['fast', 'silent', 'broken', 'leaving']);
  swarm.handlers.fast = (round, reply) => reply(update([2, 2], 8));
  swarm.handlers.broken = (round, reply) => reply({ type: 'worker-error', error: 'GPU reset' });
  swarm.handlers.leaving = () => queueMicrotask(() => swarm.connections.leaving.disconnect());
  swarm.startRound([['fast', 8], ['silent', 8], ['broken', 8], ['leaving', 8], ['never-connected', 8]]);
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(swarm.results().length, 0, 'Still waiting for the silent worker');
  await new Promise(resolve => setTimeout(resolve, 60));
  const [result] = swarm.results();
  assert.deepEqual(Array.from(unpackFloats(result.delta)), [2, 2]);
  const reasons = Object.fromEntries(result.dropped.map(({ workerId, reason }) => [workerId, reason]));
  assert.equal(reasons[swarm.ids.silent], 'missed the round 1 deadline');
  assert.equal(reasons[swarm.ids.broken], 'failed: GPU reset');
  assert.equal(reasons[swarm.ids.leaving], 'left');
  assert.equal(reasons['never-connected'], 'not connected');
});

test('with no usable updates, the result says so instead of averaging nothing', () => {
  const swarm = setup(['broken']);
  swarm.handlers.broken = (round, reply) => reply({ type: 'worker-error', error: 'out of memory' });
  swarm.startRound([['broken', 8]]);
  const [result] = swarm.results();
  assert.equal(result.delta, null);
  assert.equal(result.count, 0);
});

test('the laptop gets one message per round however many workers there are', () => {
  const names = Array.from({ length: 300 }, (_, i) => `w${i}`);
  const swarm = setup(names);
  for (const name of names) swarm.handlers[name] = (round, reply) => reply(update([1, 2], 8));
  swarm.startRound(names.map(name => [name, 8]));
  assert.equal(swarm.results().length, 1);
  assert.deepEqual(Array.from(unpackFloats(swarm.results()[0].delta)), [1, 2]);
  assert.equal(swarm.results()[0].count, 300 * 8);
});
