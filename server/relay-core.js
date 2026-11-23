import { packFloats, unpackFloats } from '../src/packing.js';

// Routes messages between one coordinator and many workers, and sums each round's updates
// so the coordinator gets one message per round. Shared by server/relay.js (Node) and
// worker/index.js (Cloudflare).

// The page's own origin, or a local dev page (http://127.0.0.1 or localhost) talking to a
// hosted relay. Blocks other websites from connecting.
export function allowedOrigin(origin, host) {
  return origin === `http://${host}` || origin === `https://${host}` ||
    /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin ?? '');
}

export function createRelay() {
  const workers = new Map(); // id → { send, ready, backend, tasks, device, note, away }
  let coordinator = null; // { id, send, demote }
  let pending = null; // The round in progress, see startRound.
  let workersJoined = 0; // Gives each worker a join number, e.g. to pick a data slice.
  let lastStatus = null; // The laptop's latest status for phones, also sent to new joiners.
  const presence = () => coordinator?.send({
    type: 'presence',
    workers: [...workers].map(([id, { ready, backend, tasks, device, note, away }]) => ({ id, ready, backend, tasks, device, note, away })),
  });

  // Fan-out: each assigned worker gets the model with its own slice of examples.
  function startRound({ assignments, deadlineMs, ...round }) {
    if (pending) clearTimeout(pending.timer); // Superseded, e.g. after a stop.
    const size = unpackFloats(round.weights).length;
    pending = {
      runId: round.runId, round: round.round, sum: new Float64Array(size), total: 0,
      expected: new Map(), answered: new Set(), updates: [], dropped: new Map(), size,
    };
    // Register everyone before sending, so fast replies can't finish the round early.
    for (const { workerId, count } of assignments) {
      if (workers.get(workerId)?.ready) pending.expected.set(workerId, count * round.localSteps);
      else pending.dropped.set(workerId, 'not connected');
    }
    const current = pending;
    pending.timer = setTimeout(() => {
      if (pending !== current) return;
      for (const workerId of current.expected.keys()) {
        if (!current.answered.has(workerId)) current.dropped.set(workerId, `missed the round ${current.round} deadline`);
      }
      finishRound();
    }, deadlineMs);
    for (const { workerId, offset, count } of assignments) {
      if (current.expected.has(workerId)) workers.get(workerId).send({ ...round, offset, count });
    }
    settleRound();
  }

  const inRound = (workerId, message) => pending && pending.expected.has(workerId) && !pending.answered.has(workerId) &&
    (!message || (message.runId === pending.runId && message.round === pending.round));

  function addUpdate(workerId, message) {
    if (!inRound(workerId, message)) return; // Stale, duplicate, or never assigned.
    let delta;
    try {
      delta = unpackFloats(message.delta);
      if (delta.length !== pending.size || message.count !== pending.expected.get(workerId) ||
          !Number.isFinite(message.loss) || !(message.computeMs >= 0) || !delta.every(Number.isFinite)) throw new Error();
    } catch {
      dropWorker(workerId, 'sent an invalid update');
      return;
    }
    // Weighted by examples, so a worker that trained on more counts for more.
    for (let i = 0; i < delta.length; i++) pending.sum[i] += delta[i] * message.count;
    pending.total += message.count;
    pending.answered.add(workerId);
    const { count, loss, computeMs, memory } = message;
    pending.updates.push({ workerId, count, loss, computeMs, memory });
    settleRound();
  }

  function dropWorker(workerId, reason, message) {
    if (!inRound(workerId, message)) return;
    pending.answered.add(workerId);
    pending.dropped.set(workerId, reason);
    settleRound();
  }

  // The laptop left or another laptop tab took over: the round in progress is lost, and
  // phones wait for the next laptop.
  function coordinatorLeft() {
    coordinator = null;
    lastStatus = null;
    if (pending) clearTimeout(pending.timer);
    pending = null;
    for (const worker of workers.values()) worker.send({ type: 'coordinator-left' });
  }

  function settleRound() {
    if (pending && pending.answered.size === pending.expected.size) finishRound();
  }

  // One message back: the average change (null if nothing usable arrived) plus small
  // per-worker stats for the dev page.
  function finishRound() {
    clearTimeout(pending.timer);
    const { runId, round, sum, total, updates, dropped } = pending;
    pending = null;
    coordinator?.send({
      type: 'round-result', runId, round, count: total, updates,
      delta: total ? packFloats(Float32Array.from(sum, value => value / total)) : null,
      dropped: [...dropped].map(([workerId, reason]) => ({ workerId, reason })),
    });
  }

  // `send(text)` delivers to this connection. Returns handlers for its messages and close.
  function connect(sendText) {
    const id = crypto.randomUUID();
    const send = message => sendText(JSON.stringify(message));
    let role;
    let connected = true;
    function receive(text) {
      try {
        if (typeof text !== 'string') throw new Error('Expected a JSON message.');
        const message = JSON.parse(text);
        if (!message || typeof message !== 'object') throw new Error('Expected a message object.');
        if (!role) {
          if (message.type !== 'join' || !['coordinator', 'worker'].includes(message.role)) {
            throw new Error('Join as coordinator or worker first.');
          }
          // A new laptop tab (e.g. after a reload) takes over from the old one.
          if (message.role === 'coordinator' && coordinator) {
            coordinator.send({ type: 'replaced' });
            coordinator.demote();
            coordinatorLeft();
          }
          role = message.role;
          if (role === 'coordinator') coordinator = { id, send, demote: () => { role = 'replaced'; } };
          else workers.set(id, { send, ready: false });
          send({ type: 'welcome', id, role, ...(role === 'worker' && { joinNumber: workersJoined++ }) });
          if (role === 'worker' && lastStatus) send(lastStatus);
          presence();
        } else if (role === 'worker' && message.type === 'ready') {
          if (!['webgl', 'webgpu', 'cpu'].includes(message.backend) || !Array.isArray(message.tasks) ||
              !message.tasks.every(task => typeof task === 'string')) {
            throw new Error('Expected backend and supported task IDs.');
          }
          const device = typeof message.device === 'string' ? message.device.slice(0, 40) : 'Unknown';
          const note = typeof message.note === 'string' ? message.note.slice(0, 80) : null;
          Object.assign(workers.get(id), { ready: true, backend: message.backend, tasks: message.tasks, device, note, away: Boolean(message.away) });
          presence();
        } else if (role === 'coordinator' && message.type === 'round') {
          if (!Array.isArray(message.assignments) || !(message.deadlineMs > 0)) {
            throw new Error('Expected round assignments and a deadline.');
          }
          startRound(message);
        } else if (role === 'coordinator' && message.type === 'status') {
          // Small run status for the phone screens (phone count, state, the model's guesses).
          lastStatus = message;
          for (const worker of workers.values()) worker.send(message);
        } else if (role === 'worker' && message.type === 'away') {
          // The phone's page went into the background (another app, locked screen), where it
          // can't compute. Stop waiting for it this round; the laptop skips it until it's back.
          workers.get(id).away = Boolean(message.away);
          if (message.away) dropWorker(id, 'page in the background');
          presence();
        } else if (role === 'worker' && message.type === 'update') {
          addUpdate(id, message);
        } else if (role === 'worker' && message.type === 'worker-error') {
          const error = String(message.error).slice(0, 200);
          dropWorker(id, `failed: ${error}`, message);
          coordinator?.send({ type: 'worker-error', workerId: id, error }); // For the dev page.
        } else {
          throw new Error('Message is not allowed for this role.');
        }
      } catch (error) {
        send({ type: 'error', error: error.message });
      }
    }
    function disconnect() {
      if (!connected) return;
      connected = false;
      if (coordinator?.id === id) coordinatorLeft();
      dropWorker(id, 'left');
      workers.delete(id);
      presence();
    }
    return { receive, disconnect };
  }
  return { connect };
}
