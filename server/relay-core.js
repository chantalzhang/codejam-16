// Routes messages between one coordinator and many workers. Training state and model math
// stay out of the relay. It has no transport of its own, so the same logic runs in Node
// (server/relay.js, local dev) and in a Cloudflare Durable Object (worker/index.js).

// The page's own origin, or a local dev page (http://127.0.0.1 or localhost) talking to a
// hosted relay. Blocks other websites from connecting.
export function allowedOrigin(origin, host) {
  return origin === `http://${host}` || origin === `https://${host}` ||
    /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin ?? '');
}

export function createRelay() {
  const workers = new Map(); // id → { send, ready, backend, tasks, device, note }
  let coordinator = null; // { id, send }
  const presence = () => coordinator?.send({
    type: 'presence',
    workers: [...workers].map(([id, { ready, backend, tasks, device, note }]) => ({ id, ready, backend, tasks, device, note })),
  });

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
          if (message.role === 'coordinator' && coordinator) throw new Error('A coordinator is already connected.');
          role = message.role;
          if (role === 'coordinator') coordinator = { id, send };
          else workers.set(id, { send, ready: false });
          send({ type: 'welcome', id, role });
          presence();
        } else if (role === 'worker' && message.type === 'ready') {
          if (!['webgl', 'webgpu', 'cpu'].includes(message.backend) || !Array.isArray(message.tasks) ||
              !message.tasks.every(task => typeof task === 'string')) {
            throw new Error('Expected backend and supported task IDs.');
          }
          const device = typeof message.device === 'string' ? message.device.slice(0, 40) : 'Unknown';
          const note = typeof message.note === 'string' ? message.note.slice(0, 80) : null;
          Object.assign(workers.get(id), { ready: true, backend: message.backend, tasks: message.tasks, device, note });
          presence();
        } else if (role === 'coordinator' && message.type === 'round') {
          const worker = workers.get(message.to);
          if (!worker?.ready) throw new Error('Selected worker is not ready.');
          const { to, ...assignment } = message;
          worker.send(assignment);
        } else if (role === 'worker' && ['update', 'worker-error'].includes(message.type)) {
          coordinator?.send({ ...message, workerId: id });
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
      if (coordinator?.id === id) {
        coordinator = null;
        for (const worker of workers.values()) worker.send({ type: 'coordinator-left' });
      }
      workers.delete(id);
      presence();
    }
    return { receive, disconnect };
  }
  return { connect };
}
