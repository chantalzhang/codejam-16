import { randomUUID } from 'node:crypto';
import { WebSocket, WebSocketServer } from 'ws';

// One local session. Training state and model math stay out of the relay.
export function attachRelay(server) {
  const sockets = new WebSocketServer({ noServer: true, maxPayload: 2 * 1024 * 1024 });
  const workers = new Map();
  let coordinator = null;
  const send = (socket, message) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  };
  const presence = () => send(coordinator, {
    type: 'presence',
    workers: [...workers].map(([id, { ready, backend, tasks }]) => ({ id, ready, backend, tasks })),
  });
  const upgrade = (request, socket, head) => {
    if (new URL(request.url, 'http://localhost').pathname !== '/swarm') return;
    const origin = request.headers.origin;
    if (origin !== `http://${request.headers.host}` && origin !== `https://${request.headers.host}`) {
      socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');
      return;
    }
    sockets.handleUpgrade(request, socket, head, connection => sockets.emit('connection', connection));
  };
  server.on('upgrade', upgrade);
  sockets.on('connection', socket => {
    const id = randomUUID();
    let role;
    socket.on('error', () => {});
    socket.on('message', (data, binary) => {
      try {
        if (binary) throw new Error('Expected a JSON message.');
        const message = JSON.parse(data.toString());
        if (!message || typeof message !== 'object') throw new Error('Expected a message object.');
        if (!role) {
          if (message.type !== 'join' || !['coordinator', 'worker'].includes(message.role)) {
            throw new Error('Join as coordinator or worker first.');
          }
          if (message.role === 'coordinator' && coordinator) throw new Error('A coordinator is already connected.');
          role = message.role;
          if (role === 'coordinator') coordinator = socket;
          else workers.set(id, { socket, ready: false });
          send(socket, { type: 'welcome', id, role });
          presence();
        } else if (role === 'worker' && message.type === 'ready') {
          if (!['webgl', 'cpu'].includes(message.backend) || !Array.isArray(message.tasks) ||
              !message.tasks.every(task => typeof task === 'string')) {
            throw new Error('Expected backend and supported task IDs.');
          }
          Object.assign(workers.get(id), { ready: true, backend: message.backend, tasks: message.tasks });
          presence();
        } else if (role === 'coordinator' && message.type === 'step') {
          const worker = workers.get(message.to);
          if (!worker?.ready) throw new Error('Selected worker is not ready.');
          const { to, ...assignment } = message;
          send(worker.socket, assignment);
        } else if (role === 'worker' && ['gradient', 'worker-error'].includes(message.type)) {
          send(coordinator, { ...message, workerId: id });
        } else {
          throw new Error('Message is not allowed for this role.');
        }
      } catch (error) {
        send(socket, { type: 'error', error: error.message });
      }
    });
    socket.on('close', () => {
      if (socket === coordinator) {
        coordinator = null;
        for (const worker of workers.values()) send(worker.socket, { type: 'coordinator-left' });
      }
      workers.delete(id);
      presence();
    });
  });
  return {
    close() {
      server.off('upgrade', upgrade);
      for (const socket of sockets.clients) socket.terminate();
      sockets.close();
    },
  };
}
