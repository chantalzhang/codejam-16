import { WebSocket, WebSocketServer } from 'ws';
import { allowedOrigin, createRelay } from './relay-core.js';

// Runs the relay on a Node HTTP server (local dev and tests).
export function attachRelay(server) {
  const sockets = new WebSocketServer({ noServer: true, maxPayload: 2 * 1024 * 1024 });
  const relay = createRelay();
  const upgrade = (request, socket, head) => {
    if (new URL(request.url, 'http://localhost').pathname !== '/swarm') return;
    if (!allowedOrigin(request.headers.origin, request.headers.host)) {
      socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');
      return;
    }
    sockets.handleUpgrade(request, socket, head, connection => sockets.emit('connection', connection));
  };
  server.on('upgrade', upgrade);
  sockets.on('connection', socket => {
    const client = relay.connect(text => { if (socket.readyState === WebSocket.OPEN) socket.send(text); });
    socket.on('error', () => {});
    socket.on('message', (data, binary) => client.receive(binary ? null : data.toString()));
    socket.on('close', () => client.disconnect());
  });
  return {
    close() {
      server.off('upgrade', upgrade);
      for (const socket of sockets.clients) socket.terminate();
      sockets.close();
    },
  };
}
