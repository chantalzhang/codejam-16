import { DurableObject } from 'cloudflare:workers';
import { allowedOrigin, createRelay } from '../server/relay-core.js';

// Hosted relay: one Durable Object holds every connection, using the same relay logic as
// local dev. It stays awake while anyone is connected, which fits the free plan: one
// object running all day is under the daily compute allowance.
export class Swarm extends DurableObject {
  relay = createRelay();

  fetch() {
    const [client, server] = Object.values(new WebSocketPair());
    server.accept();
    const connection = this.relay.connect(text => {
      try { server.send(text); } catch { /* Already closed. */ }
    });
    server.addEventListener('message', event => connection.receive(typeof event.data === 'string' ? event.data : null));
    server.addEventListener('close', () => connection.disconnect());
    server.addEventListener('error', () => connection.disconnect());
    return new Response(null, { status: 101, webSocket: client });
  }
}

// Serves the built pages (dist/) and sends /swarm WebSockets to the relay.
export default {
  fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== '/swarm') return env.ASSETS.fetch(request);
    if (request.headers.get('Upgrade') !== 'websocket' || !allowedOrigin(request.headers.get('Origin'), url.host)) {
      return new Response('Forbidden', { status: 403 });
    }
    return env.SWARM.get(env.SWARM.idFromName('main')).fetch(request);
  },
};
