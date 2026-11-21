import { defineConfig } from 'vite';
import { attachRelay } from './server/relay.js';

export default defineConfig({
  plugins: [{
    name: 'local-training-relay',
    configureServer(server) {
      const relay = attachRelay(server.httpServer);
      server.httpServer.once('close', () => relay.close());
    },
  }],
});
