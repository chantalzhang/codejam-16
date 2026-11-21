import { defineConfig } from 'vite';
import { attachRelay } from './server/relay.js';
import { runLog } from './server/run-log.js';

function relayPlugin() {
  const attach = server => {
    const relay = attachRelay(server.httpServer);
    server.httpServer.once('close', () => relay.close());
    server.middlewares.use('/api/runs', runLog(process.env.RUNS_DIR ?? 'runs'));
  };
  return { name: 'training-relay', configureServer: attach, configurePreviewServer: attach };
}

export default defineConfig({
  plugins: [relayPlugin()],
  // Lets phones reach the built app through a Cloudflare quick tunnel.
  preview: { allowedHosts: ['.trycloudflare.com'] },
});
