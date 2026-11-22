# GPUmaxxer

**the room is the GPU.** scan a QR code and your phone joins a live datacenter. every phone in the room trains the same neural net at the same time, right in the browser. no app, no install, no account.

**try it:** big screen at [gpu.chantalz.workers.dev/?role=laptop](https://gpu.chantalz.workers.dev/?role=laptop), phones scan the QR or open [gpu.chantalz.workers.dev](https://gpu.chantalz.workers.dev)

![big screen mid-run](docs/big-screen.png)

## the problem

training a model needs GPUs, and GPUs live in datacenters you rent by the hour. meanwhile everyone in this room is holding a GPU that's doing nothing. and distributed training, the thing every big lab runs on, is invisible: it happens in a server room nobody gets to see.

## what it does

GPUmaxxer turns a room of phones into one training cluster and puts the whole thing on the projector.

- you scan the QR code and your phone joins in seconds
- each phone trains its own copy of the model on its own slice of the data, using its own GPU (WebGPU), then sends back what it learned
- the laptop merges every phone's change into one shared model and sends it back out. that's one round, about 3 a second on an iPhone
- the big screen shows it happening live: held-out test images going from `?` to wrong to right, loss dropping, a square per phone that flashes when its update lands, and the room's combined speed in examples per second

<img src="docs/phone.png" alt="phone during a run" width="280" align="right" />

your phone shows what it's doing too: the current round, how many images it trained on and how fast, and what share of the room's work it did.

**more phones = a better model, not just a faster one.** each phone holds different data. in our sim (`ml/`), one phone only ever sees its 999 images and levels off in the high 70s, while three or more phones see more of the dataset between them and get to 92–94% in a 50 second run. so the crowd actually changes the result, and you can watch it.

everything on screen is real: real training, real held-out test images, real numbers. nothing is pre-trained or scripted.

<br clear="right" />

## the tasks

pick one at the top of the big screen:

- **guess the game** — 9 games (fortnite, minecraft, among us, roblox, terraria, valorant, cs2, dota 2, league) from gameplay screenshots. tested on videos the model never trained on
- **italian brainrot** — tralalero tralala, tung tung tung sahur and friends. small dataset, goofy results
- **fashion-mnist** — the classic clothing dataset

## how it works

```
phones (WebGPU)  <-->  relay (Cloudflare Durable Object)  <-->  laptop (big screen)
 train 10 steps          sends the model out, sums          merges the change with an
 on their slice          everyone's changes into one        outer optimizer, evaluates,
                         message per round                  draws everything
```

- **local SGD, DiLoCo-style.** phones take 10 Adam steps each per round, then the laptop treats the average change as a gradient for a Nesterov outer optimizer. this was about 2x faster than plain averaging in our PyTorch sim (`ml/sim.py`)
- **the relay does the heavy lifting.** it adds up every phone's update, so the laptop gets one message per round whether there are 3 phones or 400
- **small messages.** weights are packed as raw float32 instead of JSON: 124 KB down to 32 KB
- **each phone downloads one ~3 MB slice**, not the whole dataset

## numbers we measured

- iPhone on WebGPU: ~22 ms per training step, ~0.3 s per round
- iPhone + Mac reached 78% about **1.6x faster** than the iPhone alone (fashion-mnist, 3 runs each)
- **400 simulated phones:** ~0.9 s per round, and the laptop's share stays around 160–220 ms

## built for a real room

demos break in real rooms, so it handles the boring stuff:

- phones reconnect by themselves after a wifi blip or a locked screen, and rejoin from the next round
- a phone that misses a round sits out briefly and comes back, so one slow phone can't stall everyone
- reloading the big screen takes over cleanly. phones stay connected
- phones without WebGPU, or people who scanned inside instagram/snapchat, get told exactly which browser to open. the big screen shows how many can't join
- the screen stays awake while training

## run it yourself

needs Node 22.

```bash
npm ci
npm start          # builds and serves at http://127.0.0.1:4173
```

open `http://127.0.0.1:4173/?role=laptop` for the big screen and `http://127.0.0.1:4173/` in a few tabs as phones. click **Dev** on the big screen for settings and per-phone details.

```bash
npm test                 # unit tests
npm run test:browser     # full end-to-end runs in headless chrome
npm run deploy           # publish to Cloudflare (npx wrangler login first)
npm run fake-phones -- 400 wss://gpu.chantalz.workers.dev/swarm   # load test
```

data prep and experiments are in Python (`ml/`, run with `uv run ml/<script>.py`): `prepare_games.py` and `prepare_brainrot.py` download and clean the datasets, `export_images.py` writes the files phones download, `bench_images.py` checks a task is learnable, `sim.py` simulates phones training together.

## stack

TensorFlow.js on WebGPU, plain JS + Vite, Cloudflare Workers + Durable Objects, PyTorch for data prep and simulation.

## credits

game screenshots: [Gameplay Images](https://huggingface.co/datasets/Bingsu/Gameplay_Images), [Gamed](https://huggingface.co/datasets/ybashir/gamed-game-classification-dataset), [ViT Video Game Classification](https://zenodo.org/records/15574186) (all CC BY 4.0). brainrot images: [Italian Brainrot Images](https://www.kaggle.com/datasets/bubblepw/italian-brainrot-images). fashion-mnist by Zalando Research (MIT). full details in `public/data/`.
