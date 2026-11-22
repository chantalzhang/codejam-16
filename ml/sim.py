# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision"]
# ///
"""Simulate local SGD on N phones to pick K, the local steps per sync.

Fashion-MNIST stands in for the final demo task. Same model and batch size as the
phone task. Each round, every simulated phone trains its own copy of the
model for K steps, then the copies are averaged. All phones train in one batched call
(vmap); wall time is estimated from iPhone measurements, as if phones ran in parallel.

Run: uv run ml/sim.py   (~10 minutes on a laptop CPU; add --no-config if your uv
config points at a private package index that lacks torch)
"""
import statistics
import time

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torchvision.datasets import FashionMNIST

BATCH, LR, TARGET_ACCURACY = 32, 0.3, 0.70  # lr 0.3 beat 0.1 and 0.8 in quick tests
STEP_MS = 67  # iPhone compute for one 32-image step (measured)
SYNC_MS = 117 + 25  # network + laptop per sync (measured); assumed not to grow with phones
MAX_STEPS = 3000  # per phone, before giving up
PHONES, LOCAL_STEPS, SEEDS = [1, 2, 4, 8], [1, 5, 10, 20], [0, 1]

MODEL = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
)


def loss(params, x, y):
    return nn.functional.cross_entropy(functional_call(MODEL, params, (x,)), y)


phone_grads = vmap(grad(loss))  # Gradients for every phone's copy in one call.


def load(train):
    data = FashionMNIST("ml/data", train=train, download=True)
    return (data.data.float() / 255)[:, None], data.targets


def rounds_to_target(phones, local_steps, seed, train, test):
    """Rounds until the averaged model reaches TARGET_ACCURACY, or None."""
    torch.manual_seed(seed)
    for layer in MODEL:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    shared = {name: p.detach() for name, p in MODEL.named_parameters()}
    batches = torch.Generator().manual_seed(seed)
    check_every = max(1, 10 // local_steps)  # Check accuracy about every 10 local steps.
    for round_ in range(1, MAX_STEPS // local_steps + 1):
        copies = {name: p.expand(phones, *p.shape) for name, p in shared.items()}
        for _ in range(local_steps):
            index = torch.randint(0, len(train[0]), (phones, BATCH), generator=batches)
            grads = phone_grads(copies, train[0][index], train[1][index])
            copies = {name: p - LR * grads[name] for name, p in copies.items()}
        shared = {name: p.mean(0) for name, p in copies.items()}
        if round_ % check_every == 0:
            with torch.no_grad():
                accuracy = (functional_call(MODEL, shared, (test[0],)).argmax(1) == test[1]).float().mean()
            if accuracy >= TARGET_ACCURACY:
                return round_
    return None


def main():
    torch.set_num_threads(6)  # Leave some cores free so the laptop stays responsive.
    train = load(train=True)
    test = tuple(t[:1000] for t in load(train=False))
    started = time.time()
    print(f"\nEstimated time for the shared model to reach {TARGET_ACCURACY:.0%} test accuracy")
    print(f"(median of {len(SEEDS)} seeds; {STEP_MS} ms per local step + {SYNC_MS} ms per sync)\n")
    print("K = local steps per sync | " + " | ".join(f"{n} phone{'s' if n > 1 else ''}".ljust(8) for n in PHONES))
    best = {}
    for k in LOCAL_STEPS:
        cells = []
        for phones in PHONES:
            rounds = []
            for seed in SEEDS:
                rounds.append(rounds_to_target(phones, k, seed, train, test))
                if rounds[-1] is None:
                    break
            if None in rounds:
                cells.append("--".ljust(8))
                continue
            seconds = statistics.median(rounds) * (k * STEP_MS + SYNC_MS) / 1000
            best[phones] = min(best.get(phones, (seconds, k)), (seconds, k))
            cells.append(f"{seconds:5.0f} s".ljust(8))
        print(f"K = {k:<2}                   | " + " | ".join(cells) + f"   [{time.time() - started:.0f} s]", flush=True)
    print("\nFastest K per phone count:")
    for phones, (seconds, k) in sorted(best.items()):
        print(f"  {phones} phone{'s' if phones > 1 else ' '}: K = {k:<2}  {seconds:4.0f} s  "
              f"({best[1][0] / seconds:.1f}x vs 1 phone)")


if __name__ == "__main__":
    main()
