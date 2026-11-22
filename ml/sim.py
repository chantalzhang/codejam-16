# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision"]
# ///
"""Simulate local SGD on N phones to pick K, the local steps per sync.

Fashion-MNIST stands in for the final demo task. Same model and batch size as the
phone task. Each round, every simulated phone trains its own copy of the model for K
steps with Adam (keeping its optimizer state between rounds), then the copies are
averaged. Phones run one after another here; wall time is estimated from iPhone
measurements, as if they ran in parallel.

Run: uv run ml/sim.py   (~5 minutes on a laptop CPU; add --no-config if your uv
config points at a private package index that lacks torch)
"""
import copy
import statistics
import time

import torch
from torch import nn
from torchvision.datasets import FashionMNIST

BATCH, LR, TARGET_ACCURACY = 32, 0.01, 0.80  # Adam lr 0.01 needed ~2.5x fewer steps than SGD
STEP_MS = 67  # iPhone compute for one 32-image step (measured)
SYNC_MS = 117 + 25  # network + laptop per sync (measured); assumed not to grow with phones
MAX_STEPS = 3000  # per phone, before giving up
PHONES, LOCAL_STEPS, SEEDS = [1, 3, 6], [5, 10], [0, 1]


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
    )


def load(train):
    data = FashionMNIST("ml/data", train=train, download=True)
    return (data.data.float() / 255)[:, None], data.targets


def rounds_to_target(phones, local_steps, seed, train, test):
    """Rounds until the averaged model reaches TARGET_ACCURACY, or None."""
    torch.manual_seed(seed)
    batches = torch.Generator().manual_seed(seed)
    shared = make_model()
    copies = [copy.deepcopy(shared) for _ in range(phones)]
    optimizers = [torch.optim.Adam(c.parameters(), lr=LR) for c in copies]
    for round_ in range(1, MAX_STEPS // local_steps + 1):
        for model, optimizer in zip(copies, optimizers):
            model.load_state_dict(shared.state_dict())
            for _ in range(local_steps):
                index = torch.randint(0, len(train[0]), (BATCH,), generator=batches)
                optimizer.zero_grad()
                nn.functional.cross_entropy(model(train[0][index]), train[1][index]).backward()
                optimizer.step()
        shared.load_state_dict({key: torch.stack([c.state_dict()[key] for c in copies]).mean(0)
                                for key in shared.state_dict()})
        with torch.no_grad():
            accuracy = (shared(test[0]).argmax(1) == test[1]).float().mean()
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
