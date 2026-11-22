# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision"]
# ///
"""Simulate local SGD on N phones, comparing ways to combine their updates.

Uses Fashion-MNIST, with the same model and batch size as the
phone task. Each round, every simulated phone trains its own copy of the model for K
steps with Adam (keeping its optimizer state between rounds). The laptop then treats
"shared model minus the average of the copies" as a gradient for an outer optimizer
(DiLoCo-style). Outer SGD with lr 1 and no momentum is plain averaging.

Phones run one after another here; wall time is estimated from iPhone measurements,
as if they ran in parallel.

Run: uv run ml/sim.py   (~6 minutes on a laptop CPU; add --no-config if your uv
config points at a private package index that lacks torch)
"""
import copy
import statistics
import time

import torch
from torch import nn
from torchvision.datasets import FashionMNIST

BATCH, LR, TARGET_ACCURACY = 32, 0.01, 0.80  # Adam lr 0.01 needed ~2.5x fewer steps than SGD
STEP_MS = 22  # iPhone WebGPU compute for one 32-image step (measured)
SYNC_MS = 112 + 29  # network + laptop per sync (measured); assumed not to grow with phones
LOCAL_STEPS = 10  # Best of 5/10/20 in an earlier sweep.
MAX_STEPS = 3000  # per phone, before giving up
PHONES, SEEDS = [1, 3, 6], [0, 1]
# name: (outer learning rate, Nesterov momentum)
OUTER = {"plain average": (1.0, 0.0), "nesterov lr 0.5": (0.5, 0.9), "nesterov lr 0.7": (0.7, 0.9)}


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


def rounds_to_target(phones, outer_lr, outer_momentum, seed, train, test):
    """Rounds until the shared model reaches TARGET_ACCURACY, or None."""
    torch.manual_seed(seed)
    batches = torch.Generator().manual_seed(seed)
    shared = make_model()
    outer = torch.optim.SGD(shared.parameters(), lr=outer_lr, momentum=outer_momentum, nesterov=outer_momentum > 0)
    copies = [copy.deepcopy(shared) for _ in range(phones)]
    optimizers = [torch.optim.Adam(c.parameters(), lr=LR) for c in copies]
    for round_ in range(1, MAX_STEPS // LOCAL_STEPS + 1):
        for model, optimizer in zip(copies, optimizers):
            model.load_state_dict(shared.state_dict())
            for _ in range(LOCAL_STEPS):
                index = torch.randint(0, len(train[0]), (BATCH,), generator=batches)
                optimizer.zero_grad()
                nn.functional.cross_entropy(model(train[0][index]), train[1][index]).backward()
                optimizer.step()
        for name, param in shared.named_parameters():
            average = torch.stack([dict(c.named_parameters())[name].detach() for c in copies]).mean(0)
            param.grad = param.detach() - average  # The "outer gradient".
        outer.step()
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
    round_ms = LOCAL_STEPS * STEP_MS + SYNC_MS
    print(f"\nEstimated time for the shared model to reach {TARGET_ACCURACY:.0%} test accuracy")
    print(f"(median of {len(SEEDS)} seeds; {LOCAL_STEPS} local steps x {STEP_MS} ms + {SYNC_MS} ms sync per round)\n")
    print("outer optimizer  | " + " | ".join(f"{n} phone{'s' if n > 1 else ''}".ljust(8) for n in PHONES))
    results = {}
    for name, (outer_lr, momentum) in OUTER.items():
        cells = []
        for phones in PHONES:
            rounds = []
            for seed in SEEDS:
                rounds.append(rounds_to_target(phones, outer_lr, momentum, seed, train, test))
                if rounds[-1] is None:
                    break
            seconds = None if None in rounds else statistics.median(rounds) * round_ms / 1000
            results[name, phones] = seconds
            cells.append("--".ljust(8) if seconds is None else f"{seconds:5.0f} s".ljust(8))
        print(f"{name:<16} | " + " | ".join(cells) + f"   [{time.time() - started:.0f} s]", flush=True)
    baseline = results["plain average", 1]
    print(f"\nSpeedup vs 1 phone with plain averaging ({baseline:.0f} s):")
    for name in OUTER:
        print(f"  {name:<16} " + "  ".join(
            f"{n} phone{'s' if n > 1 else ' '}: {'--' if results[name, n] is None else f'{baseline / results[name, n]:.1f}x'}"
            for n in PHONES))


if __name__ == "__main__":
    main()
