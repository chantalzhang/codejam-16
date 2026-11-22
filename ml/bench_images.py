# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Benchmarks a tiny CNN, like the phones would train, on a prepared image dataset.

Usage: uv run --no-config ml/bench_images.py ml/data/cache/games64.npz [--size 32] [--width 8]

Trains on CPU with class-balanced sampling, then reports held-out accuracy overall,
per class, the most common mix-ups, and how many test images have a near copy in training
(a sign the test is too easy).
"""
import argparse
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def model(width, classes):
    # The Fashion-MNIST phone model (src/fashion-task.js), with RGB input and adjustable width.
    return nn.Sequential(
        nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, classes),
    )


def resize(images, size):
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    return F.interpolate(x, size=(size, size), mode="area") if size != x.shape[-1] else x


def near_copies(train, test, threshold):
    # Mean absolute pixel difference at 16x16; below the threshold counts as a near copy.
    a, b = resize(train, 16).flatten(1), resize(test, 16).flatten(1)
    nearest = torch.cat([torch.cdist(chunk, a, p=1).min(1).values for chunk in b.split(256)]) / a.shape[1]
    return (nearest < threshold).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    data = np.load(args.data)
    classes = sorted(set(data["labels"]), key=list(data["labels"]).index)
    y = torch.tensor([classes.index(label) for label in data["labels"]])
    train, test = data["split"] == "train", data["split"] == "test"
    x_train, x_test = resize(data["images"][train], args.size), resize(data["images"][test], args.size)
    y_train, y_test = y[train], y[test]
    mean = x_train.mean((0, 2, 3), keepdim=True)

    net = model(args.width, len(classes))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # Class-balanced batches: every class equally likely, whatever its size.
    weights = 1 / torch.bincount(y_train).float()[y_train]
    started = time.time()
    for step in range(args.steps):
        batch = torch.multinomial(weights, args.batch, replacement=True)
        xb = x_train[batch]
        if torch.rand(()) < 0.5:
            xb = xb.flip(3)
        loss = F.cross_entropy(net(xb - mean), y_train[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    seconds = time.time() - started

    with torch.no_grad():
        predicted = torch.cat([net(chunk - mean).argmax(1) for chunk in x_test.split(512)])
    correct = (predicted == y_test).float()
    per_class = [correct[y_test == c].mean().item() for c in range(len(classes))]
    params = sum(p.numel() for p in net.parameters())
    print(f"{args.size}x{args.size}, width {args.width} ({params:,} params), {args.steps} steps in {seconds:.0f} s")
    print(f"held-out accuracy {correct.mean():.1%}, balanced (mean per class) {np.mean(per_class):.1%}")
    copies = near_copies(data["images"][train], data["images"][test], threshold=0.03)
    for c, name in enumerate(classes):
        mask = y_test == c
        wrong = predicted[mask & (predicted != y_test)]
        top = torch.bincount(wrong, minlength=len(classes)).argmax().item() if len(wrong) else None
        mixup = f"mostly as {classes[top]}" if top is not None else ""
        print(f"  {name:22s} {per_class[c]:6.1%}  n={mask.sum().item():4d}  near copies in train {copies[mask].mean():5.1%}  {mixup}")


if __name__ == "__main__":
    main()
