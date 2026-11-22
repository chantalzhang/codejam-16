# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Exports a prepared color image dataset (ml/prepare_*.py) as small files for phones.

Usage: uv run --no-config ml/export_images.py games   (or brainrot)

Writes public/data/<name>-train-{0..}.bin (class-balanced slices, one per phone),
<name>-test.bin (held-out images for accuracy, up to 100 per class) and <name>-display.bin
(4 more held-out images shown on screen). Each file: N images of uint8 RGB pixels (row by
row), then N uint8 labels, like the Fashion-MNIST files. Images are 32x32, except display
images, which are 64x64 so people can see them; the model averages them down to 32x32.
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path("public/data")
SIZE = 32
DATASETS = {
    # 10 slices of 111 images per game (~3 MB each). Small games repeat across slices;
    # League (110 training frames) is in every slice.
    "games": {"cache": "ml/data/cache/games64.npz", "slices": 10, "per_class": 111,
              "classes": ["fortnite", "minecraft", "among_us", "roblox", "terraria", "valorant", "cs2", "dota2", "league"],
              # Hand-picked for being recognizable on screen: class -> position in its test images.
              "display": {"minecraft": 18, "fortnite": 109, "among_us": 109, "cs2": 177}},
    # Small dataset: one slice with 122 images per character (the smallest class), ~1.9 MB.
    "brainrot": {"cache": "ml/data/cache/brainrot64.npz", "slices": 1, "per_class": 122,
                 "classes": ["ballerina_cappuccina", "bombardino_crocodilo", "cappuccino_assassino", "tralalero_tralala", "tung_tung_sahur"],
                 "display": {"ballerina_cappuccina": 13, "bombardino_crocodilo": 13, "cappuccino_assassino": 6, "tralalero_tralala": 4}},
}


def write(name, images, labels):
    path = OUT / f"{name}.bin"
    path.write_bytes(images.tobytes() + labels.astype(np.uint8).tobytes())
    print(f"{path}: {len(labels)} images, {path.stat().st_size / 1e6:.2f} MB")


def main(name):
    config = DATASETS[name]
    data = np.load(config["cache"])
    # 64x64 -> 32x32 by averaging 2x2 blocks.
    full = data["images"]
    images = full.reshape(-1, SIZE, 2, SIZE, 2, 3).mean((2, 4)).round().astype(np.uint8)
    labels = np.array([config["classes"].index(label) for label in data["labels"]])
    train, test = data["split"] == "train", data["split"] == "test"
    OUT.mkdir(parents=True, exist_ok=True)

    # Slice s takes the next per_class images of each class, wrapping around small classes.
    rng = np.random.default_rng(0)
    pools = [rng.permutation(np.flatnonzero(train & (labels == c))) for c in range(len(config["classes"]))]
    for s in range(config["slices"]):
        picked = np.concatenate([pool[np.arange(s * config["per_class"], (s + 1) * config["per_class"]) % len(pool)]
                                 for pool in pools])
        picked = rng.permutation(picked)
        write(f"{name}-train-{s}", images[picked], labels[picked])

    # Display: the listed held-out images. Accuracy: up to 100 evenly spaced test images per
    # class from the rest, so every test video counts and the display images aren't in it.
    display = np.array([np.flatnonzero(test & (labels == config["classes"].index(label)))[position]
                        for label, position in config["display"].items()])
    accuracy = []
    for c in range(len(config["classes"])):
        pool = np.setdiff1d(np.flatnonzero(test & (labels == c)), display)
        accuracy.extend(pool[np.linspace(0, len(pool) - 1, min(100, len(pool))).round().astype(int)])
    accuracy = np.array(accuracy)
    write(f"{name}-test", images[accuracy], labels[accuracy])
    write(f"{name}-display", full[display], labels[display])


if __name__ == "__main__":
    main(sys.argv[1])
