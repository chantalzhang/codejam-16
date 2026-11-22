# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "imagehash"]
# ///
"""Builds the Italian brainrot dataset: 5 characters, 64x64 RGB, deduplicated.

Usage: uv run --no-config ml/prepare_brainrot.py (downloads 62 MB into ml/data/raw/ the
first time). Source: kaggle.com/datasets/bubblepw/italian-brainrot-images (license not
stated; images are scraped fan art and merch).

The two folders in the archive overlap, and many images are repeats or near repeats, so
images are grouped by perceptual hash and each group kept once. Groups that appear under
more than one character (placeholder thumbnails, pictures of two characters) are dropped.
Then 20% of each character goes to test.

Writes ml/data/cache/brainrot64.npz: images (N,64,64,3 uint8), labels, source, split.
"""
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

RAW = Path("ml/data/raw/brainrot")
CLASSES = ["ballerina_cappuccina", "bombardino_crocodilo", "cappuccino_assassino", "tralalero_tralala", "tung_tung_sahur"]
SIZE = 64


def download():
    if RAW.exists():
        return
    archive = RAW.with_suffix(".zip")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://www.kaggle.com/api/v1/datasets/download/bubblepw/italian-brainrot-images", archive)
    zipfile.ZipFile(archive).extractall(RAW)


def main():
    download()
    files = sorted(RAW.rglob("*.jpg"))
    label = {f: f.parent.name.replace("tung_tung_tung", "tung_tung") for f in files}
    hashes = {f: imagehash.phash(Image.open(f).convert("RGB")) for f in files}
    # Union near-identical images (hash distance <= 6) into groups.
    parent = {f: f for f in files}
    def root(f):
        while parent[f] != f:
            f = parent[f]
        return f
    for i, a in enumerate(files):
        for b in files[i + 1:]:
            if hashes[a] - hashes[b] <= 6:
                parent[root(a)] = root(b)
    groups = defaultdict(list)
    for f in files:
        groups[root(f)].append(f)

    kept = defaultdict(list)
    mixed = 0
    for members in groups.values():
        names = {label[f] for f in members}
        if len(names) > 1:
            mixed += 1
            continue
        kept[names.pop()].append(members[0])

    random.seed(0)
    images, labels, splits = [], [], []
    print(f"{len(files)} files, {len(groups)} unique, {mixed} mixed-character groups dropped")
    print(f"{'character':22s} {'train':>6s} {'test':>6s}")
    for name in CLASSES:
        members = sorted(kept[name])
        random.shuffle(members)
        test = len(members) // 5
        for i, f in enumerate(members):
            images.append(np.asarray(Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)))
            labels.append(name)
            splits.append("test" if i < test else "train")
        print(f"{name:22s} {len(members) - test:6d} {test:6d}")
    Path("ml/data/cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("ml/data/cache/brainrot64.npz", images=np.stack(images), labels=np.array(labels),
                        source=np.array(["kaggle"] * len(labels)), split=np.array(splits))


if __name__ == "__main__":
    main()
