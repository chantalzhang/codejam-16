# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision"]
# ///
"""Export a Fashion-MNIST subset for phones to download (stand-in demo task).

Writes public/data/fashion-train-{0..4}.bin (5 slices of 2,000, so each phone downloads
1.6 MB instead of all 7.8 MB) and fashion-test.bin. Each file: N*784 uint8 pixels, then N
uint8 labels.
Fashion-MNIST is (c) Zalando SE, MIT License; the license is copied alongside.

Run: uv run ml/export_fashion.py
"""
from pathlib import Path
from urllib.request import urlopen

from torchvision.datasets import FashionMNIST

OUT = Path("public/data")
LICENSE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE"


def export(train, start, count, name):
    data = FashionMNIST("ml/data", train=train, download=True)
    images = data.data[start:start + count].reshape(count, -1).numpy().tobytes()
    labels = data.targets[start:start + count].numpy().astype("uint8").tobytes()
    path = OUT / f"{name}.bin"
    path.write_bytes(images + labels)
    print(f"{path}: {count} images, {path.stat().st_size / 1e6:.1f} MB")


OUT.mkdir(parents=True, exist_ok=True)
for slice_ in range(5):
    export(train=True, start=slice_ * 2_000, count=2_000, name=f"fashion-train-{slice_}")
export(train=False, start=0, count=1_000, name="fashion-test")
(OUT / "FASHION-MNIST-LICENSE").write_bytes(urlopen(LICENSE_URL).read())
