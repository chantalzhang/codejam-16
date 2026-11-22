# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision"]
# ///
"""Export a Fashion-MNIST subset for phones to download (stand-in demo task).

Writes public/data/fashion-{train,test}.bin: N*784 uint8 pixels, then N uint8 labels.
Fashion-MNIST is (c) Zalando SE, MIT License; the license is copied alongside.

Run: uv run ml/export_fashion.py
"""
from pathlib import Path
from urllib.request import urlopen

from torchvision.datasets import FashionMNIST

OUT = Path("public/data")
LICENSE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE"


def export(train, count):
    data = FashionMNIST("ml/data", train=train, download=True)
    images = data.data[:count].reshape(count, -1).numpy().tobytes()
    labels = data.targets[:count].numpy().astype("uint8").tobytes()
    path = OUT / f"fashion-{'train' if train else 'test'}.bin"
    path.write_bytes(images + labels)
    print(f"{path}: {count} images, {path.stat().st_size / 1e6:.1f} MB")


OUT.mkdir(parents=True, exist_ok=True)
export(train=True, count=10_000)
export(train=False, count=1_000)
(OUT / "FASHION-MNIST-LICENSE").write_bytes(urlopen(LICENSE_URL).read())
