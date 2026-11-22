# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "pyarrow", "huggingface_hub", "remotezip"]
# ///
"""Builds the Guess the game dataset: 9 games, 64x64 RGB, with a leakage-safe test split.

Usage: uv run --no-config ml/prepare_games.py (downloads about 2.8 GB into ml/data/raw/ the
first time). All three sources are CC BY 4.0.
- gameplay/: Bingsu/Gameplay_Images parquet files. Fortnite, Minecraft, Among Us, Roblox,
  Terraria. Frames are stored in video order with no video ids, so each game is cut into
  10 blocks of 100 frames; blocks 4 and 9 are test, and 10 frames on each side of a test
  block are dropped so neighbouring frames can't leak.
- gamed/: ybashir/gamed-game-classification-dataset. CS2, Valorant, Dota 2. Keeps its own
  split: train videos for training, its separate eval videos for test.
- zenodo/: 4 video folders pulled out of zenodo.org/records/15574186 (vit_images.zip,
  Katsaitis), without downloading the whole 1.2 GB zip.
  CS2, Valorant and Dota 2 are one extra video each, added to training for variety.
  League of Legends is the only League source: one video, 150 frames. First 110 train,
  last 30 test, 10 dropped in between. Its test score is optimistic (same match).

Writes ml/data/cache/games64.npz: images (N,64,64,3 uint8), labels, source, split.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
from remotezip import RemoteZip

RAW = Path("ml/data/raw")
CLASSES = ["fortnite", "minecraft", "among_us", "roblox", "terraria", "valorant", "cs2", "dota2", "league"]
SIZE = 64


ZENODO_URL = "https://zenodo.org/records/15574186/files/vit_images.zip?download=1"
ZENODO_FOLDERS = {"vit_images/test/FPS/Valorant- 44 Kills": "valorant", "vit_images/test/MOBA/Dota 2 ": "dota2",
                  "vit_images/test/MOBA/League of Legends": "league", "vit_images/train/FPS/Counter Strike 2": "cs2"}


def download():
    # Each source downloads once; delete its folder in ml/data/raw/ to fetch it again.
    if not (RAW / "gameplay").exists():
        snapshot_download("Bingsu/Gameplay_Images", repo_type="dataset", local_dir=RAW / "gameplay", allow_patterns="data/*")
    if not (RAW / "gamed").exists():
        snapshot_download("ybashir/gamed-game-classification-dataset", repo_type="dataset", local_dir=RAW / "gamed",
                          allow_patterns=["train/frames_224.zip", "eval/frames.zip"])
    with RemoteZip(ZENODO_URL) as z:  # One request per file, so this part is slow (~10 min).
        for name in z.namelist():
            label = ZENODO_FOLDERS.get("/".join(name.split("/")[:4]))
            target = RAW / "zenodo" / str(label) / Path(name).name
            if label and not name.endswith("/") and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))


def shrink(data):
    # Squash to a square, like Gamed's own 224x224 frames, so the whole HUD stays in view.
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


def gameplay():
    names = ["Among Us", "Apex Legends", "Fortnite", "Forza Horizon", "Free Fire", "Genshin Impact",
             "God of War", "Minecraft", "Roblox", "Terraria"]
    keep = {"Fortnite": "fortnite", "Minecraft": "minecraft", "Among Us": "among_us", "Roblox": "roblox", "Terraria": "terraria"}
    frames = {label: [] for label in keep.values()}
    for path in sorted((RAW / "gameplay").rglob("*.parquet")):
        table = pq.read_table(path)
        for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
            if names[label] in keep:
                frames[keep[names[label]]].append(shrink(image["bytes"]))
    for label, images in frames.items():
        for i, image in enumerate(images):
            block, offset = divmod(i, 100)
            if block in (4, 9):
                yield image, label, "gameplay", "test"
            elif not (block + 1 in (4, 9) and offset >= 90) and not (block - 1 == 4 and offset < 10):
                yield image, label, "gameplay", "train"


def gamed():
    for split, archive in [("train", "train/frames_224.zip"), ("test", "eval/frames.zip")]:
        with zipfile.ZipFile(RAW / "gamed" / archive) as z:
            for name in sorted(z.namelist()):
                if name.endswith(".jpg"):
                    yield shrink(z.read(name)), Path(name).parent.name, "gamed", split


def zenodo():
    for label in ["cs2", "valorant", "dota2"]:
        for path in sorted((RAW / "zenodo" / label).glob("*.jpg")):
            yield shrink(path.read_bytes()), label, "zenodo", "train"
    for i, path in enumerate(sorted((RAW / "zenodo" / "league").glob("*.jpg"))):
        if i < 110:
            yield shrink(path.read_bytes()), "league", "zenodo", "train"
        elif i >= 120:
            yield shrink(path.read_bytes()), "league", "zenodo", "test"


def main():
    download()
    rows = [*gameplay(), *gamed(), *zenodo()]
    images, labels, sources, splits = zip(*rows)
    Path("ml/data/cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("ml/data/cache/games64.npz", images=np.stack(images), labels=np.array(labels),
                        source=np.array(sources), split=np.array(splits))
    print(f"{'game':10s} {'train':>6s} {'test':>6s}")
    for label in CLASSES:
        counts = [sum(1 for _, l, _, s in rows if l == label and s == split) for split in ("train", "test")]
        print(f"{label:10s} {counts[0]:6d} {counts[1]:6d}")


if __name__ == "__main__":
    main()
