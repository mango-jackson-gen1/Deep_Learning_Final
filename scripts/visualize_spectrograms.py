"""Dump PNGs of the exact mel spectrograms the ResNet18 backbone sees.

Picks one track per genre by default and saves both:
  - the raw 224x224 grayscale image (what the backbone actually receives)
  - a colored matplotlib version for human eyes

Usage:
    python scripts/visualize_spectrograms.py
    python scripts/visualize_spectrograms.py --per-genre 3
    python scripts/visualize_spectrograms.py --track-ids 5 193 80293
"""

import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt

from src.audio_utils import get_audio_path
from src.config import PROCESSED_DIR
from src.embeddings.spectrogram import audio_to_mel_spectrogram
from src.metadata import load_tracks

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = PROCESSED_DIR / "viz" / "spectrograms"


def pick_tracks(per_genre: int, seed: int) -> list:
    df = load_tracks()
    df.columns = ["_".join(c).strip("_") for c in df.columns]
    canonical = set(np.load(PROCESSED_DIR / "spectrogram_track_ids.npy").tolist())
    df = df[df.index.isin(canonical)]
    rng = np.random.default_rng(seed)
    picks = []
    for genre, group in df.groupby("track_genre_top"):
        n = min(per_genre, len(group))
        picks.extend((int(tid), str(genre)) for tid in rng.choice(group.index.values, size=n, replace=False))
    return picks


def save_one(track_id: int, genre: str) -> None:
    audio_path = get_audio_path(track_id)
    spec = audio_to_mel_spectrogram(str(audio_path))  # (224, 224), float32 in [0, 1]

    safe_genre = genre.replace("/", "_").replace(" ", "_")
    stem = OUT_DIR / f"{track_id:06d}_{safe_genre}"

    # 1) The exact grayscale image fed to ResNet18 (post-resize, post-normalize)
    Image.fromarray((spec * 255).astype(np.uint8)).save(f"{stem}_input.png")

    # 2) Pretty version with axes + colorbar so you can read it
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"track {track_id:06d} — {genre}", fontsize=10)
    ax.set_xlabel("time (resized to 224)")
    ax.set_ylabel("mel bins (resized to 224)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized log-mel")
    fig.tight_layout()
    fig.savefig(f"{stem}.png")
    plt.close(fig)
    logger.info(f"saved {stem.name}.png + {stem.name}_input.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-genre", type=int, default=1, help="tracks to sample per genre (default 1)")
    p.add_argument("--track-ids", type=int, nargs="*", help="explicit track IDs (overrides --per-genre)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.track_ids:
        df = load_tracks()
        df.columns = ["_".join(c).strip("_") for c in df.columns]
        picks = [(tid, str(df.loc[tid, "track_genre_top"]) if tid in df.index else "Unknown") for tid in args.track_ids]
    else:
        picks = pick_tracks(args.per_genre, args.seed)

    logger.info(f"writing {len(picks)} spectrograms → {OUT_DIR}")
    for tid, genre in picks:
        try:
            save_one(tid, genre)
        except Exception as e:
            logger.warning(f"track {tid} failed: {e}")


if __name__ == "__main__":
    main()
