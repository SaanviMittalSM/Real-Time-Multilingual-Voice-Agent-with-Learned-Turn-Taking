"""Precompute pitch/energy features once per manifest, instead of recomputing
them from scratch every training epoch.

detect_pitch_frequency costs ~21ms/call (measured) and dominates
__getitem__ cost (mel spectrogram is ~0.7ms by comparison). The audio
window for a given utterance never changes across epochs, so recomputing
this every epoch is pure waste - this was the actual cause of a training
run that took 16+ hours instead of the expected tens of minutes. Cache it
once (parallelized across CPU cores) and look it up at O(1) during training.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dataset import TurnDataset, extract_acoustic_features  # noqa: E402

_worker_ds = None


def _init_worker(manifest_path, audio_root):
    global _worker_ds
    # vocab doesn't matter here - only used for audio loading, not tokenization.
    # acoustic_features_cache=False so this doesn't try to load a stale/partial cache
    # from a previous run while we're generating a fresh one.
    _worker_ds = TurnDataset(manifest_path, audio_root, vocab={"<pad>": 0, "<unk>": 1},
                              acoustic_features_cache=False)


def _compute_one(idx):
    r = _worker_ds.records[idx]
    audio = _worker_ds._load_audio_window(r["meeting_id"], r["headset_channel"], r["utterance_end"])
    return extract_acoustic_features(audio, _worker_ds.sample_rate)


def precompute(manifest_path, audio_root, out_path, n_workers=8):
    records = json.load(open(manifest_path))
    print(f"Precomputing acoustic features for {len(records)} records from {manifest_path}...")

    with Pool(n_workers, initializer=_init_worker, initargs=(manifest_path, audio_root)) as pool:
        results = pool.map(_compute_one, range(len(records)), chunksize=64)

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifests = config.DATA_DIR / "manifests"
    audio_root = config.DATA_DIR / "raw" / "ami_audio" / args.split
    manifest_path = manifests / f"turn_labels_{args.split}.json"
    out_path = manifests / f"acoustic_features_{args.split}.json"

    precompute(manifest_path, audio_root, out_path, n_workers=args.workers)
