"""Binary shift-vs-not threshold tuning for the pause-aware turn detector.

The 4-way argmax result showed shift-F1 dropping when the model was given
more information (pause duration), likely because class-weighted
multi-class training optimizes for macro/weighted performance across all
four classes, not specifically the shift-vs-not distinction that actually
matters for latency. This sidesteps that by using the model's raw
P(shift) score directly and picking the threshold that maximizes binary
F1 on dev - then reporting that threshold's performance on held-out test,
which is the correct (no test-set peeking) way to do it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dataset import N_AUX_FEATURES, N_AUX_FEATURES_PAUSE_AWARE, TurnDataset  # noqa: E402
from models.turn_detector.model import LABEL_TO_IDX, TurnDetector  # noqa: E402

SHIFT_IDX = LABEL_TO_IDX["shift"]


def get_shift_scores(checkpoint_dir, manifest_path, audio_root, device=None, pause_aware=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    vocab = json.load(open(checkpoint_dir / "vocab.json"))
    ds = TurnDataset(manifest_path, audio_root, vocab, include_pause_feature=pause_aware)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    n_aux = N_AUX_FEATURES_PAUSE_AWARE if pause_aware else N_AUX_FEATURES
    model = TurnDetector(vocab_size=len(vocab), n_aux_features=n_aux).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    model.eval()

    y_true, y_score = [], []
    with torch.no_grad():
        for mel_spec, token_ids, aux, label in loader:
            mel_spec, token_ids, aux = mel_spec.to(device), token_ids.to(device), aux.to(device)
            probs = torch.softmax(model(mel_spec, token_ids, aux), dim=-1)
            y_score.extend(probs[:, SHIFT_IDX].cpu().tolist())
            y_true.extend((label == SHIFT_IDX).long().tolist())
    return np.array(y_true), np.array(y_score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pause-aware", action="store_true", default=True)
    parser.add_argument("--zero-latency", dest="pause_aware", action="store_false",
                         help="Tune the zero-latency variant instead (predicts before the pause starts)")
    args = parser.parse_args()

    dir_name = "turn_detector_pause_aware" if args.pause_aware else "turn_detector"
    checkpoint_dir = config.MODEL_WEIGHTS_DIR / dir_name
    manifests = config.DATA_DIR / "manifests"
    audio_root = config.DATA_DIR / "raw" / "ami_audio"
    print(f"Tuning {dir_name} (pause_aware={args.pause_aware})")

    dev_true, dev_score = get_shift_scores(
        checkpoint_dir, manifests / "turn_labels_dev.json", audio_root / "dev", pause_aware=args.pause_aware
    )

    thresholds = np.arange(0.05, 0.96, 0.05)
    f1s = [f1_score(dev_true, dev_score >= t) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    print(f"Best threshold on dev: {best_t:.2f} (dev F1={max(f1s):.3f})")

    test_true, test_score = get_shift_scores(
        checkpoint_dir, manifests / "turn_labels_test.json", audio_root / "test", pause_aware=args.pause_aware
    )
    # NOTE: t=0.5 on P(shift) is NOT the same as 4-way argmax - argmax only needs shift to be
    # the largest of four probabilities, not to exceed 0.5 absolute, so these two can differ
    # substantially (and did, for the zero-latency variant). t=0.5 here means exactly that:
    # the raw, uncalibrated binary threshold, not "what argmax would have predicted."
    for t, label in [(0.5, "raw threshold=0.5"), (best_t, "dev-tuned")]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, test_score >= t, average="binary", zero_division=0
        )
        print(f"test @ threshold={t:.2f} ({label}): precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
