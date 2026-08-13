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
from dataset import N_AUX_FEATURES_PAUSE_AWARE, TurnDataset  # noqa: E402
from models.turn_detector.model import LABEL_TO_IDX, TurnDetector  # noqa: E402

SHIFT_IDX = LABEL_TO_IDX["shift"]


def get_shift_scores(checkpoint_dir, manifest_path, audio_root, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    vocab = json.load(open(checkpoint_dir / "vocab.json"))
    ds = TurnDataset(manifest_path, audio_root, vocab, include_pause_feature=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = TurnDetector(vocab_size=len(vocab), n_aux_features=N_AUX_FEATURES_PAUSE_AWARE).to(device)
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
    checkpoint_dir = config.MODEL_WEIGHTS_DIR / "turn_detector_pause_aware"
    manifests = config.DATA_DIR / "manifests"
    audio_root = config.DATA_DIR / "raw" / "ami_audio"

    dev_true, dev_score = get_shift_scores(
        checkpoint_dir, manifests / "turn_labels_dev.json", audio_root / "dev"
    )

    thresholds = np.arange(0.05, 0.96, 0.05)
    f1s = [f1_score(dev_true, dev_score >= t) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    print(f"Best threshold on dev: {best_t:.2f} (dev F1={max(f1s):.3f}, argmax-equivalent is t=0.5)")

    test_true, test_score = get_shift_scores(
        checkpoint_dir, manifests / "turn_labels_test.json", audio_root / "test"
    )
    for t, label in [(0.5, "argmax-equivalent"), (best_t, "dev-tuned")]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, test_score >= t, average="binary", zero_division=0
        )
        print(f"test @ threshold={t:.2f} ({label}): precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
