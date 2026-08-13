"""Evaluate a trained turn detector checkpoint on a held-out split.

Separate from train_turn_detector.py's dev-set reporting deliberately -
the test split should only be touched once model selection (architecture,
hyperparameters, regularization) is actually finished, not used to guide
iteration the way dev is.
"""

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dataset import N_AUX_FEATURES, N_AUX_FEATURES_PAUSE_AWARE, TurnDataset  # noqa: E402
from models.turn_detector.model import LABELS, TurnDetector  # noqa: E402


def evaluate(checkpoint_dir, manifest_path, audio_root, device=None, pause_aware=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)

    vocab = json.load(open(checkpoint_dir / "vocab.json"))
    ds = TurnDataset(manifest_path, audio_root, vocab, include_pause_feature=pause_aware)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    n_aux = N_AUX_FEATURES_PAUSE_AWARE if pause_aware else N_AUX_FEATURES
    model = TurnDetector(vocab_size=len(vocab), n_aux_features=n_aux).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel_spec, token_ids, aux, label in loader:
            mel_spec, token_ids, aux = mel_spec.to(device), token_ids.to(device), aux.to(device)
            logits = model(mel_spec, token_ids, aux)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(label.tolist())

    report_text = classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0, output_dict=True)
    return report_text, report_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "test"])
    parser.add_argument("--pause-aware", action="store_true")
    args = parser.parse_args()

    dir_name = "turn_detector_pause_aware" if args.pause_aware else "turn_detector"
    checkpoint_dir = config.MODEL_WEIGHTS_DIR / dir_name
    manifest_path = config.DATA_DIR / "manifests" / f"turn_labels_{args.split}.json"
    audio_root = config.DATA_DIR / "raw" / "ami_audio" / args.split

    report_text, report_dict = evaluate(checkpoint_dir, manifest_path, audio_root, pause_aware=args.pause_aware)
    print(f"Learned turn detector ({dir_name}) on AMI {args.split} set:")
    print(report_text)

    suffix = "_pause_aware" if args.pause_aware else ""
    out_path = config.DATA_DIR / "manifests" / f"learned_model_eval_{args.split}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Wrote {out_path}")
