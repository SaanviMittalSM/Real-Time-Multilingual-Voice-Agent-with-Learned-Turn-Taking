import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dataset import TurnDataset, build_vocab  # noqa: E402
from models.turn_detector.model import LABELS, TurnDetector  # noqa: E402


def class_weights(records):
    counts = np.zeros(len(LABELS))
    for r in records:
        counts[LABELS.index(r["label"])] += 1
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    for mel_spec, token_ids, aux, label in loader:
        mel_spec, token_ids, aux, label = (
            mel_spec.to(device), token_ids.to(device), aux.to(device), label.to(device)
        )
        if train:
            optimizer.zero_grad()
        logits = model(mel_spec, token_ids, aux)
        loss = criterion(logits, label)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(label)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(label.cpu().tolist())
    return total_loss / len(all_labels), all_preds, all_labels


def train(train_manifest, dev_manifest, train_audio_root, dev_audio_root,
          epochs=10, batch_size=32, lr=1e-3, device=None, checkpoint_dir=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir or config.MODEL_WEIGHTS_DIR / "turn_detector")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_records = json.load(open(train_manifest))
    vocab = build_vocab(train_records)
    with open(checkpoint_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)

    train_ds = TurnDataset(train_manifest, train_audio_root, vocab)
    dev_ds = TurnDataset(dev_manifest, dev_audio_root, vocab)
    num_workers = 4 if device != "cpu" else 0  # overlap disk-bound audio loading with GPU compute
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, persistent_workers=num_workers > 0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=num_workers > 0)

    model = TurnDetector(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_dev_loss = float("inf")
    history = []
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        dev_loss, dev_preds, dev_labels = run_epoch(model, dev_loader, optimizer, criterion, device, train=False)
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss})

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

    report_text = classification_report(dev_labels, dev_preds, target_names=LABELS, zero_division=0)
    report_dict = classification_report(dev_labels, dev_preds, target_names=LABELS, zero_division=0, output_dict=True)
    print("\nFinal dev-set classification report:")
    print(report_text)

    with open(checkpoint_dir / "training_results.json", "w") as f:
        json.dump({
            "history": history,
            "best_dev_loss": best_dev_loss,
            "final_dev_classification_report": report_dict,
            "n_train": len(train_records),
            "n_dev": len(dev_ds),
        }, f, indent=2)

    return model, vocab


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Train and validate on the dev set only, to sanity-check the pipeline.")
    args = parser.parse_args()

    manifests = config.DATA_DIR / "manifests"
    audio_root = config.DATA_DIR / "raw" / "ami_audio"

    if args.smoke_test:
        train(
            manifests / "turn_labels_dev.json", manifests / "turn_labels_dev.json",
            audio_root / "dev", audio_root / "dev",
            epochs=args.epochs, batch_size=args.batch_size,
        )
    else:
        train(
            manifests / "turn_labels_train.json", manifests / "turn_labels_dev.json",
            audio_root / "train", audio_root / "dev",
            epochs=args.epochs, batch_size=args.batch_size,
        )
