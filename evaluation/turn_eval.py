"""Experiment 1 (partial): fixed VAD silence thresholds vs. ground truth.

Evaluates whether a fixed-threshold rule ("the floor changed hands iff the
pause exceeded N ms") matches what actually happened in the AMI dev set,
using the turn labels built by build_turn_labels.py. This is the baseline
the learned turn detector (once trained) has to beat.

Framed as binary: true "shift" vs. everything else (hold_short/hold_long/
backchannel_response all collapse to "not a real turn end"), since that's
what a silence-threshold VAD is capable of distinguishing in the first
place — it has no way to tell a backchannel from a real turn change.
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

THRESHOLDS_MS = [300, 500, 700, 900]


def evaluate_fixed_threshold(records, threshold_ms):
    y_true, y_pred = [], []
    for r in records:
        y_true.append(1 if r["label"] == "shift" else 0)
        pause_ms = (r["pause_after_s"] or 0) * 1000
        y_pred.append(1 if pause_ms >= threshold_ms else 0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # false interruption: predicted shift (agent would respond) but truth was hold/backchannel
    false_interruptions = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    n_negatives = sum(1 for t in y_true if t == 0)
    false_interruption_rate = false_interruptions / n_negatives if n_negatives else 0.0

    # missed turn: predicted hold (agent stays silent) but truth was a real shift
    missed_turns = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n_positives = sum(1 for t in y_true if t == 1)
    missed_turn_rate = missed_turns / n_positives if n_positives else 0.0

    return {
        "threshold_ms": threshold_ms,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_interruption_rate": false_interruption_rate,
        "missed_turn_rate": missed_turn_rate,
        "n": len(records),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test"])
    args = parser.parse_args()

    manifest_path = config.DATA_DIR / "manifests" / f"turn_labels_{args.split}.json"
    records = json.load(open(manifest_path))
    records = [r for r in records if r["pause_after_s"] is not None]

    results = [evaluate_fixed_threshold(records, t) for t in THRESHOLDS_MS]

    print(f"Fixed-VAD-threshold baseline on AMI {args.split} set (n={len(records)} utterances)\n")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'false_interrupt':>16} {'missed_turn':>12}")
    for r in results:
        print(
            f"{r['threshold_ms']:>9}ms {r['precision']:>10.3f} {r['recall']:>10.3f} "
            f"{r['f1']:>10.3f} {r['false_interruption_rate']:>16.3f} {r['missed_turn_rate']:>12.3f}"
        )

    out_path = config.DATA_DIR / "manifests" / f"fixed_vad_baseline_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
