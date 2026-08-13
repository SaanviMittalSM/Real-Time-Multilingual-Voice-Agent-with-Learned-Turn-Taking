"""Phase 4: ASR WER/CER benchmarked separately per language - English, Hindi,
and Hinglish (code-switched) are never averaged into one "multilingual"
number, per the project's explicit rule against that.

Datasets (all freely accessible, no gating):
  - English:  LibriSpeech test-clean (OpenSLR 12) - standard clean read speech
  - Hindi:    FLEURS hi_in test split (HuggingFace google/fleurs) - clean read speech
  - Hinglish: MUCS 2021 Hindi-English test set (OpenSLR 104) - real code-switched
              speech from spoken tutorials, Kaldi-style segments format

Audio is decoded manually via soundfile rather than through HF `datasets`'
automatic Audio() decoding, which requires torchcodec - torchcodec on this
machine only supports FFmpeg 4-8, and the available FFmpeg install is v9.
Sidestepping it avoids a fragile version-pinned system dependency.
"""

import io
import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from inference.asr import ASR  # noqa: E402

# WER needs word-tokenized input; CER needs normalized strings (character-level comparison
# on a list-of-words would compare word boundaries as if they were characters, which is wrong).
# Both share the same text normalization (lowercase, strip punctuation) - only the final
# reduction step differs. Missing this for CER previously produced nonsense: English WER=0.024
# but CER=0.842, purely from comparing ALL-CAPS unpunctuated LibriSpeech references against
# normal-case punctuated Whisper output with no normalization at all.
_BASE_NORMALIZE = [
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
]
WER_NORMALIZE = jiwer.Compose(_BASE_NORMALIZE + [jiwer.ReduceToListOfListOfWords()])
CER_NORMALIZE = jiwer.Compose(_BASE_NORMALIZE + [jiwer.ReduceToListOfListOfChars()])


def iter_english_samples(max_samples=None):
    """Yields {"audio", "sr", "reference"} one at a time - the machine here has only
    15GB RAM and materializing 100 samples' worth of decoded audio into a list
    at once (as an earlier version of this script did) triggered allocation
    failures, so audio is read lazily and never held onto past one transcription."""
    root = config.DATA_DIR / "raw" / "librispeech" / "LibriSpeech" / "test-clean"
    n = 0
    for trans_file in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            if max_samples and n >= max_samples:
                return
            utt_id, text = line.split(" ", 1)
            flac_path = trans_file.parent / f"{utt_id}.flac"
            audio, sr = sf.read(str(flac_path), dtype="float32")
            yield {"audio": audio, "sr": sr, "reference": text}
            n += 1


def iter_hindi_samples(max_samples=None):
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", "hi_in", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, rec in enumerate(ds):
        if max_samples and i >= max_samples:
            return
        audio, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
        yield {"audio": audio, "sr": sr, "reference": rec["transcription"]}


def iter_hinglish_samples(max_samples=None):
    root = config.DATA_DIR / "raw" / "mucs_hinglish" / "test"
    seg_map = {}
    for line in (root / "transcripts" / "segments").read_text(encoding="utf-8").splitlines():
        utt_id, rec_id, start, end = line.split()
        seg_map[utt_id] = (rec_id, float(start), float(end))

    n = 0
    for line in (root / "transcripts" / "text").read_text(encoding="utf-8").splitlines():
        if max_samples and n >= max_samples:
            return
        utt_id, text = line.split(" ", 1)
        rec_id, start, end = seg_map[utt_id]
        wav_path = root / f"{rec_id}.wav"
        info = sf.info(str(wav_path))
        start_sample, end_sample = int(start * info.samplerate), int(end * info.samplerate)
        audio, sr = sf.read(
            str(wav_path), start=start_sample, frames=end_sample - start_sample, dtype="float32"
        )
        # Whisper has no distinct "Hinglish" language code - transcribe with
        # language="hi" and let it naturally handle the embedded English words,
        # which is exactly the real-world deployment condition being tested.
        yield {"audio": audio, "sr": sr, "reference": text}
        n += 1


def evaluate_language(name, sample_iter, asr, whisper_language):
    references, hypotheses = [], []
    for s in sample_iter:
        result = asr.transcribe(s["audio"], language=whisper_language)
        references.append(s["reference"])
        hypotheses.append(result["text"])

    wer = jiwer.wer(references, hypotheses, reference_transform=WER_NORMALIZE, hypothesis_transform=WER_NORMALIZE)
    cer = jiwer.cer(references, hypotheses, reference_transform=CER_NORMALIZE, hypothesis_transform=CER_NORMALIZE)
    return {"language": name, "n_samples": len(references), "wer": wer, "cer": cer}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100,
                         help="Cap per language for a tractable first pass - not silently, this is logged.")
    parser.add_argument("--model-size", default="small")
    args = parser.parse_args()

    print(f"Loading faster-whisper ({args.model_size})...")
    asr = ASR(model_size=args.model_size)

    results = []
    for name, iter_fn, whisper_lang in [
        ("english", iter_english_samples, "en"),
        ("hindi", iter_hindi_samples, "hi"),
        ("hinglish", iter_hinglish_samples, "hi"),
    ]:
        print(f"\n[{name}] streaming up to {args.max_samples} samples (capped - see --max-samples)...")
        result = evaluate_language(name, iter_fn(max_samples=args.max_samples), asr, whisper_lang)
        results.append(result)
        print(f"[{name}] WER={result['wer']:.3f}  CER={result['cer']:.3f}  n={result['n_samples']}")

    print("\n=== Summary (never averaged across languages) ===")
    for r in results:
        print(f"{r['language']:>10}: WER={r['wer']:.3f}  CER={r['cer']:.3f}  (n={r['n_samples']})")

    out_path = config.DATA_DIR / "manifests" / "asr_multilingual_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
