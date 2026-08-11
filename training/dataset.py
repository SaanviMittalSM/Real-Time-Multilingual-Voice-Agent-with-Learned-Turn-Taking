import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from models.turn_detector.model import LABEL_TO_IDX  # noqa: E402

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z']+")
TAIL_SECONDS = 0.5  # window at the END of the utterance where pitch/energy trend is measured
N_AUX_FEATURES = 6  # duration, energy_tail, energy_full, pitch_tail_mean, pitch_tail_slope, voiced_fraction


def extract_acoustic_features(audio: torch.Tensor, sample_rate: int):
    """Engineered pitch/energy features over the tail of the utterance.

    Falling pitch and dropping energy right before a pause are classic
    turn-yielding cues in the turn-taking literature - this gives the model
    explicit access to them rather than hoping the CNN discovers the same
    pattern in a raw mel-spectrogram from only ~41k training examples.
    """
    tail_samples = int(TAIL_SECONDS * sample_rate)
    tail = audio[-tail_samples:] if len(audio) >= tail_samples else audio

    energy_tail = tail.pow(2).mean().sqrt().item()
    energy_full = audio.pow(2).mean().sqrt().item()

    # detect_pitch_frequency never returns exactly 0 for unvoiced/silent frames - on pure
    # silence it returns a degenerate ~2667Hz value, and on noise it returns near-random
    # values, neither of which is a real pitch. Gate to plausible human voice F0 instead.
    pitch = AF.detect_pitch_frequency(tail.unsqueeze(0), sample_rate).squeeze(0)
    voiced = pitch[(pitch >= 60) & (pitch <= 400)]
    if len(voiced) >= 2:
        pitch_mean = voiced.mean().item()
        pitch_slope = (voiced[-1] - voiced[0]).item()
    elif len(voiced) == 1:
        pitch_mean = voiced.item()
        pitch_slope = 0.0
    else:
        pitch_mean = 0.0
        pitch_slope = 0.0
    voiced_fraction = len(voiced) / max(len(pitch), 1)

    return [
        np.log1p(energy_tail), np.log1p(energy_full),
        pitch_mean / 100.0, pitch_slope / 100.0,  # scaled roughly to unit range (F0 ~85-400Hz)
        voiced_fraction,
    ]


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(records, max_size=2000):
    counts = Counter()
    for r in records:
        counts.update(tokenize(r["text"]))
    vocab = {PAD: 0, UNK: 1}
    for word, _ in counts.most_common(max_size - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


class TurnDataset(Dataset):
    def __init__(self, manifest_path, audio_root, vocab, max_audio_seconds=2.0,
                 n_mels=40, sample_rate=16000, max_text_tokens=30):
        records = json.load(open(manifest_path))
        # backchannel records have no pause_after_s / next-turn context computed;
        # they're still a valid class to recognize from audio+text alone.
        self.records = records
        self.audio_root = Path(audio_root)
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_seconds * sample_rate)
        self.max_text_tokens = max_text_tokens
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.records)

    def _load_audio_window(self, meeting_id, channel, end_time):
        wav_path = self.audio_root / meeting_id / f"{meeting_id}.Headset-{channel}.wav"
        end_sample = int(end_time * self.sample_rate)
        start_sample = max(0, end_sample - self.max_audio_samples)
        n_frames = end_sample - start_sample
        try:
            audio, _ = sf.read(str(wav_path), start=start_sample, frames=n_frames, dtype="float32")
        except Exception:
            audio = np.zeros(self.max_audio_samples, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # some AMI headset files are stereo; force mono
        if len(audio) < self.max_audio_samples:
            pad = np.zeros(self.max_audio_samples - len(audio), dtype="float32")
            audio = np.concatenate([pad, audio])
        return torch.from_numpy(audio)

    def _tokenize(self, text):
        tokens = tokenize(text)[: self.max_text_tokens]
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
        if len(ids) < self.max_text_tokens:
            ids = ids + [self.vocab[PAD]] * (self.max_text_tokens - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        r = self.records[idx]
        audio = self._load_audio_window(r["meeting_id"], r["headset_channel"], r["utterance_end"])
        mel_spec = self.db(self.mel(audio))  # (n_mels, time)

        token_ids = self._tokenize(r["text"])

        duration = r["utterance_end"] - r["utterance_start"]
        acoustic_features = extract_acoustic_features(audio, self.sample_rate)
        aux = torch.tensor([np.log1p(duration)] + acoustic_features, dtype=torch.float32)

        label = torch.tensor(LABEL_TO_IDX[r["label"]], dtype=torch.long)
        return mel_spec, token_ids, aux, label
