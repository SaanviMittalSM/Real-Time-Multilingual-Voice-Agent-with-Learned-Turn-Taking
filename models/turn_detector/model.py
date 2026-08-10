"""Learned multimodal turn detector: audio encoder + text encoder -> fusion -> classifier.

Both encoders are small and trained from scratch (not fine-tuned foundation
models) - this is a deliberate choice, not a shortcut: the model has to be
cheap enough for CPU inference (the whole point of the project), and a
from-scratch model is one every design choice in this file can actually
explain, rather than inheriting behavior from an opaque pretrained encoder.

Classes: shift (finished) / hold_short (continuing) / hold_long (needs more
wait time) / backchannel. See training/build_turn_labels.py for how these
are derived from AMI annotations.
"""

import torch
import torch.nn as nn

LABELS = ["shift", "hold_short", "hold_long", "backchannel"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


class AudioEncoder(nn.Module):
    """Log-mel spectrogram -> small CNN -> pooled embedding."""

    def __init__(self, n_mels=40, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, mel_spec):
        # mel_spec: (batch, n_mels, time)
        x = mel_spec.unsqueeze(1)  # (batch, 1, n_mels, time)
        x = self.conv(x)
        x = self.pool(x).flatten(1)  # (batch, 32)
        return self.proj(x)  # (batch, embed_dim)


class TextEncoder(nn.Module):
    """Token ids (partial transcript) -> embedding -> GRU -> pooled embedding."""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        x = self.embedding(token_ids)
        _, h_n = self.gru(x)
        return h_n.squeeze(0)  # (batch, hidden_dim)


class TurnDetector(nn.Module):
    def __init__(self, vocab_size, n_mels=40, audio_embed_dim=64,
                 text_embed_dim=64, text_hidden_dim=64, n_aux_features=1,
                 fusion_hidden_dim=64, num_classes=len(LABELS)):
        super().__init__()
        self.audio_encoder = AudioEncoder(n_mels=n_mels, embed_dim=audio_embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim=text_embed_dim, hidden_dim=text_hidden_dim)
        fusion_input_dim = audio_embed_dim + text_hidden_dim + n_aux_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, mel_spec, token_ids, aux_features):
        audio_emb = self.audio_encoder(mel_spec)
        text_emb = self.text_encoder(token_ids)
        fused = torch.cat([audio_emb, text_emb, aux_features], dim=-1)
        return self.fusion(fused)
