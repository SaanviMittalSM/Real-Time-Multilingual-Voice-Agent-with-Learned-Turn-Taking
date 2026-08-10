import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

WINDOW_SAMPLES = 512  # silero-vad's required chunk size at 16kHz (32ms)


class FixedThresholdTurnDetector:
    """Baseline turn detector: fixed VAD silence threshold.

    This is the Phase 1 baseline (300/500/700/900ms fixed thresholds) that
    the learned multimodal turn detector (Phase 3) has to beat — on both
    turn-detection accuracy and end-to-end latency.
    """

    def __init__(self, threshold_ms=500, sample_rate=16000):
        self.threshold_ms = threshold_ms
        self.sample_rate = sample_rate
        self.model = load_silero_vad()
        self.iterator = VADIterator(
            self.model,
            sampling_rate=sample_rate,
            min_silence_duration_ms=threshold_ms,
        )
        self.in_speech = False

    def reset(self):
        self.iterator.reset_states()
        self.in_speech = False

    def process_chunk(self, chunk: np.ndarray):
        """chunk: 1-D float32 numpy array of exactly WINDOW_SAMPLES samples.

        Returns "speech_start", "speech_end", or None.
        """
        result = self.iterator(torch.from_numpy(chunk), return_seconds=False)
        if result is None:
            return None
        if "start" in result:
            self.in_speech = True
            return "speech_start"
        if "end" in result:
            self.in_speech = False
            return "speech_end"
        return None
