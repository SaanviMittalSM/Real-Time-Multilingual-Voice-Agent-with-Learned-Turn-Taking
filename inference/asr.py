import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (sets HF_HOME before faster_whisper touches the HF cache)

from faster_whisper import WhisperModel  # noqa: E402


class ASR:
    """Thin wrapper around faster-whisper (CTranslate2 Whisper).

    device="auto" lets CTranslate2 pick CUDA if available; the CPU-inference
    benchmarks in Phase 6 will force device="cpu" explicitly.
    """

    def __init__(self, model_size="small", device="auto", compute_type="default"):
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(config.DATA_DIR / "hf_cache" / "faster-whisper"),
        )

    def transcribe(self, audio, language=None):
        """audio: path to a wav file, or a 1-D float32 numpy array at 16kHz."""
        start = time.perf_counter()
        segments, info = self.model.transcribe(audio, language=language, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "asr_ms": elapsed_ms,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe a wav file with faster-whisper")
    parser.add_argument("wav_path")
    parser.add_argument("--model-size", default="small")
    args = parser.parse_args()

    asr = ASR(model_size=args.model_size)
    result = asr.transcribe(args.wav_path)
    print(result)
