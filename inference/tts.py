import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class PiperTTS:
    """Wraps the standalone Piper binary as a subprocess.

    The `piper-tts` pip package's `piper-phonemize` dependency has no
    prebuilt wheel for Python 3.12 on Windows, so this calls the official
    Piper binary release directly instead of using Python bindings.
    """

    def __init__(self, piper_bin=None, voice=None):
        self.piper_bin = Path(piper_bin or config.PIPER_BIN)
        self.voice = Path(voice or config.PIPER_VOICE)
        if not self.piper_bin.exists():
            raise FileNotFoundError(f"Piper binary not found: {self.piper_bin}")
        if not self.voice.exists():
            raise FileNotFoundError(f"Piper voice model not found: {self.voice}")

    def synthesize(self, text: str, output_wav: str):
        start = time.perf_counter()
        subprocess.run(
            [str(self.piper_bin), "--model", str(self.voice), "--output_file", output_wav],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"output_wav": output_wav, "tts_ms": elapsed_ms}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthesize speech with Piper")
    parser.add_argument("text")
    parser.add_argument("--out", default="tts_output.wav")
    args = parser.parse_args()

    tts = PiperTTS()
    print(tts.synthesize(args.text, args.out))
