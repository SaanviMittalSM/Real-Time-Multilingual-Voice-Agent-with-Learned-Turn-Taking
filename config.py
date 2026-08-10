import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Must be set before any huggingface_hub / faster-whisper import happens downstream,
# otherwise the model cache defaults to C: (which has very little slack on this machine).
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(ROOT / "data" / "hf_cache")))

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODEL_WEIGHTS_DIR = Path(os.environ.get("MODEL_WEIGHTS_DIR", ROOT / "models"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

PIPER_BIN = os.environ.get("PIPER_BIN")
PIPER_VOICE = os.environ.get("PIPER_VOICE")

SAMPLE_RATE = 16000
