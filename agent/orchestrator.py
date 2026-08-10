import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from ollama import Client  # noqa: E402

SYSTEM_PROMPT = (
    "You are a voice assistant. Keep replies short (1-3 sentences) and "
    "conversational, since your response will be spoken aloud by a TTS engine."
)


class Orchestrator:
    """Phase 2 baseline: plain chat, no tools/retrieval yet (added in Phase 5)."""

    def __init__(self, model=None, host=None):
        self.model = model or config.OLLAMA_MODEL
        self.client = Client(host=host or config.OLLAMA_HOST)
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.history = self.history[:1]

    def respond(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        start = time.perf_counter()
        response = self.client.chat(model=self.model, messages=self.history)
        elapsed_ms = (time.perf_counter() - start) * 1000
        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "llm_ms": elapsed_ms}


if __name__ == "__main__":
    orchestrator = Orchestrator()
    print(orchestrator.respond("Say hello in one short sentence."))
