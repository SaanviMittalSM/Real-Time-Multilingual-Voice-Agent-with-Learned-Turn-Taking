"""Phase 2 baseline: audio -> VAD -> ASR -> LLM -> TTS, running locally.

Not the target streaming/WebSocket architecture (that's Phase 8) — this is
the simplest possible end-to-end loop, built first so every later piece
(learned turn detector, retrieval, agent tools, CPU optimization) has a
working baseline to be compared against and slotted into.

Push-to-talk-by-silence: speak, pause past the VAD threshold, and the loop
transcribes -> gets an LLM reply -> speaks it back. Ctrl+C to quit.
"""

import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from agent.orchestrator import Orchestrator
from inference.asr import ASR
from inference.tts import PiperTTS
from inference.turn_detector import WINDOW_SAMPLES, FixedThresholdTurnDetector


def record_one_turn(turn_detector, sample_rate):
    """Blocks until one full speech turn (VAD start -> VAD end) is captured."""
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    turn_detector.reset()
    speech_chunks = []
    recording = False

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=WINDOW_SAMPLES,
        callback=callback,
    ):
        print("Listening...")
        while True:
            chunk = audio_q.get()
            event = turn_detector.process_chunk(chunk)
            if event == "speech_start":
                recording = True
                print("  (speech detected)")
            if recording:
                speech_chunks.append(chunk)
            if event == "speech_end":
                break

    return np.concatenate(speech_chunks) if speech_chunks else np.array([], dtype="float32")


def main():
    threshold_ms = 500
    print(f"Loading models (turn detector threshold={threshold_ms}ms)...")
    turn_detector = FixedThresholdTurnDetector(threshold_ms=threshold_ms, sample_rate=config.SAMPLE_RATE)
    asr = ASR(model_size="small")
    orchestrator = Orchestrator()
    tts = PiperTTS()
    print("Ready. Speak, then pause to end your turn. Ctrl+C to quit.\n")

    while True:
        try:
            t0 = time.perf_counter()
            audio = record_one_turn(turn_detector, config.SAMPLE_RATE)
            if len(audio) < config.SAMPLE_RATE * 0.2:
                continue

            asr_result = asr.transcribe(audio)
            user_text = asr_result["text"]
            if not user_text:
                continue
            print(f"You said: {user_text}")

            llm_result = orchestrator.respond(user_text)
            print(f"Agent: {llm_result['reply']}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out_wav = f.name
            tts_result = tts.synthesize(llm_result["reply"], out_wav)

            wav_data, wav_sr = sf.read(out_wav)
            sd.play(wav_data, wav_sr)
            sd.wait()

            total_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  [latency] asr={asr_result['asr_ms']:.0f}ms "
                f"llm={llm_result['llm_ms']:.0f}ms "
                f"tts={tts_result['tts_ms']:.0f}ms "
                f"total={total_ms:.0f}ms\n"
            )
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
