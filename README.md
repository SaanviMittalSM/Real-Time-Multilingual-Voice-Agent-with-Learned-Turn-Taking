# Real-Time Multilingual Voice Agent with Learned Turn-Taking

> **Status: Phase 1/2 — scaffolding + baseline pipeline under construction.**
> No experiments have been run yet. Every metric placeholder below is explicitly marked
> "not yet measured" and will only be filled in once produced by the code in this repo
> (see [Reproducibility](#reproducibility)). No numbers on this page are estimated or
> assumed — if it's not measured, it's not written down.

## Key Results

| Metric | Value |
|---|---|
| Turn Detection F1 (learned vs. fixed VAD) | not yet measured |
| ASR WER — English / Hindi / Hinglish | not yet measured |
| p50 / p95 / p99 end-to-end latency | not yet measured |
| INT8 CPU speedup vs FP32 | not yet measured |
| Task completion rate (agent + tools) | not yet measured |

## Problem & Motivation

Most "voice agent" projects wire a VAD silence threshold to Whisper to an LLM to a TTS
call and stop there. The turn-taking logic — deciding *when* the user has actually
finished speaking — is left to a fixed silence timer (e.g. 500ms), which is either too
slow (the agent feels laggy) or too fast (it interrupts the user mid-thought,
mid-pause-to-think, or during a backchannel like "mm-hmm").

**Central question this project investigates:** can a learned, multimodal turn detector
(acoustic features + partial ASR transcript + conversational context) beat fixed
silence thresholds on both turn-detection accuracy *and* end-to-end response latency,
while staying cheap enough to run on CPU?

## Architecture

```
User
  -> WebSocket/WebRTC audio gateway
  -> VAD / preprocessing
  -> Learned turn detector  (audio encoder + text encoder -> fusion -> turn-state classifier)
  -> Streaming multilingual ASR
  -> Agent / LLM orchestrator
  -> Hybrid retrieval (BM25 + dense + reranker) / MCP tools / LLM
  -> Streaming TTS
  -> User
```

Latency and evaluation instrumentation runs alongside every stage (`turn_detection_ms`,
`asr_ms`, `retrieval_ms`, `llm_ttft_ms`, `tts_ttfa_ms`, `total_ms`).

## Datasets

Evaluated **separately per language** — English, Hindi, and Hinglish/code-switched are
never averaged into a single "multilingual" number.

**Turn-taking (English): AMI Meeting Corpus**, official "scenario-only" train/dev/test
split, CC BY 4.0. Switchboard/Fisher/CALLHOME (the other commonly-cited turn-taking
corpora) require paid LDC licenses and weren't used; Common Voice/FLEURS are
single-speaker read speech with no turn-taking behavior to learn from, so they're not
usable for this part despite being on the original candidate list.

AMI has no dedicated "turn" annotation layer — turn/hold/backchannel labels are derived
from per-speaker forced-aligned word timestamps plus AMI's dialogue-act annotations
(which do directly tag backchannels). See `training/build_turn_labels.py` for the exact
label-construction logic and its docstring for the methodology this follows.

Hindi/Hinglish turn-taking data is an open gap — no equivalent public corpus is known
yet; Phase 4 will need to either find one or construct a small eval set manually.

## Model Choices

| Component | Model | Status |
|---|---|---|
| ASR | faster-whisper (CTranslate2 Whisper) | integrated, not yet benchmarked |
| VAD baseline | Silero VAD | integrated, not yet benchmarked |
| Turn detector | custom audio+text fusion classifier | not yet built (Phase 3) |
| LLM orchestrator | local model via Ollama | integrated, not yet benchmarked |
| TTS | Piper (standalone binary) | integrated, not yet benchmarked |
| Reranker | TBD | not yet integrated (Phase 5) |

Every entry above will be updated to distinguish **inference/integration** vs.
**fine-tuned** vs. **trained from scratch** as each component is actually built — no
component is described as more than what was actually done to it.

## Baselines

Fixed VAD/silence thresholds (300ms / 500ms / 700ms / 900ms) — implemented first, before
any learned model, so the learned turn detector has something concrete to beat.

## Experiments & Ablations

10 experiments are planned (see project tracking); each will report its own
baseline-vs-treatment comparison with real numbers, not before/after prose claims:

1. Fixed VAD thresholds vs. learned turn detector

   **Preliminary result (fixed-VAD side only, AMI dev set, n=7889 utterances,
   learned-model comparison still pending):**

   | threshold | precision | recall | F1 | false interruption rate | missed turn rate |
   |---|---|---|---|---|---|
   | 300ms | 0.548 | 0.807 | 0.653 | 0.964 | 0.193 |
   | 500ms | 0.521 | 0.692 | 0.594 | 0.923 | 0.308 |
   | 700ms | 0.501 | 0.610 | 0.550 | 0.881 | 0.390 |
   | 900ms | 0.480 | 0.531 | 0.504 | 0.836 | 0.469 |

   Reproduce: `python training/download_ami.py dev && python training/build_turn_labels.py dev && python evaluation/turn_eval.py dev`

   False-interruption rate here means: of all utterances where the same
   speaker was just pausing (not actually done), what fraction had a pause
   long enough that a fixed threshold would wrongly think they were done.
   84-96% is strikingly high — same-speaker thinking pauses in real meeting
   speech routinely exceed even 900ms, which is direct empirical evidence
   for the problem this project exists to solve. Caveat: dev-set only, and
   turn/hold/backchannel labels are derived from AMI word timestamps + DA
   tags via a first-pass heuristic (see `training/build_turn_labels.py`),
   not an independently-verified gold turn-taking annotation — treat as
   preliminary until cross-checked against the literature's label
   methodology and validated on the test split.
2. Audio-only vs. text-only vs. audio+text fusion
3. English vs. Hindi vs. Hinglish
4. Clean vs. noisy audio
5. Pretrained vs. fine-tuned ASR
6. Dense vs. BM25 vs. hybrid retrieval
7. Reranker vs. no reranker
8. FP32 vs. INT8 (CPU)
9. Agent with tools vs. LLM without tools
10. Fixed threshold vs. learned turn detector — end-to-end latency impact

## Latency & CPU Benchmarks

Not yet measured. Will report P50/P95/P99 per pipeline stage plus CPU utilization,
memory, and throughput once the baseline pipeline (Phase 2) and optimization pass
(Phase 6) are complete.

## Failure Cases & Limitations

To be documented as they're found during evaluation — this section will stay honest
about where the system breaks (cross-talk, code-switch mid-sentence, tool
hallucination, etc.) rather than presenting only favorable results.

## Reproducibility

```bash
# Environment: all model caches/weights live on D: — see .env.example
cp .env.example .env

python -m venv venv   # or use the D:-based venv referenced in .env
pip install -r requirements-baseline.txt

# Ollama (local LLM) must be running separately:
ollama pull llama3.1:8b-instruct-q4_K_M

# Piper (local TTS) binary + voice model live under $PIPER_VOICES_DIR
```

Full setup, dataset download, training, and evaluation commands will be filled in as
each phase is built — see `training/`, `evaluation/`, and `optimization/` for the
scripts backing every number that eventually appears in this README.

## Project Structure

```
data/           raw / processed / manifests   (junctioned to D: — see .env)
models/         turn_detector / asr / reranker
training/       train_turn_detector.py, evaluate_turn_detector.py, configs/
inference/      asr.py, turn_detector.py, diarization.py, tts.py
agent/          orchestrator.py, tools.py, retrieval.py, guardrails.py
evaluation/     asr_eval.py, turn_eval.py, retrieval_eval.py, agent_eval.py, latency_eval.py
serving/        api.py, websocket.py, schemas.py
optimization/   quantization.py, benchmark.py
tests/
docker/
notebooks/
```
