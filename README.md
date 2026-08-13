# Real-Time Multilingual Voice Agent with Learned Turn-Taking

> **Status: Phase 3 in progress — baseline pipeline done, learned turn detector (pause-aware
> variant, threshold-tuned) beats the fixed-VAD baseline on held-out test (see Experiment 1).
> Zero-latency variant does not yet beat it.**
> Every metric on this page came from code in this repo (see
> [Reproducibility](#reproducibility)) — nothing here is estimated or assumed. Where a
> result is preliminary or not favorable, that's stated explicitly rather than omitted.

## Key Results

| Metric | Value |
|---|---|
| Turn Detection F1, fixed-VAD baseline (test) | 0.656 best-case (300ms) — see Experiment 1 |
| Turn Detection F1, learned model, pause-aware + threshold-tuned (test) | **0.750** — beats baseline; see Experiment 1 |
| Turn Detection F1, learned model, zero-latency argmax (test) | 0.61 — does not yet beat baseline; see Experiment 1 caveats |
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

   **Preliminary result (fixed-VAD side only, AMI dev set, n=7923 utterances,
   learned-model comparison still pending):**

   | threshold | precision | recall | F1 | false interruption rate | missed turn rate |
   |---|---|---|---|---|---|
   | 300ms | 0.583 | 0.818 | 0.681 | 0.988 | 0.182 |
   | 500ms | 0.554 | 0.709 | 0.622 | 0.965 | 0.291 |
   | 700ms | 0.532 | 0.629 | 0.576 | 0.938 | 0.371 |
   | 900ms | 0.510 | 0.554 | 0.531 | 0.901 | 0.446 |

   Reproduce: `python training/download_ami.py dev && python training/build_turn_labels.py dev && python evaluation/turn_eval.py dev`

   False-interruption rate here means: of all utterances where the same
   speaker was just pausing (not actually done), what fraction had a pause
   long enough that a fixed threshold would wrongly think they were done.
   90-99% is strikingly high — same-speaker thinking pauses in real meeting
   speech routinely exceed even 900ms, which is direct empirical evidence
   for the problem this project exists to solve. One likely contributor:
   AMI meeting speech (task-based, multi-party) probably has longer natural
   pauses than a 1:1 voice-agent conversation would, so absolute rates here
   may not transfer directly — the comparison that matters (fixed vs.
   learned, on the same data) still holds. Caveat: dev-set only, and
   turn/hold/backchannel labels are derived from AMI word timestamps + DA
   tags via a first-pass heuristic (see `training/build_turn_labels.py`),
   not an independently-verified gold turn-taking annotation — treat as
   preliminary until cross-checked against the literature's label
   methodology and validated on the test split.

   **Learned-model result, held-out test set (small from-scratch audio+text
   fusion model with explicit pitch/energy features, early-stopped at
   epoch 2 — see `training/train_turn_detector.py` and
   `training/evaluate_turn_detector.py`):**

   | class | precision | recall | F1 | support |
   |---|---|---|---|---|
   | shift | 0.61 | 0.61 | 0.61 | 4242 |
   | hold_short | 0.06 | 0.72 | 0.11 | 98 |
   | hold_long | 0.47 | 0.12 | 0.19 | 2790 |
   | backchannel | 0.61 | 0.93 | 0.74 | 1954 |

   Reproduce: `python training/precompute_acoustic_features.py test --workers 2 && python training/evaluate_turn_detector.py test`

   **Not a clean win over the fixed-VAD baseline.** Shift-detection F1
   (0.61 on test, 0.656 for the best fixed-VAD threshold on test) is
   consistent with the dev-set gap (0.64 vs. 0.68) - the model generalizes
   reasonably (it's not badly overfit) but doesn't beat the baseline. As
   with the dev result, the comparison isn't strictly apples-to-apples: the
   fixed-VAD baseline observes the *actual pause duration* before deciding
   (reactive), while the learned model predicts using only audio from
   *before* the pause starts (zero-latency). That's a harder task and the
   more interesting one for the project's actual goal, but it's still
   worth being direct that this architecture, as built so far, does not
   yet outperform a fixed threshold on accuracy alone.

   **Iteration history, including a negative result:**
   1. First run overfit almost immediately (best epoch 1, dev loss rose
      every epoch after). Fixed with dropout throughout both encoders,
      a smaller vocab (2000 words), and weight decay - training became
      healthy (multiple stable epochs) but shift-F1 only moved 0.63 → 0.64.
      Fixing overfitting made the *training* honest; it didn't raise the
      *model's* ceiling.
   2. Added explicit pitch/energy features (tail-window RMS energy, mean
      and slope of pitch, voiced-frame fraction) - motivated by the
      turn-taking literature's finding that falling pitch/energy are
      strong turn-yielding cues. **Result: no meaningful change** (dev
      shift-F1 0.64 → 0.64, hold_long F1 actually dropped 0.21 → 0.19).
      This is a real negative result, not hidden here: either these
      particular engineered features aren't adding information the CNN
      wasn't already extracting from the raw mel-spectrogram, or the
      fusion mechanism (simple concatenation) isn't using them well, or
      there just isn't enough training data (~41k examples) for the
      model to learn to rely on them. hold_long F1 (0.19) is still the
      weakest class and the one that matters most for this project's
      thesis - telling "long thinking pause, same speaker continues"
      apart from "real turn end."

   Also worth noting as an engineering lesson, not a research one: adding
   the pitch-feature extraction step without caching it turned a ~5-minute
   training run into one that pegged the CPU for 16+ hours before being
   caught and fixed (see `training/precompute_acoustic_features.py`) -
   profiling before assuming code is "fast enough" mattered here.

   3. Added a **pause-aware variant** (`--pause-aware` flag): gives the
      model the actual observed pause duration as an input feature, the
      same information a fixed-VAD threshold implicitly uses when it
      fires - this is the genuinely apples-to-apples comparison. Test-set
      result:

      | class | precision | recall | F1 | support |
      |---|---|---|---|---|
      | shift | 0.90 | 0.42 | 0.57 | 4242 |
      | hold_short | 0.24 | 0.73 | 0.37 | 98 |
      | hold_long | 0.55 | 0.93 | 0.69 | 2790 |
      | backchannel | 0.94 | 1.00 | 0.97 | 1954 |
      | **overall accuracy** | | | **0.70** | 9084 |

      Reproduce: `python training/train_turn_detector.py --epochs 25 --batch-size 32 --pause-aware && python training/evaluate_turn_detector.py test --pause-aware`

      **Surprising at first: giving the model the same information as the
      baseline raised overall accuracy from 0.53 to 0.70, but shift-F1 via
      raw argmax got *worse* (0.61 → 0.57), not better.** The model became
      very conservative about predicting "shift" (precision 0.90, recall
      only 0.42) and leaned on hold_long far more than before (recall
      0.93). Diagnosis: the class-weighted multi-class loss (needed
      because backchannel/hold_short are rare) optimizes for
      macro/weighted performance across all four classes, not specifically
      the shift-vs-not distinction that determines end-to-end latency -
      the training objective and the metric that matters aren't the same
      thing, so argmax's implicit 0.5 decision boundary was miscalibrated
      for this specific question.

   4. **Fix: tune the shift-probability decision threshold directly**
      (`training/tune_shift_threshold.py`) instead of relying on argmax.
      No retraining needed - just sweep thresholds on P(shift) using dev,
      pick the one that maximizes binary shift-vs-not F1, and report that
      threshold's performance on held-out test (never used for tuning):

      | | precision | recall | F1 |
      |---|---|---|---|
      | fixed-VAD baseline, best threshold (test) | 0.546 | 0.821 | **0.656** |
      | learned model, argmax (test) | 0.903 | 0.405 | 0.559 |
      | learned model, dev-tuned threshold=0.20 (test) | 0.624 | 0.940 | **0.750** |

      Reproduce: `python training/tune_shift_threshold.py`

      **This beats the fixed-VAD baseline** (0.750 vs. 0.656, +14%
      relative) - legitimately, with the threshold chosen on dev and only
      ever evaluated once on test. It confirms the original diagnosis
      exactly: the model already had the right information and the right
      architecture to beat fixed-VAD; what was actually broken was reading
      predictions off argmax instead of a properly calibrated decision
      threshold. This is the pause-aware (reactive) comparison, not the
      zero-latency one - see the caveats above about what that distinction
      means for actual deployment latency. Next: apply the same
      threshold-tuning fix to the zero-latency variant, where the same
      calibration issue likely also costs real performance.

   Next directions worth trying: threshold-tune the zero-latency variant
   the same way, a longer audio context window, and attention-based fusion
   instead of concatenation.
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
