# VoiceLock — Project Memory

Read this fully at the start of every session. Do not deviate from the current
phase's scope without asking the user first.

## What this is

A local desktop/CLI tool that isolates one or two known speakers from noisy
outdoor recordings (e.g. Speakers' Corner debates) using pretrained Target
Speaker Extraction (TSE) models — not general noise reduction. It works by
enrolling a short clean voice sample per speaker, generating an acoustic
embedding, and using that embedding to mask out every other voice, including
hecklers who are louder than the target speaker.

## Non-negotiable constraints

- Never train a model from scratch. Use pretrained checkpoints only
  (ClearerVoice-Studio's `MossFormer2_TSE`, or WeSep's SpEx+/BSRNN).
- Do exactly what the current session's prompt asks — nothing more. If a task
  is genuinely blocked or ambiguous, stop and ask rather than guessing or
  expanding scope.
- Every new phase (see below) starts in Plan Mode. Propose the approach,
  wait for explicit approval, then execute.
- Commit to git after every task that produces a verified, working result.
  Never end a session with uncommitted, half-working state.
- No UI work of any kind until Phase 2 (dual-speaker extraction) is proven
  on real audio and the user has confirmed it sounds right.
- The user is a beginner. Explain what you're doing and why in plain terms,
  especially around git and ML dependency errors — don't assume familiarity.

## Build phases (do them in order — do not skip ahead)

| Phase | Goal | Definition of done |
|---|---|---|
| 0 | Environment + repo setup | Hardware detected and reported (CPU/CUDA/MPS) via `torch.cuda.is_available()` / `torch.backends.mps.is_available()`; git initialized; remote connected to the user's GitHub URL; first commit pushed. |
| 1 | Single-speaker proof of concept | `test_extraction.py` accepts either a YouTube URL or a local file path, plus a reference clip in/out timestamp. It downloads (if URL) or loads (if local file), runs `MossFormer2_TSE` against the full clip using the reference as the target, and writes `test_output.wav`. User confirms by ear that the target voice is intelligible. |
| 2 | Dual-speaker extraction | Same pipeline, but enrolls two speakers (host + guest) and produces two isolated tracks. User confirms each track contains only its intended speaker. |
| 3 | Long-file chunking | Sliding-window inference (~30s windows, 2s cross-fade) so hour-long files don't OOM. Verify with a synthetic sine-wave test that cross-fades don't produce audible clicks before testing on real audio. |
| 4 | Mix + lossless remux | Combine both isolated tracks with an ambience-bleed blend (5–10% of raw audio), then remux into the original video container with `ffmpeg -c:v copy` (no video re-encode). |
| 5 | Minimal UI | Streamlit: file drop, host profile dropdown, guest timestamp picker, ambience slider, process button. Only build this after Phase 2 is confirmed working. |

## Architecture

```
[Input: .mp4 / .mov / .wav]
        |
        v
[FFmpeg Demux + Normalize]  -- 16kHz mono PCM, EBU R128 loudness
        |
  +-----+------+
  |            |
  v            v
[Speaker A]  [Speaker B]      3-8s clean enrollment clips
  |            |
  v            v
[ECAPA-TDNN Embedding Engine]  -- acoustic fingerprint per speaker
  |            |
  +-----+------+
        v
[TSE Extraction Backbone: MossFormer2_TSE / SpEx+]
   Pass 1 -> host_clean.wav   Pass 2 -> guest_clean.wav
        |
        v
[Mix + Ambience Bleed (5-10%) + Loudness Balance]
        |
        v
[Lossless Remux: ffmpeg -c:v copy]  -> output.mp4
```

## Tech stack

- Python 3.10+
- PyTorch, ONNX Runtime
- `clearvoice` (ClearerVoice-Studio, `MossFormer2_TSE`) as primary backbone;
  `wesep` as fallback if clearvoice has install issues
- SpeechBrain or WeSep's ECAPA-TDNN for speaker embeddings
- `ffmpeg-python`, `torchaudio`, `soundfile`, `librosa` for audio/video I/O
- `yt-dlp` for pulling test clips
- Streamlit for the eventual UI (Phase 5 only)

## Functional requirements (MVP)

| ID | Feature | Requirement |
|---|---|---|
| FR-01 | Media Ingestion | Drag-and-drop or CLI arg: `.mp4`, `.mov`, `.mkv`, `.wav` up to 4GB |
| FR-02 | Host Profile Lock | Persistent embedding (`host_profile.npy`) saved once, reused across projects |
| FR-03 | Guest Enrollment | 3–8s clean segment marked by timestamp, embedding generated on the fly |
| FR-04 | Dual-Pass Extraction | Two isolated tracks, all non-enrolled voices suppressed |
| FR-05 | Ambience Bleed | 0–20% blend of raw audio back in, avoids a "dead room" sound |
| FR-06 | Lossless Remux | Audio replaced in original container without re-encoding video |

## Testing discipline

- Stage 1 (synthetic): mix clean speech (LibriSpeech/Common Voice) with crowd
  noise at +5dB / 0dB / −5dB. Must recover target voice even at 0dB.
- Stage 2 (real footage): a real YouTube or user-provided clip with actual
  heckling. Judge by ear — no automated metric substitutes for this at MVP
  stage.
- Every phase's "definition of done" above must be confirmed by the user
  before moving to the next phase. Do not self-declare a phase complete.

## Known issues

- **Generic denoising does not fix TSE residual artifacts.** Tried
  DeepFilterNet (DeepFilterNet3) as a post-extraction cleanup pass on the
  hardest Phase 1 test case (0dB two-speaker mixture). Measured by Whisper
  word-error-rate against the known ground-truth transcript, it made things
  *worse*, not better: 29.2% WER before cleanup -> 37.5% WER after. RNNoise
  was considered but not tried once DeepFilterNet's result came back negative.
  General-purpose denoisers (RNNoise, DeepFilterNet) are trained to remove
  stationary background noise (hiss, fans, room tone) — they are not suited
  to the residual cross-talk/artifacts a TSE model leaves behind, and appear
  to smooth over correct target-speaker content along with the artifacts.
  Don't retry generic denoising as a fix for TSE output quality. If artifact
  reduction is revisited later, it needs a TSE-specific approach (e.g. a
  second-pass TSE-aware post-filter, or a better/fine-tuned TSE checkpoint),
  not a generic noise suppressor.

## Session hygiene

- One phase (or sub-task within a phase) per session. Do not jump ahead.
- Before writing any code for a new phase: enter Plan Mode, state the
  approach, wait for approval.
- After a task is verified working: `git add -A && git commit -m "..."` with
  a clear message describing what now works, then stop and report back
  rather than continuing to the next phase unprompted.
