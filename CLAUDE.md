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
  (currently ESPnet2 TD-SpeakerBeam via `espnet_model_zoo`; see Tech
  Stack section for the specific checkpoint tag in use).
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
| 1 | Single-speaker proof of concept — **DONE** | `test_extraction.py` accepts either a YouTube URL or a local file path, plus a reference clip in/out timestamp. It downloads (if URL) or loads (if local file), runs ESPnet2 TD-SpeakerBeam against the full clip using the reference as the target, and writes `test_output.wav`. Actual result on a synthetic 0dB two-speaker test: SI-SDR improvement 8.89dB (clears the checkpoint's published >=7dB bar), Whisper WER 100% (raw mixture, nonsense transcript) -> 29.2% (extracted output, recognizable and mostly correct) against a 16.7% clean-audio ceiling. An enrollment A/B swap test confirmed the model is genuinely steered by whichever speaker's clip is used as reference, not ignoring it. User confirmed the phase complete based on this evidence. |
| 2 | Dual-speaker extraction — **DONE** | `test_dual_extraction.py` enrolls two speakers (host + guest) and produces two isolated tracks. The hypothesized joint call (`enroll_ref1` + `enroll_ref2` in a single `separate_speech(...)` call, per the checkpoint's `model_conf.num_spk: 2`) worked exactly as expected — no fallback to sequential calls was needed. Actual result on a synthetic 0dB two-speaker test: SI-SDR track-swap cross-check came out clean (host track: 6.94dB vs. its own clean content, -21.98dB vs. the guest's — correct/incorrect split; guest track: -21.48dB vs. host's, 11.47dB vs. its own — correct/incorrect split), confirming no swapped or bled tracks. Whisper WER: host 45.8%, guest 27.6% (both clearly recognizable, rougher than Phase 1's single-speaker 29.2%, consistent with joint 2-speaker extraction being a harder task). User confirmed by ear that both tracks are correctly separated with no cross-bleed beyond what the known test-file timeline already explained. |
| 3 | Long-file chunking — **DONE** | Sliding-window inference (~30s windows, 2s cross-fade) so hour-long files don't OOM, added as `chunked_extract()` in `test_extraction.py` and used by both single- and dual-speaker scripts. Sine-wave sanity test (`test_chunking_sanity.py`) passed: reconstruction error 2.98e-08, seam slopes indistinguishable from the sine wave's own natural curve. Real-audio chunked-vs-one-shot comparison on a 78s two-speaker test: SI-SDR 9.17dB vs. 9.36dB (negligible difference), Whisper WER 21.6% vs. 25.7% (comparable). Found and fixed a real bug during this: end-of-file window alignment caused an oversized ~9.9s crossfade at one seam, causing severe phase-cancellation ("demonic") distortion — the sine-wave test couldn't catch this since it used identity processing (mathematically invariant to overlap length); only real-audio testing surfaced it. Fix caps the blended transition at the intended `crossfade_sec` regardless of the geometric overlap size, sourcing any extra overlap from the nearer window's output instead of blending across the full oversized span. Confirmed by ear post-fix: the seam sounds identical to a one-shot control at the same region — remaining roughness there is baseline model quality (the known Phase 1/2 issue), not a chunking artifact. |
| 4 | Mix + lossless remux — **DONE** | `test_mix_remux.py` sums the host + guest tracks, applies two peak-safety checks (post-sum and post-ambience-blend, capping at ~0.98 peak), upsamples to 48kHz (needed to blend with full-bandwidth ambience audio and land at a standard deliverable rate — does not itself improve extracted-voice fidelity, which stays capped by the model's 8kHz/~4kHz training bandwidth; see Known Issues/Future), blends in a 7.5% raw-audio ambience bleed (within FR-05's 0-20% range, confirmed by ear as sounding right), loudness-normalizes via ffmpeg's `loudnorm` (EBU R128), and remuxes into the source video container via `-c:v copy`. Prerequisite fix: `chunked_extract()` was also wired into `test_dual_extraction.py` (previously only single-speaker had it), verified bit-identical to the prior unchunked output on Phase 2's test case. Verified via `ffprobe`: output video's raw H.264 bitstream is byte-for-byte identical to the source's (true lossless passthrough, zero re-encode). Found and fixed a real bug: ffmpeg's `loudnorm` filter internally oversamples 4x for true-peak detection (48kHz -> 192kHz), and that inflated rate was leaking into the output until `-ar 48000` was pinned explicitly on both the loudnorm and remux steps. User confirmed by ear: correct structure (host intro, guest intro, then overlapping mixture), ambience level sounds right, video plays correctly. |
| 5 | Minimal UI | Streamlit: file drop, host profile dropdown, guest timestamp picker, ambience slider, process button. Only build this after Phase 2 is confirmed working. |

## Architecture

```
[Input: .mp4 / .mov / .wav]
        |
        v
[FFmpeg Demux + Normalize]  -- 8kHz mono PCM, EBU R128 loudness
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
[TSE Extraction Backbone: ESPnet2 TD-SpeakerBeam]
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
- `espnet` + `espnet_model_zoo` (ESPnet2 TD-SpeakerBeam) as the actual TSE
  backbone, via `SeparateSpeech.from_pretrained(model_tag=...)` — checkpoint
  tag `espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw`.
  `clearvoice` (ClearerVoice-Studio) is kept installed in `voicelock-env`
  only as a possible future speech-enhancement/post-processing tool, NOT
  for TSE — its only TSE model (`AV_MossFormer2_TSE_16K`) is audio-visual
  (needs video/lip cues, not a plain audio reference clip). WeSep was
  evaluated and dropped: it's a training-recipe toolkit with no pretrained
  checkpoints available, which conflicts with "never train from scratch."
- `ffmpeg-python`, `torchaudio`, `soundfile`, `librosa` for audio/video I/O
- `yt-dlp` for pulling test clips
- Streamlit for the eventual UI (Phase 5 only)

## Environment

Two separate virtual environments, both Python 3.11 (not the system's
3.13 — no prebuilt Windows wheels for numpy/librosa/clearvoice on 3.13 at
time of writing, and clearvoice hard-pins `numpy<2.0` which has no 3.13
wheel at all):

- `voicelock-env` — clearvoice, yt-dlp, ffmpeg-python, torch/torchaudio.
- `voicelock-tse-env` — espnet, espnet_model_zoo, torch/torchaudio (CUDA
  12.4 build), yt-dlp, soundfile, ffmpeg-python. This is where
  `test_extraction.py` actually runs.

Kept separate because `espnet` requires `numpy>=2.0` while `clearvoice`
pins `numpy<2.0` — installing both in the same venv forces a numpy
downgrade that silently breaks whichever package needed the newer version
(confirmed via `pip install --dry-run` before it ever hit a real
environment).

The system `ffmpeg` binary (not just the `ffmpeg-python` wrapper) is also
required and was not on PATH by default on this machine — installed via
`winget install Gyan.FFmpeg`.

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
- SI-SDR (vs. clean reference) and Whisper transcript word-error-rate (vs.
  known ground truth) are now standard fast, objective pre-checks — used
  successfully in Phase 1 to catch real signal in extraction output and to
  rule out pipeline bugs before ever needing to listen. Run these first.
  They do not replace ear confirmation as the actual definition of done —
  a track can score reasonably and still contain audible artifacts a human
  would flag (see Known issues).
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

- **Oversized end-of-file overlaps are a real edge case in overlap-add
  chunking.** `chunked_extract()`'s window-start generation forces the
  final window to align exactly with the file's end, which can make its
  geometric overlap with the previous window much larger than the intended
  `crossfade_sec` (observed: ~9.9s instead of ~2s on a 78s test file).
  Blending two independently-produced model outputs over that large a span
  caused severe phase-cancellation distortion at that seam. Fixed by
  capping the actual cross-faded transition at `crossfade_sec` regardless
  of the geometric overlap size. Worth a regression test if the window/hop
  sizing logic in `chunked_extract()` changes later — the sine-wave sanity
  test alone won't catch this class of bug (see Phase 3's row above).

- **Some players may have trouble with mono-AAC-in-mp4 output.** A Phase 4
  test file appeared to have no audio in the user's default player;
  confirmed via `ffprobe`/direct extraction that the file's audio was
  genuinely present, correctly mapped, and correctly encoded — VLC played
  it back correctly with no issues. The original playback problem was
  specific to that player, not a bug in the file or pipeline. If this
  comes up again for a real user, check player compatibility (try VLC)
  before assuming a pipeline bug.

## Future (out of scope for now)

- **Genuine audio bandwidth improvement for extracted voices.** The TSE
  model outputs 8kHz audio, capped at roughly a 4kHz frequency ceiling by
  its training bandwidth. Phase 4's upsampling to 48kHz does NOT recover
  this — it's needed only to blend with full-bandwidth ambience audio and
  to land at a standard deliverable rate, not to improve voice fidelity.
  Actually improving it would need a higher-native-rate TSE checkpoint, or
  a proper upsampling/super-resolution model applied to the extracted
  voice — separate, later scope from the mixing step.

## Project conventions

- **Debug audio** lives in `debug_audio/{phase1,phase2,phase3}/`, gitignored
  by default. Each phase's subfolder holds that phase's scratch test-output
  wav files with clear, short names (e.g. `seam58_before_fix.wav`); see
  `debug_audio/README.md` for a one-line manifest of what each file is.
  Force-add (`git add -f`) only the specific files actually needed for a
  given review, not the whole folder.

## Session hygiene

- One phase (or sub-task within a phase) per session. Do not jump ahead.
- Before writing any code for a new phase: enter Plan Mode, state the
  approach, wait for approval.
- After a task is verified working: `git add -A && git commit -m "..."` with
  a clear message describing what now works, then stop and report back
  rather than continuing to the next phase unprompted.
