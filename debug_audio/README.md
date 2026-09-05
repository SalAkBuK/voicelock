# Debug audio manifest

Scratch audio from testing/debugging each phase. Not tracked in git
(debug_audio/ is gitignored) except where specific files are force-added
for a one-off review.

## phase1/ — single-speaker TSE (Phase 1)

- `ref_clip.wav` — the sliced enrollment reference clip, isolated for a listen.
- `clean_target.wav` — clean, unmixed target speaker (ground truth).
- `mixture_0db.wav` — raw unprocessed 0dB two-speaker mixture, before extraction.
- `extracted_0db.wav` — first Phase 1 extraction result (original run, rounded ref-end).
- `extracted_0db_precise.wav` — same case, regenerated with the exact reference boundary.
- `extracted_0db_debug_dup.wav` — early debug duplicate of the raw model output.
- `extracted_0db_enrolled_target.wav` — enrolled with the target's own clip ("enrolledA").
- `extracted_0db_enrolled_interferer.wav` — enrolled with the interferer's clip instead ("enrolledB").
- `extracted_5db.wav` — extraction at +5dB SNR (easier case, for comparison).
- `extracted_0db_denoised.wav` — 0dB extraction + DeepFilterNet cleanup pass (made WER worse, see CLAUDE.md).
- `trivial_no_interferer.wav` — sanity check with no interferer at all (should be near-perfect passthrough).

## phase2/ — dual-speaker TSE (Phase 2)

- `host_track.wav` — extracted host track from the joint dual-speaker call.
- `guest_track.wav` — extracted guest track from the joint dual-speaker call.
- `guest_track_enrollment_window.wav` — isolated 5.8s-9.8s window of guest_track (checked directly for real audio).

## phase3/ — long-file chunking (Phase 3)

- `long_chunked.wav` — chunked pipeline output on the 78s test file (current, post-fix).
- `long_oneshot.wav` — one-shot/unchunked baseline on the same 78s file, for comparison.
- `seam58_before_fix.wav` — isolated 56s-61s region, before the crossfade fix (the "demonic" seam).
- `seam58_after_fix.wav` — isolated 56s-61s region, after the crossfade fix.
- `seam58_oneshot_control.wav` — same 56s-61s region sliced from the one-shot baseline, as a control.
