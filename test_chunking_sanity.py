#!/usr/bin/env python
"""
VoiceLock Phase 3 sanity check: does the sliding-window chunking + cross-fade
mechanism itself introduce clicks?

This does NOT involve the ML model at all -- it isolates the windowing/
cross-fade math in chunked_extract() by running it with an identity
process_fn on a long synthetic sine wave. With identity processing, the
reconstructed signal should exactly reproduce the original; any deviation
at a window seam would show up as a click.

Usage:
    python test_chunking_sanity.py
"""

import numpy as np

from test_extraction import chunked_extract

SR = 8000
DURATION_SEC = 90.0  # spans 3+ windows at 30s/window, 2s crossfade
FREQ_HZ = 440.0
WINDOW_SEC = 30.0
CROSSFADE_SEC = 2.0


def identity(chunk):
    return [chunk]


def main():
    t = np.arange(int(DURATION_SEC * SR), dtype=np.float64) / SR
    original = (0.5 * np.sin(2 * np.pi * FREQ_HZ * t)).astype(np.float32)

    [reconstructed] = chunked_extract(
        identity, original, SR, window_sec=WINDOW_SEC, crossfade_sec=CROSSFADE_SEC
    )

    # Primary check: with an identity process_fn, cross-fading the SAME
    # signal against itself should reproduce it exactly (fade_out + fade_in
    # == 1 everywhere). Any deviation means a real bug in the windowing math.
    max_abs_error = float(np.max(np.abs(reconstructed - original)))

    # Diagnostic: explicit click score at each seam. A click shows up as a
    # sample-to-sample jump (first difference) far outside the sine wave's
    # own smooth, bounded slope. Compare seam-region max |diff| against the
    # signal's overall max |diff| (its natural slope ceiling).
    diff = np.abs(np.diff(reconstructed))
    natural_max_slope = float(np.max(np.abs(np.diff(original))))

    window_len = int(WINDOW_SEC * SR)
    crossfade_len = int(CROSSFADE_SEC * SR)
    hop_len = window_len - crossfade_len
    total_len = len(original)
    starts = sorted(set(list(range(0, total_len - window_len, hop_len)) + [total_len - window_len]))
    seam_indices = [starts[i - 1] + window_len for i in range(1, len(starts))]

    print(f"Signal: {DURATION_SEC:.0f}s sine at {FREQ_HZ}Hz, {SR}Hz sample rate")
    print(f"Window: {WINDOW_SEC:.0f}s, crossfade: {CROSSFADE_SEC:.0f}s, seams at: {[f'{s/SR:.2f}s' for s in seam_indices]}")
    print()
    print(f"Max abs reconstruction error (reconstructed vs. original): {max_abs_error:.2e}")
    print(f"Natural max sample-to-sample slope of the clean sine wave: {natural_max_slope:.6f}")

    seam_click_scores = []
    for seam in seam_indices:
        lo, hi = max(0, seam - 5), min(len(diff), seam + 5)
        seam_max_diff = float(np.max(diff[lo:hi]))
        seam_click_scores.append(seam_max_diff)
        ratio = seam_max_diff / natural_max_slope
        print(f"  seam at {seam/SR:.2f}s: max |diff| nearby = {seam_max_diff:.6f} ({ratio:.2f}x natural slope)")

    reconstruction_ok = max_abs_error < 1e-4
    no_clicks = all(score <= natural_max_slope * 1.05 for score in seam_click_scores)

    print()
    if reconstruction_ok and no_clicks:
        print("PASS: chunking/cross-fade mechanism reproduces the signal exactly, no clicks at seams.")
    else:
        print("FAIL:")
        if not reconstruction_ok:
            print(f"  - reconstruction error too high ({max_abs_error:.2e} >= 1e-4)")
        if not no_clicks:
            print("  - at least one seam shows a slope spike above the sine wave's natural maximum")

    return 0 if (reconstruction_ok and no_clicks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
