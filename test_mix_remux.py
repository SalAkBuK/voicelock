#!/usr/bin/env python
"""
VoiceLock Phase 4: mix + lossless remux.

Combines two already-extracted tracks (e.g. from test_dual_extraction.py)
back into a single output, blends a small amount of the original raw audio
back in so it doesn't sound like a "dead room," loudness-normalizes the
result, and remuxes it into the original video container without
re-encoding video (if the source has a video stream; otherwise just writes
the mixed audio).

Usage:
    python test_mix_remux.py --source original.mp4 \
        --host-track test_output_host.wav --guest-track test_output_guest.wav \
        [--ambience 0.075] [--output test_output_mixed.mp4]

Note on sample rate: the TSE model's tracks are 8kHz, capped at roughly a
4kHz frequency ceiling by its training bandwidth. Upsampling here does NOT
recover frequencies that were never captured -- it does not fix the
muffled/telephone quality of the extracted voices themselves. It's done
for two unrelated reasons: the mix must share a sample rate with the
full-bandwidth raw ambience audio before they can be blended, and a final
deliverable should sit at a standard rate rather than 8kHz. Genuine voice
fidelity improvement is future scope (see CLAUDE.md).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf

from test_extraction import normalize_to_wav

OUTPUT_SAMPLE_RATE = 48000
PEAK_SAFETY_THRESHOLD = 0.98
PEAK_SAFETY_TARGET = 0.95


def ensure_peak_safe(audio: np.ndarray, label: str) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > PEAK_SAFETY_THRESHOLD:
        scale = PEAK_SAFETY_TARGET / peak
        print(f"  {label}: peak {peak:.3f} exceeds {PEAK_SAFETY_THRESHOLD}, scaling by {scale:.3f}")
        return audio * scale
    print(f"  {label}: peak {peak:.3f}, within safe range")
    return audio


def has_video_stream(path: Path) -> bool:
    probe = ffmpeg.probe(str(path))
    return any(s.get("codec_type") == "video" for s in probe.get("streams", []))


def apply_loudnorm(input_path: Path, output_path: Path, sample_rate: int) -> dict:
    """Single-pass EBU R128 loudness normalization via ffmpeg's loudnorm filter.

    loudnorm internally oversamples for true-peak detection (e.g. 48kHz -> 192kHz),
    and that inflated rate leaks into the output unless explicitly forced back down.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", "loudnorm=print_format=summary",
        "-ar", str(sample_rate),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg loudnorm failed, see error above")
    return result.stderr


def run_mix_remux(
    source_path,
    host_audio: np.ndarray,
    guest_audio: np.ndarray,
    track_sr: int,
    ambience: float = 0.075,
    output_path=None,
):
    """
    Run the full mix + lossless remux pipeline in-process. `host_audio` and
    `guest_audio` are already-extracted track arrays (e.g. from
    run_dual_extraction), not file paths -- callers like a UI don't need to
    round-trip through disk for the tracks.

    Returns (output_path, loudnorm_stats_text).
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"No such file: {source_path}")

    video_present = has_video_stream(source_path)
    if output_path is None:
        output_path = Path("test_output_mixed.mp4" if video_present else "test_output_mixed.wav")
    else:
        output_path = Path(output_path)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        n = min(len(host_audio), len(guest_audio))
        mix = host_audio[:n] + guest_audio[:n]

        print("Peak-safety check #1 (after summing host + guest):")
        mix = ensure_peak_safe(mix, "combined mix")

        print(f"Upsampling mix from {track_sr}Hz to {OUTPUT_SAMPLE_RATE}Hz...")
        mix_path = tmp / "mix.wav"
        mix_upsampled_path = tmp / "mix_upsampled.wav"
        sf.write(str(mix_path), mix, track_sr)
        normalize_to_wav(mix_path, mix_upsampled_path, OUTPUT_SAMPLE_RATE)
        mix_up, _ = sf.read(str(mix_upsampled_path), dtype="float32")

        print(f"Extracting and resampling source audio to {OUTPUT_SAMPLE_RATE}Hz for ambience bleed...")
        raw_path = tmp / "raw_audio.wav"
        normalize_to_wav(source_path, raw_path, OUTPUT_SAMPLE_RATE)
        raw, _ = sf.read(str(raw_path), dtype="float32")

        m = min(len(mix_up), len(raw))
        mix_up = mix_up[:m]
        raw = raw[:m]

        print(f"Blending {ambience*100:.1f}% raw ambience into the mix...")
        final = mix_up * (1.0 - ambience) + raw * ambience

        print("Peak-safety check #2 (after ambience blend):")
        final = ensure_peak_safe(final, "final blend")

        final_path = tmp / "final_prenorm.wav"
        sf.write(str(final_path), final, OUTPUT_SAMPLE_RATE)

        print("Applying loudness normalization (EBU R128)...")
        normalized_audio_path = tmp / "final_normalized.wav"
        loudnorm_stats = apply_loudnorm(final_path, normalized_audio_path, OUTPUT_SAMPLE_RATE)
        for line in loudnorm_stats.splitlines():
            if any(key in line for key in ("Input", "Output", "Normalization")):
                print(f"  {line.strip()}")

        if video_present:
            print("Remuxing into video container (lossless video passthrough)...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(source_path),
                "-i", str(normalized_audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-ar", str(OUTPUT_SAMPLE_RATE),
                "-shortest",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
                raise RuntimeError("ffmpeg remux failed, see error above")
        else:
            print("Source has no video stream -- writing mixed audio directly.")
            import shutil
            shutil.copyfile(normalized_audio_path, output_path)

    return output_path, loudnorm_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original input media file")
    parser.add_argument("--host-track", required=True, help="Extracted host track wav")
    parser.add_argument("--guest-track", required=True, help="Extracted guest track wav")
    parser.add_argument(
        "--ambience", type=float, default=0.075,
        help="Fraction of raw original audio blended back in (0-0.20 range, default 0.075)",
    )
    parser.add_argument("--output", default=None, help="Output path (default: test_output_mixed.mp4 or .wav)")
    args = parser.parse_args()

    host, host_sr = sf.read(args.host_track, dtype="float32")
    guest, guest_sr = sf.read(args.guest_track, dtype="float32")
    if host_sr != guest_sr:
        raise ValueError(f"host/guest sample rates differ: {host_sr} vs {guest_sr}")

    output_path, _ = run_mix_remux(
        args.source, host, guest, host_sr,
        ambience=args.ambience, output_path=args.output,
    )

    print()
    print(f"Wrote {output_path}")
    print("Done. Please confirm by ear: both speakers audible, ambience present but")
    print("not overwhelming, no 'dead room' effect, and (if video) it plays correctly.")


if __name__ == "__main__":
    main()
