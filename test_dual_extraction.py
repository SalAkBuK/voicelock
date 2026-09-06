#!/usr/bin/env python
"""
VoiceLock Phase 2: dual-speaker extraction.

Given a mixture (YouTube URL or local file) and two short clean reference
clips -- one for the host, one for the guest -- extract both speakers'
voices using ESPnet2's pretrained TD-SpeakerBeam target speaker extraction
model.

Usage:
    python test_dual_extraction.py <input> \
        --host-start MM:SS --host-end MM:SS \
        --guest-start MM:SS --guest-end MM:SS \
        [--host-output test_output_host.wav] [--guest-output test_output_guest.wav]

<input> can be a YouTube URL or a local file path (.mp4/.mov/.mkv/.wav).
Both reference clips are timestamps within that same input.

This checkpoint's config shows model_conf.num_spk: 2, so it may support a
single joint call (enroll_ref1=host, enroll_ref2=guest) returning both
tracks at once. This script tries that first and falls back to two
sequential single-enrollment calls (Phase 1's proven pattern) if the joint
call doesn't return two usable results.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from test_extraction import (
    DEFAULT_MODEL_TAG,
    MODEL_SAMPLE_RATE,
    chunked_extract,
    download_audio,
    is_url,
    normalize_to_wav,
    parse_timestamp,
)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.squeeze(0).cpu().numpy()
    return np.squeeze(np.asarray(x), axis=0)


def slice_clip(mixture, sr, start_str, end_str, label):
    start_sec = parse_timestamp(start_str)
    end_sec = parse_timestamp(end_str)
    if end_sec <= start_sec:
        raise ValueError(f"--{label}-end must be after --{label}-start")
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    if end_sample > len(mixture):
        raise ValueError(
            f"{label} reference end ({end_sec:.1f}s) is beyond the audio length "
            f"({len(mixture) / sr:.1f}s)"
        )
    print(f"{label.capitalize()} reference clip: {end_sec - start_sec:.1f}s ({start_str} to {end_str})")
    return mixture[start_sample:end_sample]


def run_joint_call(separate_speech, mixture_batch, host_batch, guest_batch, sr):
    """Try a single call with both enrollments. Returns (host_arr, guest_arr) or None."""
    try:
        results = separate_speech(
            mixture_batch, fs=sr, enroll_ref1=host_batch, enroll_ref2=guest_batch
        )
    except Exception as exc:
        print(f"Joint call failed ({exc}); falling back to two sequential calls.")
        return None

    if len(results) != 2:
        print(
            f"Joint call returned {len(results)} result(s), expected 2; "
            "falling back to two sequential calls."
        )
        return None

    return _to_numpy(results[0]), _to_numpy(results[1])


def run_sequential_calls(separate_speech, mixture_batch, host_batch, guest_batch, sr):
    """Proven Phase 1 fallback: one call per speaker, each with only its own enrollment."""
    host_results = separate_speech(mixture_batch, fs=sr, enroll_ref1=host_batch)
    guest_results = separate_speech(mixture_batch, fs=sr, enroll_ref1=guest_batch)
    return _to_numpy(host_results[0]), _to_numpy(guest_results[0])


def run_dual_extraction(
    input_path,
    host_start: str,
    host_end: str,
    guest_start: str,
    guest_end: str,
    model_tag: str = DEFAULT_MODEL_TAG,
):
    """
    Run the full dual-speaker extraction pipeline in-process (no file I/O
    for the outputs -- returns arrays directly, so callers like a UI don't
    need to round-trip through disk). `input_path` may be a YouTube URL or
    a local file path.

    Returns (host_audio, guest_audio, sr).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected -- running on CPU, this will be noticeably slower.")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        if is_url(str(input_path)):
            source_path = download_audio(str(input_path), tmp)
        else:
            source_path = Path(input_path)
            if not source_path.exists():
                raise FileNotFoundError(f"No such file: {source_path}")

        normalized_path = tmp / "normalized.wav"
        normalize_to_wav(source_path, normalized_path, MODEL_SAMPLE_RATE)

        mixture, sr = sf.read(normalized_path, dtype="float32")
        assert sr == MODEL_SAMPLE_RATE, f"expected {MODEL_SAMPLE_RATE}Hz, got {sr}Hz"

        host_enroll = slice_clip(mixture, sr, host_start, host_end, "host")
        guest_enroll = slice_clip(mixture, sr, guest_start, guest_end, "guest")

        print(f"Loading pretrained model: {model_tag}")
        print("(first run downloads the checkpoint, this can take a minute)")
        from espnet2.bin.enh_tse_inference import SeparateSpeech

        separate_speech = SeparateSpeech.from_pretrained(
            model_tag=model_tag,
            device=device,
        )

        host_batch = torch.as_tensor(host_enroll).unsqueeze(0)
        guest_batch = torch.as_tensor(guest_enroll).unsqueeze(0)

        def process_chunk(chunk: np.ndarray):
            chunk_batch = torch.as_tensor(chunk).unsqueeze(0)
            joint_result = run_joint_call(separate_speech, chunk_batch, host_batch, guest_batch, sr)
            if joint_result is not None:
                return joint_result
            return run_sequential_calls(separate_speech, chunk_batch, host_batch, guest_batch, sr)

        print("Running extraction (chunked into ~30s windows for long inputs;")
        print("each chunk tries the joint call first, falling back to sequential calls)...")
        host_out, guest_out = chunked_extract(process_chunk, mixture, sr)

    return host_out, guest_out, sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="YouTube URL or local file path")
    parser.add_argument("--host-start", required=True, help="Host reference clip start (MM:SS or seconds)")
    parser.add_argument("--host-end", required=True, help="Host reference clip end (MM:SS or seconds)")
    parser.add_argument("--guest-start", required=True, help="Guest reference clip start (MM:SS or seconds)")
    parser.add_argument("--guest-end", required=True, help="Guest reference clip end (MM:SS or seconds)")
    parser.add_argument("--host-output", default="test_output_host.wav", help="Where to write the host's extracted audio")
    parser.add_argument("--guest-output", default="test_output_guest.wav", help="Where to write the guest's extracted audio")
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="ESPnet model_zoo tag for the pretrained TD-SpeakerBeam checkpoint",
    )
    args = parser.parse_args()

    host_out, guest_out, sr = run_dual_extraction(
        args.input, args.host_start, args.host_end, args.guest_start, args.guest_end,
        model_tag=args.model_tag,
    )

    sf.write(args.host_output, host_out, sr)
    sf.write(args.guest_output, guest_out, sr)
    print(f"Wrote {args.host_output} ({len(host_out) / sr:.1f}s at {sr}Hz)")
    print(f"Wrote {args.guest_output} ({len(guest_out) / sr:.1f}s at {sr}Hz)")

    print()
    print(f"Done. Please listen to {args.host_output} and {args.guest_output} and")
    print("confirm each contains only its intended speaker (Phase 2 definition of done).")


if __name__ == "__main__":
    main()
