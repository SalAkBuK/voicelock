#!/usr/bin/env python
"""
VoiceLock Phase 1 proof of concept.

Given a mixture (YouTube URL or local file) and a short clean reference
clip of the target speaker, extract that speaker's voice using ESPnet2's
pretrained TD-SpeakerBeam target speaker extraction model.

Usage:
    python test_extraction.py <input> --ref-start MM:SS --ref-end MM:SS [--output test_output.wav]

<input> can be a YouTube URL or a local file path (.mp4/.mov/.mkv/.wav).
--ref-start/--ref-end mark a clean segment of the target speaker within
that same input, used to enroll them.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import ffmpeg
import numpy as np
import soundfile as sf
import torch

# Both candidate TD-SpeakerBeam checkpoints (LibriMix and WSJ0-2mix) were
# trained at 8kHz, not 16kHz -- confirmed by inspecting each model's
# downloaded config.yaml ("sample_rate: 8000"). Audio must be normalized to
# this rate to match what the model was trained on.
DEFAULT_MODEL_TAG = "espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw"
MODEL_SAMPLE_RATE = 8000


def is_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def parse_timestamp(value: str) -> float:
    """Accept 'MM:SS', 'HH:MM:SS', or plain seconds like '12.5'."""
    if ":" in value:
        parts = [float(p) for p in value.split(":")]
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    return float(value)


def download_audio(url: str, out_dir: Path) -> Path:
    import yt_dlp

    print(f"Downloading audio from: {url}")
    out_template = str(out_dir / "downloaded.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded = list(out_dir.glob("downloaded.*"))
    if not downloaded:
        raise RuntimeError("yt-dlp finished but no output file was found")
    print(f"  -> saved as {downloaded[0].name}")
    return downloaded[0]


def chunked_extract(process_fn, mixture: np.ndarray, sr: int, window_sec: float = 30.0, crossfade_sec: float = 2.0):
    """
    Split `mixture` into overlapping windows, run `process_fn` on each, and
    reassemble with a linear cross-fade over the overlap so long inputs can
    be processed in bounded-memory chunks instead of one giant pass.

    process_fn(chunk: np.ndarray) -> sequence of np.ndarray, one or more
    output tracks the same length as `chunk`.

    Returns a list of np.ndarray, one per output track, each the same
    length as `mixture`. If `mixture` fits in one window, chunking is
    skipped entirely and process_fn is called once on the whole thing.
    """
    window_len = int(window_sec * sr)
    crossfade_len = int(crossfade_sec * sr)
    hop_len = window_len - crossfade_len
    total_len = len(mixture)

    if total_len <= window_len:
        return list(process_fn(mixture))

    starts = sorted(set(list(range(0, total_len - window_len, hop_len)) + [total_len - window_len]))

    outputs = None
    for i, start in enumerate(starts):
        end = start + window_len
        chunk_outputs = list(process_fn(mixture[start:end]))
        if outputs is None:
            outputs = [np.zeros(total_len, dtype=np.float32) for _ in chunk_outputs]

        if i == 0:
            for t in range(len(outputs)):
                outputs[t][start:end] = chunk_outputs[t]
        else:
            prev_end = starts[i - 1] + window_len
            raw_overlap_len = prev_end - start

            # The geometric overlap between windows can be much larger than
            # crossfade_len -- e.g. when the final window is shifted to align
            # exactly with the end of the file, it can end up mostly
            # overlapping its predecessor. Blending two independently
            # produced model outputs over an oversized span causes severe
            # phase-cancellation/comb-filtering artifacts, not a smooth
            # transition. So the actual cross-faded region is always capped
            # at crossfade_len, placed right at the true boundary
            # (prev_end); any extra overlap before that is redundant and
            # resolved by keeping the previous window's output untouched
            # (already written, more central within that window and further
            # from its edges than the new window's competing estimate would
            # be at its own start).
            fade_len = min(raw_overlap_len, crossfade_len)
            fade_start = prev_end - fade_len
            new_part_offset = fade_start - start

            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = 1.0 - fade_in
            for t in range(len(outputs)):
                existing = outputs[t][fade_start:prev_end]
                new_part = chunk_outputs[t][new_part_offset:new_part_offset + fade_len]
                outputs[t][fade_start:prev_end] = existing * fade_out + new_part * fade_in
                outputs[t][prev_end:end] = chunk_outputs[t][prev_end - start:]

    return outputs


def normalize_to_wav(input_path: Path, output_path: Path, sample_rate: int) -> None:
    """Convert any input media to mono PCM WAV at the given sample rate."""
    print(f"Normalizing audio to {sample_rate}Hz mono WAV...")
    try:
        (
            ffmpeg.input(str(input_path))
            .output(str(output_path), ac=1, ar=sample_rate, vn=None)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        print(stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg failed to normalize the input audio") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="YouTube URL or local file path")
    parser.add_argument(
        "--ref-start", required=True, help="Reference clip start (MM:SS or seconds)"
    )
    parser.add_argument(
        "--ref-end", required=True, help="Reference clip end (MM:SS or seconds)"
    )
    parser.add_argument(
        "--output", default="test_output.wav", help="Where to write the extracted audio"
    )
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="ESPnet model_zoo tag for the pretrained TD-SpeakerBeam checkpoint",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected -- running on CPU, this will be noticeably slower.")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        if is_url(args.input):
            source_path = download_audio(args.input, tmp)
        else:
            source_path = Path(args.input)
            if not source_path.exists():
                raise FileNotFoundError(f"No such file: {source_path}")

        normalized_path = tmp / "normalized.wav"
        normalize_to_wav(source_path, normalized_path, MODEL_SAMPLE_RATE)

        mixture, sr = sf.read(normalized_path, dtype="float32")
        assert sr == MODEL_SAMPLE_RATE, f"expected {MODEL_SAMPLE_RATE}Hz, got {sr}Hz"

        ref_start_sec = parse_timestamp(args.ref_start)
        ref_end_sec = parse_timestamp(args.ref_end)
        if ref_end_sec <= ref_start_sec:
            raise ValueError("--ref-end must be after --ref-start")

        start_sample = int(ref_start_sec * sr)
        end_sample = int(ref_end_sec * sr)
        if end_sample > len(mixture):
            raise ValueError(
                f"Reference end ({ref_end_sec:.1f}s) is beyond the audio length "
                f"({len(mixture) / sr:.1f}s)"
            )
        enrollment = mixture[start_sample:end_sample]
        print(
            f"Reference clip: {ref_end_sec - ref_start_sec:.1f}s "
            f"({args.ref_start} to {args.ref_end})"
        )

        print(f"Loading pretrained model: {args.model_tag}")
        print("(first run downloads the checkpoint, this can take a minute)")
        from espnet2.bin.enh_tse_inference import SeparateSpeech

        separate_speech = SeparateSpeech.from_pretrained(
            model_tag=args.model_tag,
            device=device,
        )

        enrollment_batch = torch.as_tensor(enrollment).unsqueeze(0)

        def process_chunk(chunk: np.ndarray):
            chunk_batch = torch.as_tensor(chunk).unsqueeze(0)
            results = separate_speech(chunk_batch, fs=sr, enroll_ref1=enrollment_batch)
            out = results[0]
            if isinstance(out, torch.Tensor):
                out = out.squeeze(0).cpu().numpy()
            else:
                out = np.squeeze(out, axis=0)
            return [out]

        print("Running extraction (chunked into ~30s windows for long inputs)...")
        [extracted] = chunked_extract(process_chunk, mixture, sr)

        sf.write(args.output, extracted, sr)
        print(f"Wrote {args.output} ({len(extracted) / sr:.1f}s at {sr}Hz)")

    print()
    print(f"Done. Please listen to {args.output} and confirm by ear that the")
    print("target speaker's voice is intelligible (Phase 1 definition of done).")


if __name__ == "__main__":
    main()
