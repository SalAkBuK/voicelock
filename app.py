#!/usr/bin/env python
"""
VoiceLock Phase 5: minimal Streamlit UI.

Wraps the proven CLI pipeline (test_dual_extraction.py + test_mix_remux.py)
in a single-page app: file drop, host profile dropdown, guest timestamp
picker, ambience slider, process button.

Run with: streamlit run app.py
"""

import tempfile
from pathlib import Path

import soundfile as sf
import streamlit as st

from test_dual_extraction import run_dual_extraction
from test_mix_remux import run_mix_remux

HOST_PROFILES_DIR = Path("host_profiles")
HOST_PROFILES_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="VoiceLock", page_icon="\U0001F3A4")
st.title("VoiceLock")
st.caption(
    "Isolate host and guest voices from a noisy recording, blend in a touch "
    "of ambience, and remux back into the original video -- no video re-encode."
)

uploaded_file = st.file_uploader(
    "Source file", type=["mp4", "mov", "mkv", "wav"],
    help="The recording containing both speakers.",
)

if uploaded_file is not None:
    tmp_dir = Path(tempfile.gettempdir()) / "voicelock_ui"
    tmp_dir.mkdir(exist_ok=True)
    source_path = tmp_dir / uploaded_file.name
    source_path.write_bytes(uploaded_file.getvalue())

    if source_path.suffix.lower() == ".wav":
        st.audio(str(source_path))
    else:
        st.video(str(source_path))

    st.subheader("Host")
    existing_profiles = sorted(p.stem for p in HOST_PROFILES_DIR.glob("*.wav"))
    profile_choice = st.selectbox(
        "Host profile", options=["Enroll new host..."] + existing_profiles,
    )

    host_start = host_end = new_profile_name = None
    if profile_choice == "Enroll new host...":
        col1, col2 = st.columns(2)
        with col1:
            host_start = st.text_input("Host clip start (MM:SS)", value="0:00", key="host_start")
        with col2:
            host_end = st.text_input("Host clip end (MM:SS)", value="0:05", key="host_end")
        new_profile_name = st.text_input("Save this host as (name)", value="host1")
    else:
        st.caption(f"Using saved profile: host_profiles/{profile_choice}.wav")

    st.subheader("Guest")
    col3, col4 = st.columns(2)
    with col3:
        guest_start = st.text_input("Guest clip start (MM:SS)", value="0:10", key="guest_start")
    with col4:
        guest_end = st.text_input("Guest clip end (MM:SS)", value="0:15", key="guest_end")

    st.subheader("Ambience")
    ambience_pct = st.slider(
        "Raw ambience blended back in (%)", min_value=0, max_value=20, value=8,
        help="Higher values sound less like a 'dead room' but bring back more background noise.",
    )

    if st.button("Process", type="primary"):
        with st.spinner("Extracting host and guest voices (this can take a while for long files)..."):
            if profile_choice == "Enroll new host...":
                # Extract and save the raw enrollment clip for reuse next time,
                # per CLAUDE.md's decision to persist the clip itself (not a
                # literal embedding vector -- no documented API for that).
                import soundfile as _sf
                from test_extraction import normalize_to_wav, parse_timestamp, MODEL_SAMPLE_RATE

                normalized_tmp = tmp_dir / "normalized_for_enroll.wav"
                normalize_to_wav(source_path, normalized_tmp, MODEL_SAMPLE_RATE)
                full_audio, sr = _sf.read(str(normalized_tmp), dtype="float32")
                start_sample = int(parse_timestamp(host_start) * sr)
                end_sample = int(parse_timestamp(host_end) * sr)
                host_clip = full_audio[start_sample:end_sample]
                profile_path = HOST_PROFILES_DIR / f"{new_profile_name}.wav"
                _sf.write(str(profile_path), host_clip, sr)
                st.info(f"Saved host profile: {profile_path}")

                # Host clip's timestamps are already valid within this source
                # file, so extraction can run directly against it.
                host_audio, guest_audio, track_sr = run_dual_extraction(
                    source_path, host_start, host_end, guest_start, guest_end,
                )
            else:
                # Enroll "from" the saved profile clip by temporarily prepending
                # it to the source so the existing timestamp-based pipeline can
                # use it as the host reference, without changing that pipeline.
                profile_path = HOST_PROFILES_DIR / f"{profile_choice}.wav"
                profile_audio, profile_sr = sf.read(str(profile_path), dtype="float32")

                from test_extraction import normalize_to_wav, MODEL_SAMPLE_RATE
                import numpy as np

                normalized_tmp = tmp_dir / "normalized_for_run.wav"
                normalize_to_wav(source_path, normalized_tmp, MODEL_SAMPLE_RATE)
                source_audio, sr = sf.read(str(normalized_tmp), dtype="float32")

                combined = np.concatenate([profile_audio, source_audio]).astype("float32")
                combined_path = tmp_dir / "combined_with_profile.wav"
                sf.write(str(combined_path), combined, sr)

                profile_duration = len(profile_audio) / sr
                host_start_arg = "0"
                host_end_arg = str(profile_duration)

                from test_extraction import parse_timestamp
                guest_start_shifted = str(parse_timestamp(guest_start) + profile_duration)
                guest_end_shifted = str(parse_timestamp(guest_end) + profile_duration)

                host_audio, guest_audio, track_sr = run_dual_extraction(
                    combined_path, host_start_arg, host_end_arg,
                    guest_start_shifted, guest_end_shifted,
                )

        with st.spinner("Mixing and remuxing..."):
            output_path, loudnorm_stats = run_mix_remux(
                source_path, host_audio, guest_audio, track_sr,
                ambience=ambience_pct / 100.0,
                output_path=tmp_dir / f"mixed_{uploaded_file.name}",
            )

        st.success("Done!")
        if Path(output_path).suffix.lower() in (".mp4", ".mov", ".mkv"):
            st.video(str(output_path))
        else:
            st.audio(str(output_path))

        with open(output_path, "rb") as f:
            st.download_button("Download result", f, file_name=Path(output_path).name)
