#!/usr/bin/env python3
"""
app.py
------
A single-page web UI for the whole Room Acoustic Fingerprint & Analysis
System, built with Streamlit.

Run it with:
    streamlit run app.py

Everything the project does is reachable from this one screen, across
three tabs:
  1. Analyze a Room   -- upload a .wav, record from your mic, or use a
                          synthetic demo room; runs the FULL pipeline
                          (sampling -> filtering -> correlation -> RIR ->
                          FFT -> STFT -> energy decay -> RT60 ->
                          fingerprint) and shows every plot.
  2. Room Library      -- every room you've analyzed (this session, or
                          previously saved to disk), with an at-a-glance
                          comparison table.
  3. Compare Rooms      -- pick any two analyzed rooms and see their
                          similarity score, key differences, and full
                          fingerprint JSON.
"""

import os
import json

import numpy as np
import pandas as pd
import streamlit as st

from src import io_utils
from src import pipeline as pipeline_mod
from src import fingerprint as fp_mod
from src import compare as compare_mod
from src import synthetic

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Room Acoustic Fingerprint", page_icon="🔊", layout="wide")


# ----------------------------------------------------------------------
# Session state: a small in-memory "room library"
# ----------------------------------------------------------------------
def _slug(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "room"


if "rooms" not in st.session_state:
    st.session_state.rooms = {}  # room_name -> {"fingerprint": dict, "results": dict|None}

    # Auto-load any fingerprints already saved on disk (e.g. from a
    # previous session, or from running demo.py / main.py) so the
    # library isn't empty on first launch.
    if os.path.isdir(OUTPUT_DIR):
        for sub in sorted(os.listdir(OUTPUT_DIR)):
            fp_path = os.path.join(OUTPUT_DIR, sub, "fingerprint.json")
            if os.path.isfile(fp_path):
                try:
                    fp = fp_mod.load_fingerprint(fp_path)
                    st.session_state.rooms[fp["room_name"]] = {"fingerprint": fp, "results": None}
                except Exception:
                    pass


def save_room(room_name, results):
    out_dir = os.path.join(OUTPUT_DIR, _slug(room_name))
    os.makedirs(out_dir, exist_ok=True)
    fp_mod.save_fingerprint(results["fingerprint"], os.path.join(out_dir, "fingerprint.json"))
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(analyzer_report_text(results, room_name))
    st.session_state.rooms[room_name] = {"fingerprint": results["fingerprint"], "results": results}


def analyzer_report_text(results, room_name):
    a = pipeline_mod.RoomAcousticAnalyzer(room_name=room_name)
    a.results = results
    return a.report_text()


def feature_row(room_name, features):
    return {
        "Room": room_name,
        "RT60 T30 (s)": features["rt60_t30_s"],
        "RT60 T20 (s)": features["rt60_t20_s"],
        "EDT (s)": features["rt60_edt_s"],
        "C50 (dB)": features["clarity_c50_db"],
        "C80 (dB)": features["clarity_c80_db"],
        "DRR (dB)": features["direct_to_reverberant_ratio_db"],
        "Centroid (Hz)": features["spectral_centroid_hz"],
        "Reflections": features["num_reflections_detected"],
    }


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🔊 Room Acoustic Fingerprint & Analysis System")
st.caption(
    "Sampling → Noise Filtering → Correlation → Room Impulse Response → "
    "FFT / STFT → Energy Decay → RT60 → Acoustic Fingerprint → Room Comparison"
)

tab_analyze, tab_library, tab_compare = st.tabs(
    ["🎙️ Analyze a Room", "📚 Room Library", "⚖️ Compare Rooms"]
)

# ========================================================================
# TAB 1 — Analyze a Room
# ========================================================================
with tab_analyze:
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("1. Provide a recording")
        room_name = st.text_input("Room name", value="My Room", key="room_name_input")

        source = st.radio(
            "Audio source",
            ["Upload a .wav file", "Record with microphone", "Use a synthetic demo room"],
        )

        raw_signal, fs = None, None

        if source == "Upload a .wav file":
            uploaded = st.file_uploader("Clap recording (.wav)", type=["wav"])
            if uploaded is not None:
                tmp_path = os.path.join("/tmp", f"upload_{uploaded.name}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                raw_signal, fs = io_utils.load_wav(tmp_path)
                st.audio(uploaded)

        elif source == "Record with microphone":
            st.caption("Clap once sharply, leave ~1-2s of quiet tail, then stop.")
            try:
                audio_value = st.audio_input("Record a clap")
            except AttributeError:
                audio_value = None
                st.error(
                    "Your installed Streamlit version doesn't support in-browser "
                    "mic recording (`st.audio_input`). Run `pip install -U streamlit` "
                    "(needs >= 1.34), or use the 'Upload a .wav file' option instead."
                )
            if audio_value is not None:
                tmp_path = "/tmp/mic_recording.wav"
                with open(tmp_path, "wb") as f:
                    f.write(audio_value.getbuffer())
                raw_signal, fs = io_utils.load_wav(tmp_path)

        else:  # synthetic demo room
            demo_choice = st.selectbox(
                "Synthetic room type", ["Small Dry Room", "Medium Office", "Large Reverberant Hall"]
            )
            seed = st.number_input("Random seed", value=1, step=1)
            rt60_map = {"Small Dry Room": 0.3, "Medium Office": 0.5, "Large Reverberant Hall": 1.8}
            n_refl_map = {"Small Dry Room": 8, "Medium Office": 12, "Large Reverberant Hall": 20}
            if st.button("🎲 Generate synthetic recording"):
                target_rt60 = rt60_map[demo_choice]
                rir = synthetic.generate_synthetic_rir(
                    rt60=target_rt60, duration=max(1.5, target_rt60 * 2),
                    n_early_reflections=n_refl_map[demo_choice], seed=int(seed),
                )
                sig = synthetic.simulate_recording(rir, 44100, seed=int(seed))
                st.session_state["_synthetic_signal"] = (sig, 44100)
            if "_synthetic_signal" in st.session_state:
                raw_signal, fs = st.session_state["_synthetic_signal"]
                st.success(f"Synthetic '{demo_choice}' recording ready ({len(raw_signal)/fs:.2f}s)")

        st.subheader("2. Run the pipeline")
        run_clicked = st.button(
            "▶ Run Full Analysis", type="primary", disabled=raw_signal is None, use_container_width=True
        )

    with right:
        st.subheader("Pipeline stages")
        st.markdown(
            """
1. **Sampling & pre-processing** — resample, remove DC offset, normalize
2. **Noise filtering** — band-pass filter + spectral noise gate
3. **Correlation analysis** — autocorrelation (echo/periodicity structure)
4. **Room impulse response h[n]** — trimmed from the clap / deconvolved
5. **Echo & reflection detection** — peak-picking on the envelope
6. **FFT → frequency response H(f)**
7. **STFT → spectrogram**
8. **Energy decay analysis** — Schroeder backward integration
9. **RT60 estimation** — EDT / T20 / T30, plus C50, C80, DRR
10. **Acoustic fingerprint** — combined feature vector, saved for comparison
            """
        )

    if run_clicked and raw_signal is not None:
        with st.spinner("Running the full DSP pipeline..."):
            analyzer = pipeline_mod.RoomAcousticAnalyzer(room_name=room_name)
            analyzer.run(raw_signal, fs)
            save_room(room_name, analyzer.results)
        st.success(f"✅ Analysis complete for **{room_name}** — saved to the Room Library.")

    # Show results for the currently-named room, if analyzed
    room_entry = st.session_state.rooms.get(room_name)
    if room_entry and room_entry.get("results"):
        results = room_entry["results"]
        f = results["fingerprint"]["features"]

        st.divider()
        st.subheader(f"Results — {room_name}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("RT60 (T30)", f"{f['rt60_t30_s']:.2f} s" if f["rt60_t30_s"] else "n/a")
        m2.metric("Clarity C50", f"{f['clarity_c50_db']:.1f} dB")
        m3.metric("DRR", f"{f['direct_to_reverberant_ratio_db']:.1f} dB")
        m4.metric("Spectral Centroid", f"{f['spectral_centroid_hz']:.0f} Hz")
        m5.metric("Reflections", f["num_reflections_detected"])

        analyzer_for_plots = pipeline_mod.RoomAcousticAnalyzer(room_name=room_name)
        analyzer_for_plots.results = results
        figs = analyzer_for_plots.generate_figures()

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(figs["impulse_response"], use_container_width=True)
            st.pyplot(figs["spectrogram"], use_container_width=True)
            st.pyplot(figs["autocorrelation"], use_container_width=True)
        with c2:
            st.pyplot(figs["frequency_response"], use_container_width=True)
            st.pyplot(figs["energy_decay"], use_container_width=True)
            st.pyplot(figs["rt60_bands"], use_container_width=True)

        with st.expander("Full fingerprint (JSON)"):
            st.json(results["fingerprint"]["features"])


# ========================================================================
# TAB 2 — Room Library
# ========================================================================
with tab_library:
    st.subheader("All analyzed rooms")
    if not st.session_state.rooms:
        st.info("No rooms analyzed yet — go to the **Analyze a Room** tab to add one.")
    else:
        rows = [
            feature_row(name, entry["fingerprint"]["features"])
            for name, entry in st.session_state.rooms.items()
        ]
        df = pd.DataFrame(rows).set_index("Room")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇ Download library as CSV",
            df.to_csv().encode("utf-8"),
            file_name="room_library.csv",
            mime="text/csv",
        )

        st.divider()
        del_room = st.selectbox("Remove a room from the library", ["—"] + list(st.session_state.rooms.keys()))
        if del_room != "—" and st.button("🗑 Remove selected room"):
            del st.session_state.rooms[del_room]
            st.rerun()


# ========================================================================
# TAB 3 — Compare Rooms
# ========================================================================
with tab_compare:
    st.subheader("Room-to-room comparison")
    room_names = list(st.session_state.rooms.keys())

    if len(room_names) < 2:
        st.info("Analyze at least two rooms to compare them.")
    else:
        c1, c2 = st.columns(2)
        room_a = c1.selectbox("Room A", room_names, index=0)
        room_b = c2.selectbox("Room B", room_names, index=1 if len(room_names) > 1 else 0)

        if room_a == room_b:
            st.warning("Pick two different rooms to compare.")
        elif st.button("⚖️ Compare", type="primary"):
            fp_a = st.session_state.rooms[room_a]["fingerprint"]
            fp_b = st.session_state.rooms[room_b]["fingerprint"]
            result = compare_mod.compare_fingerprints(fp_a, fp_b)

            st.metric("Overall similarity", f"{result['overall_similarity_percent']:.1f}%")

            m1, m2 = st.columns(2)
            m1.metric("Cosine similarity", f"{result['cosine_similarity']:.3f}")
            m2.metric("Normalized Euclidean distance", f"{result['normalized_euclidean_distance']:.2f}")

            st.markdown("#### Key feature differences")
            diffs = result["key_feature_differences"]
            st.dataframe(pd.DataFrame([diffs]).T.rename(columns={0: "Room A − Room B"}), use_container_width=True)

            with st.expander("Full comparison JSON"):
                st.json(result)
