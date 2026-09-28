"""
pipeline.py
-----------
Orchestrates the complete pipeline described in the project brief:

Record Audio/Clap -> Sampling & Pre-processing -> Noise Filtering ->
Correlation Analysis -> Echo & Reflection Detection ->
Room Impulse Response h[n] -> FFT/Frequency Response ->
STFT/Spectrogram -> Energy Decay Analysis -> RT60 Estimation ->
Acoustic Feature Extraction -> Room Acoustic Fingerprint ->
Room-to-Room Comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from . import preprocessing
from . import correlation
from . import rir as rir_mod
from . import peak_detection
from . import spectral
from . import stft as stft_mod
from . import energy_decay
from . import rt60 as rt60_mod
from . import fingerprint as fp_mod


class RoomAcousticAnalyzer:
    """
    High-level façade that runs every stage of the pipeline on a
    recorded clap (or any impulsive excitation) and stores every
    intermediate result for inspection, plotting, and reporting.
    """

    def __init__(self, room_name="room"):
        self.room_name = room_name
        self.results = {}

    def run(self, raw_signal, fs_in, target_fs=44100, is_clap=True, excitation=None):
        """
        Run the full pipeline.

        Parameters
        ----------
        raw_signal : the recorded audio (1D numpy array)
        fs_in : sample rate of raw_signal
        is_clap : True if raw_signal is a clap recording (RIR ~= trimmed
                  recording). False if `excitation` (the known reference
                  signal, e.g. a sine sweep) is supplied for deconvolution.
        """
        # 1. Sampling & Pre-processing / 2. Noise Filtering
        x, fs = preprocessing.preprocess_pipeline(raw_signal, fs_in, target_fs=target_fs)

        # 3. Correlation Analysis (self-similarity / echo lag structure)
        autocorr = correlation.normalized_autocorrelation(x)

        # 4/5. Room Impulse Response h[n]
        if is_clap:
            h, direct_idx = rir_mod.estimate_rir_from_clap(x, fs)
        else:
            if excitation is None:
                raise ValueError("excitation signal required when is_clap=False")
            h = rir_mod.estimate_rir_deconvolution(x, excitation)
            h, direct_idx = rir_mod.estimate_rir_from_clap(h, fs)  # trim to onset+tail
        h = rir_mod.trim_noise_floor(h, fs)

        # 6. Echo & Reflection Detection
        peaks = peak_detection.detect_peaks(h, fs)

        # 7. FFT / Frequency Response
        freqs, mag_db, phase = spectral.compute_fft(h, fs)

        # 8. STFT / Spectrogram
        f_stft, t_stft, spec_db = stft_mod.spectrogram_db(h, fs)

        # 9. Energy Decay Analysis
        edc_db = energy_decay.compute_edc_db(h)

        # 10. RT60 Estimation
        rt60_all = rt60_mod.estimate_rt60_all_methods(edc_db, fs)

        # 11. Acoustic Feature Extraction / 12. Fingerprint
        fingerprint = fp_mod.build_fingerprint(h, fs, room_name=self.room_name)

        self.results = {
            "fs": fs,
            "preprocessed_signal": x,
            "autocorrelation": autocorr,
            "rir": h,
            "direct_sound_index": direct_idx,
            "reflections": peaks,
            "freqs": freqs,
            "magnitude_db": mag_db,
            "phase": phase,
            "stft_freqs": f_stft,
            "stft_times": t_stft,
            "spectrogram_db": spec_db,
            "edc_db": edc_db,
            "rt60": rt60_all,
            "fingerprint": fingerprint,
        }
        return self.results

    # ------------------------------------------------------------------
    # Plotting / reporting
    # ------------------------------------------------------------------
    def plot_summary(self, out_dir, filename="analysis_summary.png"):
        """Save a multi-panel figure covering every analysis stage."""
        os.makedirs(out_dir, exist_ok=True)
        r = self.results
        fs = r["fs"]
        h = r["rir"]
        t_h = np.arange(len(h)) / fs

        fig, axes = plt.subplots(3, 2, figsize=(13, 12))

        # (1) Impulse response with reflections marked
        ax = axes[0, 0]
        ax.plot(t_h * 1000, h, linewidth=0.7, color="steelblue")
        for p in r["reflections"]:
            ax.axvline(p["time_s"] * 1000, color="red", alpha=0.3, linewidth=0.8)
        ax.set_title("Room Impulse Response h[n] (red = detected reflections)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")

        # (2) Frequency response
        ax = axes[0, 1]
        ax.semilogx(r["freqs"][1:], r["magnitude_db"][1:], color="darkorange")
        ax.set_title("Frequency Response H(f)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(20, fs / 2)
        ax.grid(True, which="both", alpha=0.3)

        # (3) Spectrogram (STFT)
        ax = axes[1, 0]
        pcm = ax.pcolormesh(r["stft_times"] * 1000, r["stft_freqs"], r["spectrogram_db"],
                             shading="gouraud", cmap="magma")
        ax.set_title("Spectrogram (STFT)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, min(8000, fs / 2))
        fig.colorbar(pcm, ax=ax, label="dB")

        # (4) Energy decay curve with RT60 fit region
        ax = axes[1, 1]
        t_edc = np.arange(len(r["edc_db"])) / fs
        ax.plot(t_edc * 1000, r["edc_db"], color="seagreen")
        ax.axhline(-5, color="gray", linestyle="--", linewidth=0.7)
        ax.axhline(-35, color="gray", linestyle="--", linewidth=0.7)
        rt30 = r["rt60"]["T30"]
        title = f"Energy Decay Curve (T30 RT60 = {rt30:.3f}s)" if rt30 else "Energy Decay Curve (T30: n/a)"
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Energy (dB)")
        ax.set_ylim(-80, 5)

        # (5) Autocorrelation (echo structure)
        ax = axes[2, 0]
        auto = r["autocorrelation"]
        mid = len(auto) // 2
        lags_ms = (np.arange(len(auto)) - mid) / fs * 1000
        window = int(0.05 * fs)  # +/-50ms around zero lag
        ax.plot(lags_ms[mid - window: mid + window], auto[mid - window: mid + window], color="purple")
        ax.set_title("Autocorrelation (echo/periodicity structure)")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Normalized correlation")

        # (6) RT60 per octave band
        ax = axes[2, 1]
        bands = fp_mod.OCTAVE_BANDS
        rt_vals = [r["fingerprint"]["features"]["rt60_per_octave_band_s"].get(b) or 0 for b in bands]
        ax.bar([str(b) for b in bands], rt_vals, color="teal")
        ax.set_title("RT60 per Octave Band (T30)")
        ax.set_xlabel("Octave band center (Hz)")
        ax.set_ylabel("RT60 (s)")

        fig.suptitle(f"Room Acoustic Analysis — {self.room_name}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def generate_figures(self):
        """
        Build the same six analysis plots as `plot_summary`, but as
        separate standalone matplotlib Figure objects (not saved to
        disk). Intended for interactive UIs (e.g. the Streamlit app)
        that want to lay figures out themselves and/or let the user
        zoom/pan each one individually.

        Returns a dict: {key: matplotlib.figure.Figure}
        """
        r = self.results
        fs = r["fs"]
        h = r["rir"]
        t_h = np.arange(len(h)) / fs
        figs = {}

        # Impulse response with reflections marked
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(t_h * 1000, h, linewidth=0.7, color="steelblue")
        for p in r["reflections"]:
            ax.axvline(p["time_s"] * 1000, color="red", alpha=0.3, linewidth=0.8)
        ax.set_title("Room Impulse Response h[n]  (red = detected reflections)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs["impulse_response"] = fig

        # Frequency response
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.semilogx(r["freqs"][1:], r["magnitude_db"][1:], color="darkorange")
        ax.set_title("Frequency Response H(f)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(20, fs / 2)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        figs["frequency_response"] = fig

        # Spectrogram
        fig, ax = plt.subplots(figsize=(6.5, 4))
        pcm = ax.pcolormesh(r["stft_times"] * 1000, r["stft_freqs"], r["spectrogram_db"],
                             shading="gouraud", cmap="magma")
        ax.set_title("Spectrogram (STFT)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, min(8000, fs / 2))
        fig.colorbar(pcm, ax=ax, label="dB")
        fig.tight_layout()
        figs["spectrogram"] = fig

        # Energy decay curve
        fig, ax = plt.subplots(figsize=(6.5, 4))
        t_edc = np.arange(len(r["edc_db"])) / fs
        ax.plot(t_edc * 1000, r["edc_db"], color="seagreen")
        ax.axhline(-5, color="gray", linestyle="--", linewidth=0.7)
        ax.axhline(-35, color="gray", linestyle="--", linewidth=0.7)
        rt30 = r["rt60"]["T30"]
        title = f"Energy Decay Curve (T30 RT60 = {rt30:.3f}s)" if rt30 else "Energy Decay Curve (T30: n/a)"
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Energy (dB)")
        ax.set_ylim(-80, 5)
        fig.tight_layout()
        figs["energy_decay"] = fig

        # Autocorrelation
        fig, ax = plt.subplots(figsize=(6.5, 4))
        auto = r["autocorrelation"]
        mid = len(auto) // 2
        lags_ms = (np.arange(len(auto)) - mid) / fs * 1000
        window = int(0.05 * fs)
        ax.plot(lags_ms[mid - window: mid + window], auto[mid - window: mid + window], color="purple")
        ax.set_title("Autocorrelation (echo/periodicity structure)")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Normalized correlation")
        fig.tight_layout()
        figs["autocorrelation"] = fig

        # RT60 per octave band
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bands = fp_mod.OCTAVE_BANDS
        rt_vals = [r["fingerprint"]["features"]["rt60_per_octave_band_s"].get(b) or 0 for b in bands]
        ax.bar([str(b) for b in bands], rt_vals, color="teal")
        ax.set_title("RT60 per Octave Band (T30)")
        ax.set_xlabel("Octave band center (Hz)")
        ax.set_ylabel("RT60 (s)")
        fig.tight_layout()
        figs["rt60_bands"] = fig

        return figs

    def report_text(self):
        """Human-readable text summary of the analysis."""
        f = self.results["fingerprint"]["features"]
        lines = [
            f"Room Acoustic Report — {self.room_name}",
            "=" * 50,
            f"RT60 (T30): {f['rt60_t30_s']:.3f} s" if f['rt60_t30_s'] else "RT60 (T30): n/a",
            f"RT60 (T20): {f['rt60_t20_s']:.3f} s" if f['rt60_t20_s'] else "RT60 (T20): n/a",
            f"EDT:        {f['rt60_edt_s']:.3f} s" if f['rt60_edt_s'] else "EDT: n/a",
            f"Clarity C50: {f['clarity_c50_db']:.2f} dB",
            f"Clarity C80: {f['clarity_c80_db']:.2f} dB",
            f"Direct-to-Reverberant Ratio: {f['direct_to_reverberant_ratio_db']:.2f} dB",
            f"Spectral Centroid: {f['spectral_centroid_hz']:.1f} Hz",
            f"Spectral Flatness: {f['spectral_flatness']:.3f}",
            f"Reflections detected: {f['num_reflections_detected']}",
            f"Reflection density: {f['reflection_density_per_s']:.1f} /s",
            f"First reflection delay: {f['first_reflection_delay_ms']:.2f} ms" if f['first_reflection_delay_ms'] else "First reflection delay: n/a",
        ]
        return "\n".join(lines)
