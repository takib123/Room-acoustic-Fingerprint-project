"""
fingerprint.py
--------------
Combine everything upstream (reflections, frequency response,
reverberation properties) into a single, compact "acoustic
fingerprint" for a room: a dict of named features plus a fixed-length
numeric vector suitable for distance/similarity comparisons.

Concepts used: Acoustic feature extraction, Room Acoustic Fingerprint
(this is the terminal node of the pipeline before comparison).
"""

import json
import numpy as np

from . import spectral
from . import energy_decay
from . import rt60 as rt60_mod
from . import peak_detection


OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


def extract_features(h, fs):
    """
    Run the full feature-extraction stage on an impulse response h[n]
    and return a JSON-serializable dict of acoustic features.
    """
    # --- Frequency response (FFT) ---
    freqs, mag_db, phase = spectral.compute_fft(h, fs)
    centroid = spectral.spectral_centroid(freqs, mag_db)
    flatness = spectral.spectral_flatness(mag_db)
    band_energy = spectral.octave_band_energy(h, fs, OCTAVE_BANDS)

    # --- Energy decay / RT60 ---
    edc_db = energy_decay.compute_edc_db(h)
    rt60_all = rt60_mod.estimate_rt60_all_methods(edc_db, fs)

    # Compact, normalized Schroeder-decay signature.  Unlike absolute
    # waveform amplitude, this describes how the room's energy dies away
    # over time and is therefore useful for repeated recordings of the
    # same room.  Keep it small so fingerprint JSON files remain compact.
    signature_points = 128
    if len(edc_db) > 1:
        old_axis = np.linspace(0.0, 1.0, len(edc_db))
        new_axis = np.linspace(0.0, 1.0, signature_points)
        decay_signature = np.interp(
            new_axis,
            old_axis,
            np.clip(np.asarray(edc_db, dtype=np.float64), -60.0, 0.0),
        ).tolist()
    else:
        decay_signature = [0.0] * signature_points
    c50 = rt60_mod.clarity_index(h, fs, t_ms=50.0)
    c80 = rt60_mod.clarity_index(h, fs, t_ms=80.0)
    drr = rt60_mod.direct_to_reverberant_ratio(h, fs)

    # --- Per-octave-band RT60 (frequency-resolved reverberation) ---
    from scipy import signal as sig
    nyq = fs / 2.0
    rt60_per_band = {}
    for center in OCTAVE_BANDS:
        low = center / np.sqrt(2)
        high = min(center * np.sqrt(2), nyq * 0.999)
        if low <= 0 or low >= high:
            continue
        b, a = sig.butter(4, [low / nyq, high / nyq], btype="band")
        h_band = sig.filtfilt(b, a, h)
        edc_band = energy_decay.compute_edc_db(h_band)
        rt60_per_band[center] = rt60_mod.estimate_rt60(edc_band, fs, method="T30")

    # --- Reflections ---
    peaks = peak_detection.detect_peaks(h, fs)
    reflection_delays_ms = [p["delay_from_direct_ms"] for p in peaks[1:]]  # skip direct sound
    reflection_levels_db = [p["level_db"] for p in peaks[1:]]
    density = peak_detection.reflection_density(peaks)

    features = {
        "rt60_edt_s": rt60_all["EDT"],
        "rt60_t20_s": rt60_all["T20"],
        "rt60_t30_s": rt60_all["T30"],
        "rt60_per_octave_band_s": rt60_per_band,
        "clarity_c50_db": c50,
        "clarity_c80_db": c80,
        "direct_to_reverberant_ratio_db": drr,
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "octave_band_energy": band_energy,
        "num_reflections_detected": max(0, len(peaks) - 1),
        "reflection_density_per_s": density,
        "first_reflection_delay_ms": reflection_delays_ms[0] if reflection_delays_ms else None,
        "reflection_delays_ms": reflection_delays_ms,
        "reflection_levels_db": reflection_levels_db,
        "decay_signature_db": decay_signature,
    }
    return features


def _safe(value, default=0.0):
    """Replace None/inf/nan with a bounded default for vectorization."""
    if value is None:
        return default
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return default
    return value


def to_vector(features, max_reflections=10):
    """
    Flatten a feature dict into a fixed-length numeric vector so two
    fingerprints can be compared with distance/similarity metrics.

    Layout:
      [rt60_edt, rt60_t20, rt60_t30,
       rt60_band_125..8000 (7),
       c50, c80, drr,
       centroid, flatness,
       band_energy_125..8000 (7, log-scaled & normalized),
       num_reflections, reflection_density,
       reflection_delays (padded/truncated to max_reflections),
       reflection_levels (padded/truncated to max_reflections)]
    """
    vec = []
    vec += [_safe(features["rt60_edt_s"]), _safe(features["rt60_t20_s"]), _safe(features["rt60_t30_s"])]
    vec += [_safe(features["rt60_per_octave_band_s"].get(b)) for b in OCTAVE_BANDS]
    vec += [_safe(features["clarity_c50_db"], 60.0), _safe(features["clarity_c80_db"], 60.0),
            _safe(features["direct_to_reverberant_ratio_db"], 60.0)]
    vec += [_safe(features["spectral_centroid_hz"]) / 1000.0, _safe(features["spectral_flatness"])]

    band_energies = list(features["octave_band_energy"].values())
    if band_energies:
        log_energies = np.log10(np.array(band_energies) + 1e-12)
        log_energies -= np.max(log_energies)  # normalize relative to strongest band
    else:
        log_energies = np.zeros(len(OCTAVE_BANDS))
    vec += list(log_energies)

    vec += [features["num_reflections_detected"], features["reflection_density_per_s"]]

    delays = (features["reflection_delays_ms"] or [])[:max_reflections]
    delays = delays + [0.0] * (max_reflections - len(delays))
    vec += delays

    levels = (features["reflection_levels_db"] or [])[:max_reflections]
    levels = levels + [-100.0] * (max_reflections - len(levels))
    vec += levels

    return np.array(vec, dtype=np.float64)


def build_fingerprint(h, fs, room_name="room"):
    """Build the full fingerprint object: metadata + features + vector."""
    features = extract_features(h, fs)
    vector = to_vector(features)
    return {
        "room_name": room_name,
        "sample_rate": fs,
        "features": features,
        "vector": vector.tolist(),
    }


def save_fingerprint(fingerprint, path):
    with open(path, "w") as f:
        json.dump(fingerprint, f, indent=2)


def load_fingerprint(path):
    with open(path, "r") as f:
        return json.load(f)
