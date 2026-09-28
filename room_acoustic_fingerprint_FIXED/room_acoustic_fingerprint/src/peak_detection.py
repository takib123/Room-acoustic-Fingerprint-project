"""
peak_detection.py
------------------
Detect the direct sound and discrete early reflections in h[n].

Concepts used: Peak/reflection detection, correlation (the envelope
used here is closely related to the analytic-signal/Hilbert-transform
view of correlation-based energy tracking).
"""

import numpy as np
from scipy import signal as sig


def envelope(h):
    """Amplitude envelope of h[n] via the analytic signal (Hilbert transform)."""
    return np.abs(sig.hilbert(h))


def detect_peaks(h, fs, min_prominence_db=6.0, max_peaks=20):
    """
    Detect the direct sound and subsequent reflections as prominent
    peaks in the envelope of h[n].

    Returns a list of dicts: {sample, time_s, amplitude, level_db,
    delay_from_direct_ms}, sorted by time, with the first entry being
    the direct sound (t = 0 reference).
    """
    env = envelope(h)
    peak = np.max(env)
    if peak < 1e-12:
        return []

    env_db = 20 * np.log10(env / peak + 1e-12)

    # Prominence in dB translated to a linear threshold works better
    # numerically, so we search on the linear envelope but filter by dB.
    peaks_idx, props = sig.find_peaks(
        env, distance=int(0.001 * fs), prominence=peak * (10 ** (-min_prominence_db / 20.0))
    )

    if len(peaks_idx) == 0:
        return []

    # Direct sound = first strong peak in time (assume it's near the global max)
    direct_idx = peaks_idx[np.argmax(env[peaks_idx])] if peaks_idx[0] != np.argmax(env) else peaks_idx[0]
    direct_idx = int(np.argmax(env))  # robust: true global peak is the direct sound

    results = []
    for idx in peaks_idx:
        results.append({
            "sample": int(idx),
            "time_s": idx / fs,
            "amplitude": float(env[idx]),
            "level_db": float(env_db[idx]),
            "delay_from_direct_ms": (idx - direct_idx) / fs * 1000.0,
        })

    results.sort(key=lambda d: d["time_s"])

    # Keep only the strongest `max_peaks` reflections (by level) to
    # keep the fingerprint compact, but always keep the direct sound.
    if len(results) > max_peaks:
        direct = next(r for r in results if r["sample"] == direct_idx)
        others = sorted([r for r in results if r["sample"] != direct_idx],
                         key=lambda d: -d["level_db"])[: max_peaks - 1]
        results = sorted([direct] + others, key=lambda d: d["time_s"])

    return results


def reflection_density(peaks, window_ms=100.0):
    """
    Number of detected reflections per unit time within the first
    `window_ms` after the direct sound -- a simple measure of how
    'busy' early reflections are (relates to room size/diffusivity).
    """
    if not peaks:
        return 0.0
    count = sum(1 for p in peaks if 0 <= p["delay_from_direct_ms"] <= window_ms)
    return count / (window_ms / 1000.0)  # reflections per second
