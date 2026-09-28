"""
rt60.py
-------
RT60 (reverberation time) estimation from the Schroeder energy decay
curve (EDC), following the standard ISO 3382 linear-regression
approach used in room acoustics (T20/T30/EDT).

Concepts used: RT60 estimation (built directly on top of
energy_decay.py).
"""

import numpy as np


def _linear_fit_rt(edc_db, fs, upper_db, lower_db):
    """
    Fit a line to the EDC (in dB) between `upper_db` and `lower_db`
    (e.g. -5 dB to -35 dB for T30), then extrapolate that line's slope
    to estimate the time needed to decay by 60 dB total = RT60.
    """
    t = np.arange(len(edc_db)) / fs

    mask = (edc_db <= upper_db) & (edc_db >= lower_db)
    if np.sum(mask) < 2:
        return None  # not enough data in this range (e.g. too short a recording)

    slope, intercept = np.polyfit(t[mask], edc_db[mask], 1)
    if slope >= 0:
        return None  # decay curve isn't decaying -- likely noisy/invalid

    rt60 = -60.0 / slope
    return float(rt60)


def estimate_rt60(edc_db, fs, method="T30"):
    """
    Estimate RT60 using one of the standard methods:
      - "EDT" (Early Decay Time): fit from 0 to -10 dB, extrapolated to -60 dB.
      - "T20": fit from -5 to -25 dB, extrapolated to -60 dB.
      - "T30": fit from -5 to -35 dB, extrapolated to -60 dB (most common).

    Returns RT60 in seconds, or None if it could not be reliably
    estimated (e.g. impulse response too short or too noisy).
    """
    ranges = {
        "EDT": (0.0, -10.0),
        "T20": (-5.0, -25.0),
        "T30": (-5.0, -35.0),
    }
    if method not in ranges:
        raise ValueError(f"Unknown RT60 method '{method}'. Use one of {list(ranges)}.")

    upper, lower = ranges[method]
    return _linear_fit_rt(edc_db, fs, upper, lower)


def estimate_rt60_all_methods(edc_db, fs):
    """Compute EDT, T20, and T30 in one call for convenience/reporting."""
    return {m: estimate_rt60(edc_db, fs, method=m) for m in ("EDT", "T20", "T30")}


def clarity_index(h, fs, t_ms=50.0):
    """
    Clarity index C50 (or C80 with t_ms=80): ratio, in dB, of early
    energy (0..t_ms) to late energy (t_ms..end). Higher C50 -> clearer
    speech intelligibility; lower/negative -> more reverberant/"muddy".
    """
    n_split = int(t_ms * fs / 1000.0)
    n_split = min(n_split, len(h) - 1)
    early_energy = np.sum(h[:n_split] ** 2)
    late_energy = np.sum(h[n_split:] ** 2)
    if late_energy < 1e-20:
        return float("inf")
    return float(10 * np.log10((early_energy + 1e-20) / (late_energy + 1e-20)))


def direct_to_reverberant_ratio(h, fs, direct_window_ms=2.5):
    """
    DRR: ratio, in dB, of the energy in a short window around the
    direct sound to the remaining (reverberant) energy. A rough,
    clap-friendly stand-in for the standard DRR metric.
    """
    direct_idx = int(np.argmax(np.abs(h)))
    half_win = int(direct_window_ms * fs / 1000.0 / 2)
    start = max(0, direct_idx - half_win)
    end = min(len(h), direct_idx + half_win)

    direct_energy = np.sum(h[start:end] ** 2)
    reverberant = np.concatenate([h[:start], h[end:]])
    reverberant_energy = np.sum(reverberant ** 2)

    if reverberant_energy < 1e-20:
        return float("inf")
    return float(10 * np.log10((direct_energy + 1e-20) / (reverberant_energy + 1e-20)))
