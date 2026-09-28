"""
stft.py
-------
Short-Time Fourier Transform and spectrogram.

Concepts used: STFT and spectrogram (time-frequency analysis of h[n],
showing how the room's frequency content decays over time).
"""

import numpy as np
from scipy import signal as sig


def compute_stft(x, fs, nperseg=512, noverlap=None, window="hann"):
    """
    Compute the STFT of x[n]. Returns (f, t, Zxx) where Zxx[k, m] is
    the complex STFT coefficient at frequency bin k and time frame m.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = sig.stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx


def spectrogram_db(x, fs, nperseg=512, noverlap=None, ref_db_floor=-100.0):
    """
    Compute a magnitude spectrogram in dB, clipped at `ref_db_floor`
    relative to the peak, ready for plotting or feature extraction.
    """
    f, t, Zxx = compute_stft(x, fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    peak = np.max(mag) + 1e-12
    mag_db = 20 * np.log10(mag / peak + 1e-12)
    mag_db = np.maximum(mag_db, ref_db_floor)
    return f, t, mag_db
