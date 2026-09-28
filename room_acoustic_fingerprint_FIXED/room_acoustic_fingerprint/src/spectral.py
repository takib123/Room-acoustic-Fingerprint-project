"""
spectral.py
-----------
FFT and frequency-response analysis of the room impulse response.

Concepts used: FFT, frequency response of an LTI system (H(f) is
literally the frequency response of the room, obtained by taking the
Fourier Transform of h[n]).
"""

import numpy as np


def compute_fft(h, fs):
    """
    Compute the single-sided magnitude/phase spectrum of h[n], i.e.
    the room's frequency response H(f).
    """
    n = len(h)
    n_fft = int(2 ** np.ceil(np.log2(n)))
    H = np.fft.rfft(h, n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    magnitude = np.abs(H)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    phase = np.angle(H)

    return freqs, magnitude_db, phase


def octave_band_energy(h, fs, bands=None):
    """
    Compute the energy of h[n] within standard octave bands using
    band-pass (Butterworth) filtering. Useful for a frequency-resolved
    view of reverberation (e.g. RT60 per octave band).

    Returns a dict {center_freq_hz: energy}.
    """
    from scipy import signal as sig

    if bands is None:
        bands = [125, 250, 500, 1000, 2000, 4000, 8000]

    nyq = fs / 2.0
    energies = {}
    for center in bands:
        low = center / np.sqrt(2)
        high = center * np.sqrt(2)
        if high >= nyq:
            high = nyq * 0.999
        if low <= 0:
            continue
        b, a = sig.butter(4, [low / nyq, high / nyq], btype="band")
        filtered = sig.filtfilt(b, a, h)
        energies[center] = float(np.sum(filtered ** 2))

    return energies


def spectral_centroid(freqs, magnitude_db):
    """Weighted-mean frequency of the spectrum -- a single-number
    descriptor of 'brightness' used as a fingerprint feature."""
    mag_lin = 10 ** (magnitude_db / 20.0)
    mag_lin = np.maximum(mag_lin, 0.0)
    if np.sum(mag_lin) < 1e-12:
        return 0.0
    return float(np.sum(freqs * mag_lin) / np.sum(mag_lin))


def spectral_flatness(magnitude_db):
    """
    Ratio of the geometric mean to the arithmetic mean of the
    magnitude spectrum -- close to 1.0 for noise-like (flat) spectra,
    close to 0 for tonal/peaky spectra.
    """
    mag_lin = np.maximum(10 ** (magnitude_db / 20.0), 1e-12)
    gmean = np.exp(np.mean(np.log(mag_lin)))
    amean = np.mean(mag_lin)
    return float(gmean / amean) if amean > 0 else 0.0
