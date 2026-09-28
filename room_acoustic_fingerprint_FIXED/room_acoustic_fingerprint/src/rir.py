"""
rir.py
------
Room Impulse Response (RIR) estimation.

Concepts used: LTI systems, convolution, correlation, FFT-based
deconvolution.

A room, acoustically, behaves (approximately) as a Linear
Time-Invariant system: y[n] = x[n] * h[n], where x[n] is the emitted
sound, h[n] is the room impulse response, and * is convolution. If we
excite the room with a signal that is approximately a Kronecker delta
(a hand clap), the recorded signal itself IS (approximately) h[n],
since delta[n] * h[n] = h[n]. For a more rigorous excitation (e.g. a
logarithmic sine sweep), h[n] must be recovered via deconvolution.
"""

import numpy as np
from scipy import signal as sig


def estimate_rir_from_clap(recorded, fs, pre_ms=5.0, tail_ms=1500.0):
    """
    Treat a recorded clap as an approximate room impulse response.

    Steps:
      1. Locate the direct-sound onset (first large peak) using the
         signal envelope.
      2. Trim a short pre-roll before the onset and a decay tail after
         it, discarding silence/noise outside that window.

    This is the simplest, most classroom-friendly way to obtain h[n]
    and is what the "Record Audio / Clap" branch of the pipeline uses.
    """
    envelope = np.abs(sig.hilbert(recorded))
    onset = int(np.argmax(envelope))

    pre_samples = int(pre_ms * fs / 1000.0)
    tail_samples = int(tail_ms * fs / 1000.0)

    start = max(0, onset - pre_samples)
    end = min(len(recorded), onset + tail_samples)

    h = recorded[start:end].copy()
    return h, onset - start  # h[n], index of direct sound within h


def estimate_rir_deconvolution(recorded, excitation, regularization=1e-3):
    """
    Recover h[n] from a recorded response to a *known* excitation
    signal (e.g. a sine sweep or MLS) via frequency-domain
    deconvolution:

        H(f) = Y(f) / X(f)   (regularized to avoid division by ~0)
        h[n] = IFFT(H(f))

    This models the room as an LTI system and inverts the convolution
    y[n] = x[n] * h[n] performed by the room.
    """
    n = len(recorded) + len(excitation) - 1
    n_fft = int(2 ** np.ceil(np.log2(n)))

    Y = np.fft.rfft(recorded, n_fft)
    X = np.fft.rfft(excitation, n_fft)

    # Regularized (Tikhonov-style) inverse to keep the deconvolution stable
    X_mag2 = np.abs(X) ** 2
    eps = regularization * np.max(X_mag2)
    H = (Y * np.conj(X)) / (X_mag2 + eps)

    h = np.fft.irfft(H, n_fft)
    return h[: len(recorded)]


def trim_noise_floor(h, fs, floor_db=-60.0, min_len_ms=50.0):
    """
    Trim the tail of an impulse response once its envelope has decayed
    `floor_db` below its peak, to avoid analyzing pure noise.
    """
    envelope = np.abs(sig.hilbert(h))
    peak = np.max(envelope)
    if peak < 1e-12:
        return h
    envelope_db = 20 * np.log10(envelope / peak + 1e-12)

    below_floor = np.where(envelope_db < floor_db)[0]
    min_len = int(min_len_ms * fs / 1000.0)
    if len(below_floor) == 0:
        return h
    cutoff = max(below_floor[0], min_len)
    return h[:cutoff]
