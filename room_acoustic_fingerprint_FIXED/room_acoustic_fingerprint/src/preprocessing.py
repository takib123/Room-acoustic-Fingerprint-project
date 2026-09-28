"""
preprocessing.py
----------------
Sampling / pre-processing / noise filtering stage.

Concepts used: Sampling and discrete-time signals, LTI systems
(the filters below ARE LTI systems), convolution (filtering == convolution
with a filter's impulse response).
"""

import numpy as np
from scipy import signal as sig


def remove_dc(x):
    """Remove DC offset (mean) from the signal."""
    return x - np.mean(x)


def normalize(x, target_peak=1.0):
    """Peak-normalize a signal to `target_peak`."""
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return x
    return x * (target_peak / peak)


def resample_signal(x, fs_in, fs_out):
    """Resample x[n] from fs_in to fs_out using polyphase filtering."""
    if fs_in == fs_out:
        return x, fs_out
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out // g), int(fs_in // g)
    y = sig.resample_poly(x, up, down)
    return y, fs_out


def highpass_filter(x, fs, cutoff=20.0, order=4):
    """
    Butterworth high-pass filter to remove sub-audible rumble / DC drift.

    This is a concrete example of an LTI system: it is defined entirely
    by its impulse response h[n] (implicitly, via its coefficients), and
    applying it to x[n] is a convolution y[n] = x[n] * h[n].
    """
    nyq = fs / 2.0
    b, a = sig.butter(order, cutoff / nyq, btype="high")
    return sig.filtfilt(b, a, x)


def bandpass_filter(x, fs, low=80.0, high=16000.0, order=4):
    """Butterworth band-pass filter, used to restrict analysis to the
    audible band relevant to room acoustics and to reduce noise."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)
    b, a = sig.butter(order, [low_n, high_n], btype="band")
    return sig.filtfilt(b, a, x)


def spectral_noise_gate(x, fs, noise_frames=2048, reduction_db=12.0):
    """
    A light-weight spectral noise gate: estimate a noise floor spectrum
    from the quietest frame of the recording and subtractively reduce it
    across the whole signal (simple spectral subtraction).
    """
    nperseg = 1024
    f, t, Zxx = sig.stft(x, fs=fs, nperseg=nperseg)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise spectrum from the quietest column (lowest energy frame)
    frame_energy = np.sum(mag ** 2, axis=0)
    noise_col = np.argmin(frame_energy)
    noise_spectrum = mag[:, noise_col]

    reduction = 10 ** (-reduction_db / 20.0)
    mag_clean = np.maximum(mag - noise_spectrum[:, None] * (1 - reduction), 0.0)

    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, x_clean = sig.istft(Zxx_clean, fs=fs, nperseg=nperseg)
    # istft can change length slightly; trim/pad to original length
    if len(x_clean) > len(x):
        x_clean = x_clean[: len(x)]
    else:
        x_clean = np.pad(x_clean, (0, len(x) - len(x_clean)))
    return x_clean


def preprocess_pipeline(x, fs, target_fs=44100, do_noise_gate=True):
    """
    Full pre-processing chain:
      raw x[n] -> resample -> DC removal -> band-pass filter
               -> (optional) spectral noise gate -> normalize
    """
    x, fs = resample_signal(x, fs, target_fs)
    x = remove_dc(x)
    x = bandpass_filter(x, fs, low=50.0, high=18000.0)
    if do_noise_gate:
        x = spectral_noise_gate(x, fs)
    x = normalize(x)
    return x, fs
