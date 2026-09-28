"""
io_utils.py
-----------
Audio input/output helpers.

Concepts used: Sampling / discrete-time signals (Sec: "Sampling and
discrete-time signals" in the project brief).
"""

import numpy as np
from scipy.io import wavfile


def load_wav(path):
    """
    Load a .wav file and return (signal, fs) where `signal` is a
    float64 numpy array in the range [-1, 1] (mono).

    This is where a continuous-time acoustic pressure wave, already
    sampled by the recording device's ADC, enters our discrete-time
    pipeline as x[n].
    """
    fs, data = wavfile.read(path)
    data = np.asarray(data)

    # Convert to mono if stereo/multi-channel
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize integer PCM formats to float64 in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)

    return data, fs


def save_wav(path, signal, fs):
    """Save a float64 [-1, 1] signal to a 16-bit PCM .wav file."""
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    wavfile.write(path, fs, pcm)


def try_record(duration_sec=3.0, fs=44100):
    """
    Attempt to record `duration_sec` seconds of audio from the default
    microphone using `sounddevice`. Returns (signal, fs) on success.

    Raises RuntimeError if no microphone / sounddevice is available,
    which is expected in headless/server environments -- in that case
    the caller should fall back to `load_wav()` or synthetic data
    (see src/synthetic.py) for development and testing.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Install it with "
            "`pip install sounddevice` to enable live recording, "
            "or supply a .wav file / use synthetic data instead."
        ) from exc

    print(f"Recording for {duration_sec:.1f}s at {fs} Hz... clap now!")
    recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    return recording.flatten(), fs
