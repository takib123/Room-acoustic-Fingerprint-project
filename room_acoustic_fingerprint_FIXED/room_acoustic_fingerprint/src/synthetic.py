"""
synthetic.py
------------
Synthetic data generators, used for testing and for the demo when a
real microphone/room recording is not available.

Generates a physically-plausible (statistical) room impulse response:
a sparse set of early discrete reflections followed by an
exponentially-decaying diffuse reverberant tail (Poisson-distributed
reflection arrivals, as in the classic Moorer/Schroeder statistical
room models), band-limited to sound natural.
"""

import numpy as np
from scipy import signal as sig


def generate_synthetic_rir(fs=44100, rt60=0.6, duration=1.2, n_early_reflections=12,
                            seed=None):
    """
    Generate a synthetic room impulse response h[n].

    Parameters
    ----------
    rt60 : target reverberation time in seconds (used to set the decay rate)
    duration : total length of h[n] in seconds
    n_early_reflections : number of discrete early reflections to add
                          on top of the diffuse tail
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # --- Diffuse reverberant tail: exponentially-decaying Gaussian noise ---
    # Decay constant chosen so that the envelope drops 60 dB at t = rt60
    decay_rate = 60.0 / (20 * np.log10(np.e) * rt60)  # nepers/sec s.t. exp(-decay_rate*rt60)= -60dB equiv
    # (equivalent, simpler form below)
    tau = rt60 / np.log(1000.0)  # since exp(-t/tau) reaches 1/1000 (~ -60dB) at t = rt60
    envelope = np.exp(-t / tau)
    diffuse = rng.standard_normal(n_samples) * envelope

    # --- Direct sound: strong delta-like spike near the start ---
    h = np.zeros(n_samples)
    direct_idx = int(0.002 * fs)  # ~2ms in, mimicking mic/speaker distance
    h[direct_idx] = 1.0

    # --- Discrete early reflections: sparse, decaying spikes in first ~80ms ---
    early_window_s = 0.08
    reflection_times = np.sort(rng.uniform(0.003, early_window_s, n_early_reflections))
    for rt in reflection_times:
        idx = int(rt * fs)
        if idx < n_samples:
            amp = np.exp(-rt / tau) * rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
            h[idx] += amp

    # Add the diffuse tail (scaled down relative to the direct sound)
    h = h + diffuse * 0.15

    # Band-limit to a natural acoustic range
    nyq = fs / 2
    b, a = sig.butter(4, [80 / nyq, 16000 / nyq], btype="band")
    h = sig.filtfilt(b, a, h)

    # Normalize peak to 1.0
    h = h / (np.max(np.abs(h)) + 1e-12)
    return h


def generate_clap_excitation(fs=44100, duration_ms=3.0, seed=None):
    """
    A short burst of band-limited noise, approximating the acoustic
    excitation of a hand clap (broadband, very short duration).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_ms * fs / 1000.0)
    burst = rng.standard_normal(n)
    window = sig.windows.tukey(n, alpha=0.5)
    burst *= window

    nyq = fs / 2
    b, a = sig.butter(4, [200 / nyq, 10000 / nyq], btype="band")
    burst = sig.filtfilt(b, a, burst)
    return burst / (np.max(np.abs(burst)) + 1e-12)


def simulate_recording(rir, fs, snr_db=30.0, pre_silence_ms=50.0, seed=None):
    """
    Simulate what a microphone would actually record: convolve a clap
    excitation with the room impulse response, add background noise,
    and prepend a bit of silence (as in a real recording).

    This models y[n] = x[n] * h[n] + noise[n], the fundamental LTI
    convolution equation this whole project is built around.
    """
    rng = np.random.default_rng(seed)
    clap = generate_clap_excitation(fs, seed=seed)
    y = sig.fftconvolve(clap, rir, mode="full")

    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(y)) * np.sqrt(noise_power)
    y_noisy = y + noise

    pre_silence = np.zeros(int(pre_silence_ms * fs / 1000.0))
    recorded = np.concatenate([pre_silence, y_noisy])

    recorded = recorded / (np.max(np.abs(recorded)) + 1e-12)
    return recorded
