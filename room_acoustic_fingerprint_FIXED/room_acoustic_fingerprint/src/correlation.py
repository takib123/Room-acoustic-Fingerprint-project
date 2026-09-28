"""
correlation.py
--------------
Correlation-based analysis.

Concepts used: Correlation, and its relationship to convolution
(cross-correlation of x with y equals convolution of x with the
time-reversed y).
"""

import numpy as np
from scipy import signal as sig


def cross_correlate(x, y, mode="full"):
    """
    Cross-correlation of x and y: measures similarity between x and
    time-shifted copies of y. Used to find the delay between a
    reference (e.g. excitation) signal and a recorded/reflected copy.
    """
    return sig.correlate(x, y, mode=mode, method="fft")


def autocorrelate(x, mode="full"):
    """Autocorrelation of x with itself -- used to reveal periodicities
    and self-similar reflection patterns (e.g. flutter echo)."""
    return cross_correlate(x, x, mode=mode)


def find_lag_samples(x, y, fs):
    """
    Find the lag (in samples and seconds) at which y best matches a
    shifted copy of x, via the argmax of the cross-correlation.

    Returns (lag_samples, lag_seconds). Positive lag means y is
    delayed relative to x (as expected for a reflected copy of a
    reference excitation).
    """
    corr = cross_correlate(x, y)
    lags = sig.correlation_lags(len(x), len(y))
    best = lags[np.argmax(corr)]
    return best, best / fs


def normalized_autocorrelation(x):
    """Autocorrelation normalized so that the zero-lag value is 1.0,
    useful for comparing reflection strengths across recordings."""
    r = autocorrelate(x)
    r = r / (np.max(np.abs(r)) + 1e-12)
    return r
