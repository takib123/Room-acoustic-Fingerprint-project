"""
energy_decay.py
---------------
Energy decay analysis of the room impulse response using the
Schroeder backward-integration method (ISO 3382 standard approach).

Concepts used: Energy decay analysis (foundation for RT60 estimation).
"""

import numpy as np


def schroeder_integration(h):
    """
    Schroeder backward integration:

        EDC(t) = integral_t^infinity h(tau)^2 dtau

    Implemented in discrete time as a reverse cumulative sum of h^2.
    This converts the (noisy, oscillating) squared impulse response
    into a smooth, monotonically decreasing energy decay curve (EDC),
    which is what RT60 is actually measured from.
    """
    energy = h.astype(np.float64) ** 2
    # Reverse cumulative sum = integral from t to the end
    edc = np.cumsum(energy[::-1])[::-1]
    return edc


def edc_to_db(edc):
    """Normalize the EDC so it starts at 0 dB and express it in dB."""
    edc_norm = edc / (edc[0] + 1e-20)
    edc_db = 10 * np.log10(edc_norm + 1e-20)
    return edc_db


def compute_edc_db(h):
    """Convenience wrapper: h[n] -> Schroeder EDC in dB."""
    return edc_to_db(schroeder_integration(h))
