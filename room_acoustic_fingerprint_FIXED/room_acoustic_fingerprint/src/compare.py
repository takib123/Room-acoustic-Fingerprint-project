"""
compare.py
----------
Robust room-to-room comparison for acoustic fingerprints.

The original implementation compared the raw fingerprint vector directly.
That is unreliable for room recordings because the vector mixes different
units (seconds, dB, Hz, milliseconds, counts), contains zero-filled
missing RT60 values, and contains reflection detections that can move or
appear/disappear slightly between recordings.

This module therefore compares acoustically meaningful feature groups with
fixed, recording-tolerant scales. The UI/API remains unchanged.
"""

import numpy as np


# Typical variation tolerated between repeated clap recordings of the same
# room. These are comparison scales, not physical measurement limits.
FEATURE_SCALES = {
    "rt60_edt_s": 0.18,
    "rt60_t20_s": 0.18,
    "rt60_t30_s": 0.18,
    "clarity_c50_db": 4.0,
    "clarity_c80_db": 4.0,
    "direct_to_reverberant_ratio_db": 4.0,
    "spectral_centroid_hz": 900.0,
    "spectral_flatness": 0.08,
    "reflection_density_per_s": 35.0,
    "num_reflections_detected": 5.0,
    "first_reflection_delay_ms": 4.0,
}

# Group weights. RT60/reverberation describes the room strongly, while
# reflection detection is intentionally given less weight because peak
# picking is the noisiest stage of the pipeline.
GROUP_WEIGHTS = {
    "reverberation": 0.30,
    "decay_shape": 0.20,
    "clarity": 0.10,
    "spectrum": 0.20,
    "bands": 0.12,
    "reflections": 0.08,
}


def _finite(value):
    """Return a finite float or None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value



def _adaptive_scale(a, b, fixed_floor, relative_fraction=0.0):
    """Scale that tolerates both small-room absolute noise and larger values."""
    a = _finite(a)
    b = _finite(b)
    if a is None or b is None:
        return fixed_floor
    magnitude = (abs(a) + abs(b)) / 2.0
    return max(float(fixed_floor), float(relative_fraction) * magnitude)


def _feature_similarity(a, b, scale):
    """
    Convert an absolute feature difference into [0, 1].

    A difference of `scale` gives exp(-1) ~= 0.368. Small differences
    therefore remain highly similar instead of being amplified by a
    two-sample z-score.
    """
    a = _finite(a)
    b = _finite(b)
    if a is None or b is None:
        return None

    scale = max(float(scale), 1e-12)
    return float(np.exp(-abs(a - b) / scale))


def _aggregate(values, weights=None):
    """Weighted mean while ignoring unavailable features."""
    clean = []
    clean_weights = []

    if weights is None:
        weights = [1.0] * len(values)

    for value, weight in zip(values, weights):
        if value is not None and np.isfinite(value) and weight > 0:
            clean.append(float(value))
            clean_weights.append(float(weight))

    if not clean:
        return 0.0

    return float(np.average(clean, weights=clean_weights))


def _band_profile(features):
    """
    Build a scale-independent octave-band energy profile.

    Absolute recording level is deliberately removed. The profile is
    compared by shape, because microphone gain and clap loudness should
    not make the same room look different.
    """
    energies = features.get("octave_band_energy", {})
    values = []

    for band in (125, 250, 500, 1000, 2000, 4000, 8000):
        value = _finite(energies.get(str(band), energies.get(band)))
        if value is None or value <= 0:
            values.append(np.nan)
        else:
            # Log energy prevents one numerically unstable band from
            # dominating the comparison.
            values.append(np.log10(value))

    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)

    if np.count_nonzero(valid) < 2:
        return None

    # Compare relative spectral shape, not absolute energy.
    values[valid] -= np.mean(values[valid])
    norm = np.linalg.norm(values[valid])
    if norm < 1e-12:
        return None

    result = np.zeros_like(values)
    result[valid] = values[valid] / norm
    return result


def _band_similarity(f1, f2):
    """Cosine similarity of relative octave-band energy profiles."""
    p1 = _band_profile(f1)
    p2 = _band_profile(f2)

    if p1 is None or p2 is None:
        return None

    valid = np.isfinite(p1) & np.isfinite(p2)
    if np.count_nonzero(valid) < 2:
        return None

    value = float(np.dot(p1[valid], p2[valid]))
    # Map cosine [-1, 1] to [0, 1].
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def _decay_similarity(f1, f2):
    """
    Compare the normalized Schroeder energy-decay shape.

    This is especially useful for repeated clap recordings: microphone
    gain disappears after EDC normalization, while the room's reverberant
    decay shape remains.
    """
    a = f1.get("decay_signature_db")
    b = f2.get("decay_signature_db")

    if not a or not b:
        return None

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 8:
        return None

    a = np.clip(a[:n], -60.0, 0.0)
    b = np.clip(b[:n], -60.0, 0.0)

    corr = 0.0
    if np.std(a) > 1e-9 and np.std(b) > 1e-9:
        corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        corr = (corr + 1.0) / 2.0

    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    shape_sim = float(np.exp(-rmse / 8.0))

    return float(np.clip(0.90 * corr + 0.10 * shape_sim, 0.0, 1.0))


def _reflection_similarity(f1, f2):
    """
    Compare reflection structure without requiring exact peak indices.

    Reflection count/density are soft features. Delay and level sequences
    are compared after sorting by delay; a few missing/moved peaks therefore
    do not destroy an otherwise good room match.
    """
    density_sim = _feature_similarity(
        f1.get("reflection_density_per_s"),
        f2.get("reflection_density_per_s"),
        _adaptive_scale(
            f1.get("reflection_density_per_s"), f2.get("reflection_density_per_s"), 20.0, 0.60
        ),
    )
    count_sim = _feature_similarity(
        f1.get("num_reflections_detected"),
        f2.get("num_reflections_detected"),
        _adaptive_scale(
            f1.get("num_reflections_detected"), f2.get("num_reflections_detected"), 4.0, 0.60
        ),
    )

    parts = []
    weights = []
    if density_sim is not None:
        parts.append(density_sim)
        weights.append(0.40)
    if count_sim is not None:
        parts.append(count_sim)
        weights.append(0.20)

    delays1 = np.asarray(f1.get("reflection_delays_ms") or [], dtype=float)
    delays2 = np.asarray(f2.get("reflection_delays_ms") or [], dtype=float)
    levels1 = np.asarray(f1.get("reflection_levels_db") or [], dtype=float)
    levels2 = np.asarray(f2.get("reflection_levels_db") or [], dtype=float)

    # Ignore invalid entries and compare the strongest/earliest common part.
    delays1 = delays1[np.isfinite(delays1)]
    delays2 = delays2[np.isfinite(delays2)]

    if len(delays1) and len(delays2):
        n = min(len(delays1), len(delays2), 10)
        d1 = np.sort(delays1)[:n]
        d2 = np.sort(delays2)[:n]

        # Delay tolerance grows mildly for later reflections.
        delay_scale = 4.0 + 0.05 * np.maximum(d1, d2)
        delay_sim = float(np.mean(np.exp(-np.abs(d1 - d2) / delay_scale)))
        parts.append(delay_sim)
        weights.append(0.25)

    if len(levels1) and len(levels2):
        l1 = levels1[np.isfinite(levels1)]
        l2 = levels2[np.isfinite(levels2)]
        n = min(len(l1), len(l2), 10)
        if n:
            l1 = np.sort(l1)[::-1][:n]
            l2 = np.sort(l2)[::-1][:n]
            level_sim = float(np.mean(np.exp(-np.abs(l1 - l2) / 6.0)))
            parts.append(level_sim)
            weights.append(0.15)

    if not parts:
        return None

    return _aggregate(parts, weights)


def cosine_similarity(v1, v2):
    """Standard cosine similarity for callers that need the utility."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2))


def euclidean_distance(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    return float(np.linalg.norm(v1 - v2))


def normalized_euclidean_distance(v1, v2):
    """
    Kept for API compatibility.

    The old implementation standardized each dimension using only the two
    compared rooms. That makes every non-identical feature become +/-1 and
    is not a meaningful room-distance measure. This function now computes a
    stable RMS distance after robust global-scale normalization.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n = min(len(v1), len(v2))

    if n == 0:
        return 0.0

    a = v1[:n]
    b = v2[:n]

    # A fixed scale based on the magnitude of the two values avoids the
    # pairwise z-score problem while preventing huge units from dominating.
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.sqrt(np.mean(((a - b) / scale) ** 2)))


def frequency_response_correlation(mag_db_1, mag_db_2):
    """
    Pearson correlation between two frequency-response magnitude curves.
    """
    a = np.asarray(mag_db_1, dtype=np.float64)
    b = np.asarray(mag_db_2, dtype=np.float64)
    n = min(len(a), len(b))

    if n < 2:
        return 0.0

    a, b = a[:n], b[:n]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0

    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _compare_core_features(f1, f2):
    """Return feature-level similarities used by the final room score."""
    groups = {}

    groups["reverberation"] = _aggregate(
        [
            _feature_similarity(
                f1.get("rt60_edt_s"), f2.get("rt60_edt_s"),
                _adaptive_scale(f1.get("rt60_edt_s"), f2.get("rt60_edt_s"), 0.12, 0.35)
            ),
            _feature_similarity(
                f1.get("rt60_t20_s"), f2.get("rt60_t20_s"),
                _adaptive_scale(f1.get("rt60_t20_s"), f2.get("rt60_t20_s"), 0.12, 0.35)
            ),
            _feature_similarity(
                f1.get("rt60_t30_s"), f2.get("rt60_t30_s"),
                _adaptive_scale(f1.get("rt60_t30_s"), f2.get("rt60_t30_s"), 0.12, 0.35)
            ),
        ]
    )

    decay_sim = _decay_similarity(f1, f2)
    groups["decay_shape"] = decay_sim if decay_sim is not None else groups["reverberation"]

    groups["clarity"] = _aggregate(
        [
            _feature_similarity(f1.get("clarity_c50_db"), f2.get("clarity_c50_db"), 6.0),
            _feature_similarity(f1.get("clarity_c80_db"), f2.get("clarity_c80_db"), 6.0),
            _feature_similarity(
                f1.get("direct_to_reverberant_ratio_db"),
                f2.get("direct_to_reverberant_ratio_db"),
                6.0,
            ),
        ]
    )

    groups["spectrum"] = _aggregate(
        [
            _feature_similarity(
                f1.get("spectral_centroid_hz"),
                f2.get("spectral_centroid_hz"),
                _adaptive_scale(
                    f1.get("spectral_centroid_hz"), f2.get("spectral_centroid_hz"), 300.0, 0.30
                ),
            ),
            _feature_similarity(
                f1.get("spectral_flatness"),
                f2.get("spectral_flatness"),
                FEATURE_SCALES["spectral_flatness"],
            ),
        ]
    )

    band_sim = _band_similarity(f1, f2)
    groups["bands"] = band_sim if band_sim is not None else 0.5

    reflection_sim = _reflection_similarity(f1, f2)
    groups["reflections"] = reflection_sim if reflection_sim is not None else 0.5

    return groups


def _robust_feature_vector(f):
    """
    Construct a fixed-scale vector used only for the displayed cosine metric.

    Features are converted to dimensionless, room-acoustic scales. Absolute
    energy is excluded because the preprocessing normalizes recording gain.
    """
    vals = []

    mapping = [
        ("rt60_edt_s", 1.0),
        ("rt60_t20_s", 1.0),
        ("rt60_t30_s", 1.0),
        ("clarity_c50_db", 5.0),
        ("clarity_c80_db", 5.0),
        ("direct_to_reverberant_ratio_db", 5.0),
        ("spectral_centroid_hz", 1000.0),
        ("spectral_flatness", 0.1),
        ("reflection_density_per_s", 40.0),
        ("num_reflections_detected", 5.0),
    ]

    for key, scale in mapping:
        value = _finite(f.get(key))
        if value is None:
            value = 0.0
        vals.append(value / scale)

    profile = _band_profile(f)
    if profile is not None:
        vals.extend(profile.tolist())
    else:
        vals.extend([0.0] * 7)

    return np.asarray(vals, dtype=np.float64)


def compare_fingerprints(fp1, fp2):
    """
    Compare two room fingerprints.

    The returned keys intentionally match the original UI contract, so no
    Streamlit UI changes are required.
    """
    f1 = fp1.get("features", {})
    f2 = fp2.get("features", {})

    groups = _compare_core_features(f1, f2)

    overall = _aggregate(
        [groups[name] for name in GROUP_WEIGHTS],
        [GROUP_WEIGHTS[name] for name in GROUP_WEIGHTS],
    )

    # If the strongest room-identity signals agree (decay shape, spectral
    # shape and octave profile), do not let an unstable single-number
    # estimate such as C50 or a noisy RT60 fit overturn the match.  This is
    # important for repeated clap recordings where the direct peak/noise
    # floor can move between takes.
    if (
        groups["decay_shape"] >= 0.90
        and groups["bands"] >= 0.90
        and groups["spectrum"] >= 0.75
    ):
        consensus = _aggregate(
            [groups["decay_shape"], groups["spectrum"], groups["bands"], groups["reflections"]],
            [0.55, 0.25, 0.15, 0.05],
        )
        overall = max(overall, consensus)

    # A stable cosine over dimensionless acoustic descriptors.
    robust_cos = cosine_similarity(_robust_feature_vector(f1), _robust_feature_vector(f2))

    # Use the group score as the main metric because it is bounded and
    # explicitly tolerant of ordinary repeated-recording variation.
    overall_percent = float(np.clip(overall * 100.0, 0.0, 100.0))

    v1 = fp1.get("vector", [])
    v2 = fp2.get("vector", [])
    raw_euclid = euclidean_distance(v1, v2)
    stable_distance = normalized_euclidean_distance(v1, v2)

    key_diffs = {
        "rt60_t30_diff_s": _diff(f1.get("rt60_t30_s"), f2.get("rt60_t30_s")),
        "rt60_edt_diff_s": _diff(f1.get("rt60_edt_s"), f2.get("rt60_edt_s")),
        "clarity_c50_diff_db": _diff(f1.get("clarity_c50_db"), f2.get("clarity_c50_db")),
        "spectral_centroid_diff_hz": _diff(
            f1.get("spectral_centroid_hz"), f2.get("spectral_centroid_hz")
        ),
        "reflection_count_diff": _difference_or_none(
            f1.get("num_reflections_detected"), f2.get("num_reflections_detected")
        ),
        "rt60_similarity": groups["reverberation"],
        "decay_shape_similarity": groups["decay_shape"],
        "clarity_similarity": groups["clarity"],
        "spectrum_similarity": groups["spectrum"],
        "octave_profile_similarity": groups["bands"],
        "reflection_similarity": groups["reflections"],
    }

    return {
        "room_1": fp1.get("room_name", "room_1"),
        "room_2": fp2.get("room_name", "room_2"),
        "cosine_similarity": float(np.clip(robust_cos, -1.0, 1.0)),
        "euclidean_distance": raw_euclid,
        "normalized_euclidean_distance": stable_distance,
        "overall_similarity_percent": overall_percent,
        "key_feature_differences": key_diffs,
    }


def _difference_or_none(a, b):
    a = _finite(a)
    b = _finite(b)
    if a is None or b is None:
        return None
    return float(a - b)


def _diff(a, b):
    return _difference_or_none(a, b)
