"""
synchronization.py
------------------
Core algorithm: compares the audio energy envelope with the mouth-aperture
signal to estimate the temporal offset between them.

Algorithm summary
-----------------
1. Both signals are resampled to a common time grid (TARGET_FPS Hz).
2. Each signal is Gaussian-smoothed to reduce frame-by-frame noise.
3. Both signals are zero-mean, unit-variance normalized.
4. Normalized cross-correlation is computed across all feasible lags.
5. The lag at the peak of the correlation gives the estimated offset.
6. A "peak quality ratio" assesses whether the peak is distinct enough to trust.
7. Based on the estimated offset and peak quality, the result is classified as
   ALIGNED, MISMATCH, or INCONCLUSIVE.

Sign convention (documented)
----------------------------
  Positive offset → audio leads video  (audio event occurs before the mouth opens)
  Negative offset → video leads audio  (mouth opens before the corresponding audio)

Thresholds are configurable and are NOT claimed to be universally scientifically valid.
They are reasonable defaults that can be tuned for a specific use case.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
from scipy.ndimage import gaussian_filter1d



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_FPS: float = 25.0           # Common time grid frequency (Hz)
GAUSSIAN_SIGMA: float = 2.0        # Smoothing sigma (samples at TARGET_FPS)

# Alignment decision thresholds
ALIGNED_OFFSET_THRESHOLD_MS: float = 100.0   # |offset| < this → ALIGNED
MISMATCH_OFFSET_THRESHOLD_MS: float = 100.0  # |offset| >= this → MISMATCH candidate
PEAK_QUALITY_RATIO_THRESHOLD: float = 1.8    # peak / mean-of-rest must exceed this
MIN_SIGNAL_STD: float = 0.01                 # minimum std for a signal to be "active"
MIN_SIGNAL_LENGTH_SEC: float = 1.0           # minimum overlap duration (seconds)


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------

class SyncVerdict(str, Enum):
    ALIGNED = "ALIGNED"
    MISMATCH = "MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class SyncResult:
    """Complete synchronization result."""
    verdict: SyncVerdict

    # Estimated offset in milliseconds (positive = audio leads video)
    estimated_offset_ms: Optional[float]

    # Peak of the normalized cross-correlation [-1, 1]
    peak_correlation: Optional[float]

    # Peak quality ratio (peak / mean-of-rest); higher = more confident
    peak_quality_ratio: Optional[float]

    # Signal quality flags
    audio_signal_active: bool
    mouth_signal_active: bool

    # Overlap duration used for analysis (seconds)
    overlap_duration_sec: float

    # Reason string explaining the verdict
    reason: str

    # The resampled, normalized signals (for visualization)
    common_timestamps: Optional[np.ndarray] = None
    mouth_signal_normalized: Optional[np.ndarray] = None
    audio_signal_normalized: Optional[np.ndarray] = None

    # Full cross-correlation (for visualization)
    correlation: Optional[np.ndarray] = None
    lags_ms: Optional[np.ndarray] = None

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal preprocessing helpers
# ---------------------------------------------------------------------------

def _resample_to_grid(
    values: np.ndarray,
    original_timestamps: np.ndarray,
    target_fps: float = TARGET_FPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample an irregularly-sampled signal to a uniform time grid at `target_fps`.

    Returns (resampled_values, uniform_timestamps).
    Uses linear interpolation (np.interp) for generality.
    """
    if len(values) == 0 or len(original_timestamps) == 0:
        return np.array([]), np.array([])

    t_start = original_timestamps[0]
    t_end = original_timestamps[-1]
    if t_end <= t_start:
        return values[:1], original_timestamps[:1]

    n_samples = max(2, int((t_end - t_start) * target_fps) + 1)
    uniform_t = np.linspace(t_start, t_end, n_samples)
    resampled = np.interp(uniform_t, original_timestamps, values)
    return resampled, uniform_t


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization. Returns zeros if std ≈ 0."""
    mu = signal.mean()
    std = signal.std()
    if std < MIN_SIGNAL_STD:
        return np.zeros_like(signal)
    return (signal - mu) / std


def _smooth(signal: np.ndarray, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Apply Gaussian smoothing."""
    if len(signal) < 3:
        return signal
    return gaussian_filter1d(signal.astype(float), sigma=sigma)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def _cross_correlate(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation of two equal-length signals.
    Returns (correlation_array, lags_in_milliseconds).

    When called as _cross_correlate(mouth, audio, fps), the result at lag k
    (samples) means: the correlation between mouth[t] and audio[t - k].
    If audio leads mouth by k0 samples, the peak appears at lag k0 > 0,
    consistent with the convention: positive offset = audio leads video.
    """
    n = len(signal_a)
    corr = np.correlate(signal_a, signal_b, mode="full")

    # Normalize by n to make the metric comparable across signal lengths
    corr = corr / n

    # Lags array: from -(n-1) to +(n-1) samples
    lags_samples = np.arange(-(n - 1), n)
    lags_ms = (lags_samples / fps) * 1000.0  # convert to milliseconds

    return corr, lags_ms


def _peak_quality_ratio(corr: np.ndarray, peak_idx: int, neighbourhood: int = 5) -> float:
    """
    Compute a quality ratio for the detected peak.

    Ratio = corr[peak_idx] / mean(|corr| excluding neighbourhood around peak).
    Higher ratio → peak is more distinct (more reliable estimate).
    """
    mask = np.ones(len(corr), dtype=bool)
    lo = max(0, peak_idx - neighbourhood)
    hi = min(len(corr), peak_idx + neighbourhood + 1)
    mask[lo:hi] = False

    rest = np.abs(corr[mask])
    if len(rest) == 0 or rest.mean() < 1e-10:
        return 1.0
    return float(corr[peak_idx] / rest.mean())


def _secondary_peak_ratio(corr: np.ndarray, peak_idx: int, neighbourhood: int = 10) -> float:
    """
    Find the highest peak *outside* the neighbourhood of the primary peak,
    and return (secondary_peak / primary_peak).

    If this ratio is close to 1.0 (e.g., > 0.95), the correlation has two
    nearly-equal competing peaks and the result is ambiguous.
    Returns 0.0 if no secondary peak exists.
    """
    primary_val = corr[peak_idx]
    if primary_val <= 0:
        return 0.0

    mask = np.ones(len(corr), dtype=bool)
    lo = max(0, peak_idx - neighbourhood)
    hi = min(len(corr), peak_idx + neighbourhood + 1)
    mask[lo:hi] = False

    outside = corr[mask]
    if len(outside) == 0:
        return 0.0

    secondary_val = float(outside.max())
    return secondary_val / primary_val


# ---------------------------------------------------------------------------
# Main synchronization function
# ---------------------------------------------------------------------------

def estimate_sync(
    mouth_timestamps: np.ndarray,
    mouth_apertures: np.ndarray,
    audio_timestamps: np.ndarray,
    audio_energy: np.ndarray,
    aligned_threshold_ms: float = ALIGNED_OFFSET_THRESHOLD_MS,
    peak_quality_threshold: float = PEAK_QUALITY_RATIO_THRESHOLD,
) -> SyncResult:
    """
    Estimate the audio-visual synchronization offset.

    Parameters
    ----------
    mouth_timestamps : 1D array, seconds
    mouth_apertures : 1D array, normalized mouth opening values
    audio_timestamps : 1D array, seconds
    audio_energy : 1D array, short-time energy values
    aligned_threshold_ms : float
        Maximum |offset| (ms) to classify as ALIGNED.
    peak_quality_threshold : float
        Minimum peak quality ratio to trust the correlation peak.

    Returns
    -------
    SyncResult
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ------------------------------------------------------------------
    # 1. Basic validation
    # ------------------------------------------------------------------
    if len(mouth_timestamps) < 5 or len(audio_timestamps) < 5:
        return SyncResult(
            verdict=SyncVerdict.INCONCLUSIVE,
            estimated_offset_ms=None,
            peak_correlation=None,
            peak_quality_ratio=None,
            audio_signal_active=False,
            mouth_signal_active=False,
            overlap_duration_sec=0.0,
            reason="Insufficient data: fewer than 5 data points in one or both signals.",
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 2. Find temporal overlap
    # ------------------------------------------------------------------
    t_start = max(mouth_timestamps[0], audio_timestamps[0])
    t_end = min(mouth_timestamps[-1], audio_timestamps[-1])
    overlap_sec = max(0.0, t_end - t_start)

    if overlap_sec < MIN_SIGNAL_LENGTH_SEC:
        return SyncResult(
            verdict=SyncVerdict.INCONCLUSIVE,
            estimated_offset_ms=None,
            peak_correlation=None,
            peak_quality_ratio=None,
            audio_signal_active=False,
            mouth_signal_active=False,
            overlap_duration_sec=overlap_sec,
            reason=f"Temporal overlap ({overlap_sec:.2f}s) is too short for reliable analysis.",
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 3. Resample to common grid over the overlap window
    # ------------------------------------------------------------------
    # Clip to overlap
    m_mask = (mouth_timestamps >= t_start) & (mouth_timestamps <= t_end)
    a_mask = (audio_timestamps >= t_start) & (audio_timestamps <= t_end)

    m_t = mouth_timestamps[m_mask]
    m_v = mouth_apertures[m_mask]
    a_t = audio_timestamps[a_mask]
    a_v = audio_energy[a_mask]

    m_resampled, common_t = _resample_to_grid(m_v, m_t, TARGET_FPS)
    a_resampled, _ = _resample_to_grid(a_v, a_t, TARGET_FPS)

    # Match lengths to the shorter after resampling
    min_len = min(len(m_resampled), len(a_resampled), len(common_t))
    m_resampled = m_resampled[:min_len]
    a_resampled = a_resampled[:min_len]
    common_t = common_t[:min_len]

    if min_len < int(TARGET_FPS * MIN_SIGNAL_LENGTH_SEC):
        return SyncResult(
            verdict=SyncVerdict.INCONCLUSIVE,
            estimated_offset_ms=None,
            peak_correlation=None,
            peak_quality_ratio=None,
            audio_signal_active=False,
            mouth_signal_active=False,
            overlap_duration_sec=overlap_sec,
            reason="Not enough samples after resampling to perform correlation.",
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 4. Smooth and normalize
    # ------------------------------------------------------------------
    m_smooth = _smooth(m_resampled)
    a_smooth = _smooth(a_resampled)

    mouth_std = m_smooth.std()
    audio_std = a_smooth.std()

    mouth_active = mouth_std >= MIN_SIGNAL_STD
    audio_active = audio_std >= MIN_SIGNAL_STD

    m_norm = _normalize(m_smooth)
    a_norm = _normalize(a_smooth)

    if not mouth_active:
        warnings.append(
            "Mouth-aperture signal has very low variance — the speaker may not be moving "
            "their mouth noticeably, or face detection was poor."
        )
    if not audio_active:
        warnings.append(
            "Audio energy signal has very low variance — the audio may be silent or near-silent."
        )

    if not mouth_active or not audio_active:
        return SyncResult(
            verdict=SyncVerdict.INCONCLUSIVE,
            estimated_offset_ms=None,
            peak_correlation=None,
            peak_quality_ratio=None,
            audio_signal_active=audio_active,
            mouth_signal_active=mouth_active,
            overlap_duration_sec=overlap_sec,
            reason=(
                "One or both signals are nearly flat (insufficient variation). "
                "Cannot perform meaningful cross-correlation."
            ),
            common_timestamps=common_t,
            mouth_signal_normalized=m_norm,
            audio_signal_normalized=a_norm,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 5. Cross-correlation
    # ------------------------------------------------------------------
    # Convention: correlate(mouth, audio) → peak at positive lag k
    # means audio leads mouth by k samples (positive offset = audio leads).
    # np.correlate(a, b, 'full')[n-1+k] = sum_t a[t]*b[t-k], so for
    # mouth[t] = audio[t-k0] (audio leads by k0), the peak is at lag=k0 > 0.
    corr, lags_ms = _cross_correlate(m_norm, a_norm, TARGET_FPS)

    # Find peak of positive correlation
    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])
    peak_lag_ms = float(lags_ms[peak_idx])

    quality_ratio = _peak_quality_ratio(corr, peak_idx)
    secondary_ratio = _secondary_peak_ratio(corr, peak_idx)

    # ------------------------------------------------------------------
    # 6. Classify verdict
    # ------------------------------------------------------------------
    # Check for competing near-equal secondary peak (ambiguous correlation)
    SECONDARY_AMBIGUITY_THRESHOLD = 0.95  # secondary/primary > this → inconclusive
    if quality_ratio < peak_quality_threshold:
        verdict = SyncVerdict.INCONCLUSIVE
        reason = (
            f"Cross-correlation peak quality ratio ({quality_ratio:.2f}) is below the "
            f"threshold ({peak_quality_threshold:.2f}). The correlation does not show a "
            "clear dominant alignment point — result is inconclusive."
        )
    elif secondary_ratio > SECONDARY_AMBIGUITY_THRESHOLD:
        verdict = SyncVerdict.INCONCLUSIVE
        reason = (
            f"A competing secondary correlation peak ({secondary_ratio:.2%} of the primary) "
            "was found. The timing estimate is ambiguous — result is inconclusive."
        )
        warnings.append(
            "Multiple near-equal correlation peaks detected. This may indicate a highly "
            "periodic signal or insufficient signal variation."
        )
    elif abs(peak_lag_ms) < aligned_threshold_ms:
        verdict = SyncVerdict.ALIGNED
        reason = (
            f"Estimated offset {peak_lag_ms:+.0f} ms is within the ±{aligned_threshold_ms:.0f} ms "
            "alignment tolerance. Audio and video appear to be synchronized."
        )
    else:
        verdict = SyncVerdict.MISMATCH
        direction = "audio leads video" if peak_lag_ms > 0 else "video leads audio"
        reason = (
            f"Estimated offset {peak_lag_ms:+.0f} ms exceeds the ±{aligned_threshold_ms:.0f} ms "
            f"alignment tolerance ({direction}). A synchronization mismatch is possible."
        )

    return SyncResult(
        verdict=verdict,
        estimated_offset_ms=peak_lag_ms if verdict != SyncVerdict.INCONCLUSIVE else None,
        peak_correlation=peak_val,
        peak_quality_ratio=quality_ratio,
        audio_signal_active=audio_active,
        mouth_signal_active=mouth_active,
        overlap_duration_sec=overlap_sec,
        reason=reason,
        common_timestamps=common_t,
        mouth_signal_normalized=m_norm,
        audio_signal_normalized=a_norm,
        correlation=corr,
        lags_ms=lags_ms,
        errors=errors,
        warnings=warnings,
    )
