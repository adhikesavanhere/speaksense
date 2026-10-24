"""
evaluation.py
-------------
Controlled synthetic experiments to validate the synchronization algorithm.

These tests use mathematically generated signals with KNOWN offsets.
They do NOT use real video or audio files.

Results from these experiments are used in the Streamlit "Evaluation" section.
All numbers reported are from actual execution — nothing is fabricated.

Synthetic signal design
-----------------------
- Mouth-aperture signal: a sum of sine waves at typical syllable-rate frequencies
  (2–6 Hz), representing the rhythmic opening and closing of the mouth during speech.
- Audio energy signal: same base signal shifted by a known number of samples,
  then lightly perturbed with Gaussian noise to simulate real-world imperfections.

Note: synthetic tests validate the algorithm's cross-correlation mechanics.
They do not establish performance on every real-world video, which is affected
by face detection quality, audio complexity, and many other factors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.synchronization import estimate_sync, SyncVerdict, TARGET_FPS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_DURATION_SEC = 8.0          # Length of each synthetic signal
EVAL_NOISE_STD = 0.1             # Gaussian noise σ added to audio signal
# Test offsets in milliseconds (positive = audio leads, negative = video leads)
TEST_OFFSETS_MS = [0, 50, 100, -100, 200, -200, 350, -350, 500, -500]
TOLERANCE_MS = 80.0              # Acceptable estimation error for "within tolerance" metric


# ---------------------------------------------------------------------------
# Synthetic signal generation
# ---------------------------------------------------------------------------

def generate_synthetic_mouth_signal(
    duration_sec: float = EVAL_DURATION_SEC,
    fps: float = TARGET_FPS,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic mouth-aperture signal.

    Uses a sum of sine waves at 2, 3, and 5 Hz to simulate realistic
    syllable-rate mouth movements during speech.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fps)
    t = np.linspace(0, duration_sec, n)

    # Combine multiple frequencies for a realistic, non-periodic signal
    signal = (
        0.5 * np.sin(2 * np.pi * 2.0 * t)
        + 0.3 * np.sin(2 * np.pi * 3.5 * t + 0.8)
        + 0.2 * np.sin(2 * np.pi * 5.0 * t + 1.5)
    )
    # Add low-level noise
    signal += rng.normal(0, 0.03, n)
    # Shift to be non-negative (mouth aperture is always ≥ 0)
    signal = signal - signal.min()
    signal = signal / signal.max()  # normalize to [0, 1]
    return signal


def generate_synthetic_audio_signal(
    mouth_signal: np.ndarray,
    offset_ms: float,
    fps: float = TARGET_FPS,
    noise_std: float = EVAL_NOISE_STD,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a synthetic audio energy signal from a given mouth signal.

    The audio signal is created by:
    1. Shifting the mouth signal by `offset_ms` milliseconds using a LINEAR
       shift (zero-filled boundary), not a circular wrap.  This ensures that
       large offsets produce genuinely misaligned signals rather than
       creating spurious correlations through period aliasing.
    2. Adding Gaussian noise.

    A positive offset_ms means the audio signal leads (occurs earlier than) the mouth.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(mouth_signal)
    offset_samples = int(round(offset_ms * fps / 1000.0))

    # Linear (non-circular) shift — fill boundary with zeros
    audio = np.zeros(n, dtype=float)
    if offset_samples == 0:
        audio[:] = mouth_signal
    elif offset_samples > 0:
        # Audio leads by offset_samples: audio[t] = mouth[t + offset_samples]
        # Equivalently: shift mouth left → audio values appear earlier
        src_len = max(0, n - offset_samples)
        audio[:src_len] = mouth_signal[offset_samples: offset_samples + src_len]
        # Tail (last offset_samples positions) stays 0
    else:
        # Negative offset: mouth leads audio by |offset_samples|
        k = -offset_samples  # k > 0
        src_len = max(0, n - k)
        audio[k: k + src_len] = mouth_signal[:src_len]
        # Head (first k positions) stays 0

    # Add noise
    audio = audio + rng.normal(0, noise_std, n)
    audio = np.clip(audio, 0, None)  # energy cannot be negative
    return audio


def make_timestamps(n: int, fps: float = TARGET_FPS) -> np.ndarray:
    """Generate a uniform timestamp array of length n at fps."""
    return np.linspace(0.0, (n - 1) / fps, n)


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Result of one synthetic experiment."""
    true_offset_ms: float
    estimated_offset_ms: Optional[float]
    error_ms: Optional[float]          # |estimated - true|
    within_tolerance: Optional[bool]   # error_ms <= TOLERANCE_MS
    verdict: SyncVerdict
    peak_quality_ratio: Optional[float]
    note: str = ""


def run_single_experiment(
    offset_ms: float,
    duration_sec: float = EVAL_DURATION_SEC,
    fps: float = TARGET_FPS,
    noise_std: float = EVAL_NOISE_STD,
    seed: int = 42,
) -> ExperimentResult:
    """
    Run the synchronization algorithm on a synthetic pair with known offset.
    """
    mouth = generate_synthetic_mouth_signal(duration_sec, fps, seed)
    audio = generate_synthetic_audio_signal(mouth, offset_ms, fps, noise_std, seed)
    timestamps = make_timestamps(len(mouth), fps)

    result = estimate_sync(
        mouth_timestamps=timestamps,
        mouth_apertures=mouth,
        audio_timestamps=timestamps,
        audio_energy=audio,
    )

    if result.estimated_offset_ms is None:
        return ExperimentResult(
            true_offset_ms=offset_ms,
            estimated_offset_ms=None,
            error_ms=None,
            within_tolerance=None,
            verdict=result.verdict,
            peak_quality_ratio=result.peak_quality_ratio,
            note=result.reason,
        )

    error = abs(result.estimated_offset_ms - offset_ms)
    return ExperimentResult(
        true_offset_ms=offset_ms,
        estimated_offset_ms=result.estimated_offset_ms,
        error_ms=error,
        within_tolerance=(error <= TOLERANCE_MS),
        verdict=result.verdict,
        peak_quality_ratio=result.peak_quality_ratio,
    )


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics."""
    experiments: List[ExperimentResult]
    # Only consider experiments that produced a numeric estimate
    eligible_count: int
    inconclusive_count: int
    mae_ms: Optional[float]              # Mean absolute error (eligible only)
    within_tolerance_count: int
    within_tolerance_pct: Optional[float]
    tolerance_ms: float
    notes: str


def run_evaluation(
    offsets_ms: List[float] = TEST_OFFSETS_MS,
    duration_sec: float = EVAL_DURATION_SEC,
    noise_std: float = EVAL_NOISE_STD,
) -> EvaluationSummary:
    """
    Run the full evaluation suite on synthetic signals.
    Returns an EvaluationSummary with actual computed metrics.
    """
    experiments: List[ExperimentResult] = []
    for i, offset_ms in enumerate(offsets_ms):
        exp = run_single_experiment(
            offset_ms=offset_ms,
            duration_sec=duration_sec,
            noise_std=noise_std,
            seed=42 + i,
        )
        experiments.append(exp)

    eligible = [e for e in experiments if e.error_ms is not None]
    inconclusive = [e for e in experiments if e.verdict == SyncVerdict.INCONCLUSIVE]

    mae = float(np.mean([e.error_ms for e in eligible])) if eligible else None
    within = [e for e in eligible if e.within_tolerance]
    within_pct = (len(within) / len(eligible) * 100.0) if eligible else None

    notes = (
        f"Synthetic test results on {len(offsets_ms)} experiments with "
        f"signal duration {duration_sec:.0f}s and noise σ={noise_std}. "
        f"{len(inconclusive)} inconclusive result(s). "
        f"These results reflect algorithm performance on synthetic signals only; "
        f"real-world video performance may differ."
    )

    return EvaluationSummary(
        experiments=experiments,
        eligible_count=len(eligible),
        inconclusive_count=len(inconclusive),
        mae_ms=mae,
        within_tolerance_count=len(within),
        within_tolerance_pct=within_pct,
        tolerance_ms=TOLERANCE_MS,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Inconclusive-signal test
# ---------------------------------------------------------------------------

def run_inconclusive_test() -> Dict[str, object]:
    """
    Verify that flat/silent signals return INCONCLUSIVE verdict.
    Returns a dict with 'passed' (bool) and 'details' (str).
    """
    fps = TARGET_FPS
    n = int(EVAL_DURATION_SEC * fps)
    timestamps = make_timestamps(n, fps)

    # Completely flat signals
    flat = np.zeros(n)
    result = estimate_sync(flat, flat, timestamps, timestamps)

    passed = result.verdict == SyncVerdict.INCONCLUSIVE
    return {
        "passed": passed,
        "verdict": result.verdict,
        "details": f"Flat signal test → verdict: {result.verdict}. Expected: INCONCLUSIVE.",
    }
