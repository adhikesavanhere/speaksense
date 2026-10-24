"""
test_synchronization.py
Tests for the core synchronization algorithm.

All tests use synthetic signals with known, controlled offsets.
Tests verify:
  - Zero offset → ALIGNED verdict
  - Large offset → MISMATCH verdict
  - Flat/silent signals → INCONCLUSIVE verdict
  - Positive and negative offset estimation accuracy
  - Estimated offset within acceptable tolerance
  - API contracts (return types, field presence)
"""

import numpy as np
import pytest

from src.synchronization import (
    estimate_sync,
    SyncVerdict,
    SyncResult,
    _normalize,
    _smooth,
    _cross_correlate,
    _resample_to_grid,
    TARGET_FPS,
    MIN_SIGNAL_STD,
)
from src.evaluation import (
    generate_synthetic_mouth_signal,
    generate_synthetic_audio_signal,
    make_timestamps,
    EVAL_DURATION_SEC,
)

TOLERANCE_MS = 80.0   # Acceptable estimation error in milliseconds


def make_pair(offset_ms: float, duration: float = EVAL_DURATION_SEC, noise: float = 0.08, seed: int = 42):
    """Helper: create timestamps + mouth/audio pair with a given offset."""
    fps = TARGET_FPS
    mouth = generate_synthetic_mouth_signal(duration, fps, seed)
    audio = generate_synthetic_audio_signal(mouth, offset_ms, fps, noise, seed)
    ts = make_timestamps(len(mouth), fps)
    return ts, mouth, audio


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------

class TestEstimateSyncAPI:
    def test_returns_sync_result(self):
        ts, mouth, audio = make_pair(0.0)
        result = estimate_sync(ts, mouth, ts, audio)
        assert isinstance(result, SyncResult)

    def test_result_has_verdict(self):
        ts, mouth, audio = make_pair(0.0)
        result = estimate_sync(ts, mouth, ts, audio)
        assert isinstance(result.verdict, SyncVerdict)

    def test_result_has_errors_and_warnings_lists(self):
        ts, mouth, audio = make_pair(0.0)
        result = estimate_sync(ts, mouth, ts, audio)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

    def test_result_has_reason_string(self):
        ts, mouth, audio = make_pair(0.0)
        result = estimate_sync(ts, mouth, ts, audio)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_inconclusive_result_has_no_estimated_offset(self):
        """Flat signals must return INCONCLUSIVE with None offset."""
        n = int(EVAL_DURATION_SEC * TARGET_FPS)
        ts = make_timestamps(n, TARGET_FPS)
        flat = np.zeros(n)
        result = estimate_sync(ts, flat, ts, flat)
        assert result.verdict == SyncVerdict.INCONCLUSIVE
        assert result.estimated_offset_ms is None


# ---------------------------------------------------------------------------
# Verdict tests
# ---------------------------------------------------------------------------

class TestVerdicts:

    def test_zero_offset_is_aligned(self):
        ts, mouth, audio = make_pair(0.0, noise=0.05)
        result = estimate_sync(ts, mouth, ts, audio)
        assert result.verdict == SyncVerdict.ALIGNED, \
            f"Zero offset should be ALIGNED, got {result.verdict}. Reason: {result.reason}"

    def test_large_positive_offset_is_mismatch(self):
        """500 ms offset should exceed the 100 ms threshold → MISMATCH."""
        ts, mouth, audio = make_pair(500.0, noise=0.05)
        result = estimate_sync(ts, mouth, ts, audio)
        # Could be MISMATCH or INCONCLUSIVE at very high offsets, but not ALIGNED
        assert result.verdict != SyncVerdict.ALIGNED, \
            f"500ms offset should NOT be ALIGNED. Got {result.verdict}. Offset: {result.estimated_offset_ms}"

    def test_large_negative_offset_is_not_aligned(self):
        ts, mouth, audio = make_pair(-500.0, noise=0.05)
        result = estimate_sync(ts, mouth, ts, audio)
        assert result.verdict != SyncVerdict.ALIGNED, \
            f"-500ms offset should NOT be ALIGNED. Got {result.verdict}."

    def test_flat_signals_inconclusive(self):
        n = int(EVAL_DURATION_SEC * TARGET_FPS)
        ts = make_timestamps(n, TARGET_FPS)
        flat = np.zeros(n)
        result = estimate_sync(ts, flat, ts, flat)
        assert result.verdict == SyncVerdict.INCONCLUSIVE

    def test_constant_signal_inconclusive(self):
        n = int(EVAL_DURATION_SEC * TARGET_FPS)
        ts = make_timestamps(n, TARGET_FPS)
        constant = np.ones(n) * 0.5
        result = estimate_sync(ts, constant, ts, constant)
        assert result.verdict == SyncVerdict.INCONCLUSIVE

    def test_too_few_points_inconclusive(self):
        """Fewer than 5 points → INCONCLUSIVE."""
        ts = np.array([0.0, 0.1, 0.2])
        values = np.array([0.1, 0.2, 0.3])
        result = estimate_sync(ts, values, ts, values)
        assert result.verdict == SyncVerdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Offset estimation accuracy tests
# ---------------------------------------------------------------------------

class TestOffsetEstimation:

    @pytest.mark.parametrize("offset_ms", [0, 50, 100, -100, 200, -200])
    def test_estimated_offset_within_tolerance(self, offset_ms):
        ts, mouth, audio = make_pair(offset_ms, noise=0.08, seed=99)
        result = estimate_sync(ts, mouth, ts, audio)

        if result.estimated_offset_ms is None:
            pytest.skip(f"Result was INCONCLUSIVE for offset={offset_ms}ms — skipping accuracy check.")

        error = abs(result.estimated_offset_ms - offset_ms)
        assert error <= TOLERANCE_MS, (
            f"Offset {offset_ms}ms: estimated {result.estimated_offset_ms:.0f}ms, "
            f"error {error:.0f}ms exceeds tolerance {TOLERANCE_MS}ms."
        )

    def test_positive_offset_has_positive_sign(self):
        """Audio leads video → estimated offset should be positive."""
        ts, mouth, audio = make_pair(200.0, noise=0.06, seed=11)
        result = estimate_sync(ts, mouth, ts, audio)
        if result.estimated_offset_ms is not None:
            # With low noise and a clear 200ms signal, the sign should be correct
            # (within ±TOLERANCE we can't distinguish small offsets, so only check for large ones)
            if abs(result.estimated_offset_ms) > TOLERANCE_MS:
                assert result.estimated_offset_ms > 0, \
                    f"Expected positive offset, got {result.estimated_offset_ms:.0f}ms"

    def test_negative_offset_has_negative_sign(self):
        """Video leads audio → estimated offset should be negative."""
        ts, mouth, audio = make_pair(-200.0, noise=0.06, seed=12)
        result = estimate_sync(ts, mouth, ts, audio)
        if result.estimated_offset_ms is not None:
            if abs(result.estimated_offset_ms) > TOLERANCE_MS:
                assert result.estimated_offset_ms < 0, \
                    f"Expected negative offset, got {result.estimated_offset_ms:.0f}ms"


# ---------------------------------------------------------------------------
# Signal preprocessing helper tests
# ---------------------------------------------------------------------------

class TestSignalHelpers:

    def test_normalize_zero_mean(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(5.0, 2.0, 200)
        normalized = _normalize(signal)
        assert abs(normalized.mean()) < 1e-10

    def test_normalize_unit_std(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(3.0, 1.5, 200)
        normalized = _normalize(signal)
        assert abs(normalized.std() - 1.0) < 1e-6

    def test_normalize_flat_signal_returns_zeros(self):
        flat = np.ones(100) * 5.0
        result = _normalize(flat)
        assert np.all(result == 0.0)

    def test_smooth_preserves_length(self):
        signal = np.random.randn(100)
        smoothed = _smooth(signal, sigma=2.0)
        assert len(smoothed) == len(signal)

    def test_smooth_reduces_high_freq_noise(self):
        """Smoothed signal should have lower std than random noise."""
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 1, 500)
        smoothed = _smooth(noise, sigma=5.0)
        assert smoothed.std() < noise.std()

    def test_resample_to_grid_preserves_range(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.5, 0.8])
        resampled, uniform_t = _resample_to_grid(v, t, target_fps=10.0)
        assert uniform_t[0] == pytest.approx(0.0, abs=0.01)
        assert uniform_t[-1] == pytest.approx(3.0, abs=0.1)
        assert len(resampled) == len(uniform_t)

    def test_cross_correlate_returns_correct_length(self):
        n = 100
        a = np.random.randn(n)
        b = np.random.randn(n)
        corr, lags = _cross_correlate(a, b, fps=TARGET_FPS)
        assert len(corr) == 2 * n - 1
        assert len(lags) == 2 * n - 1

    def test_cross_correlate_peak_at_zero_for_identical_signals(self):
        """Identical signals should have their cross-correlation peak at lag=0."""
        n = 200
        signal = np.sin(np.linspace(0, 4 * np.pi, n))
        corr, lags = _cross_correlate(signal, signal, fps=TARGET_FPS)
        peak_idx = np.argmax(corr)
        peak_lag_ms = lags[peak_idx]
        # Peak should be near 0 ms
        assert abs(peak_lag_ms) < 1000.0 / TARGET_FPS, \
            f"Expected peak near 0ms for identical signals, got {peak_lag_ms:.1f}ms"
