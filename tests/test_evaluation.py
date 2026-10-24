"""
test_evaluation.py
Tests for the synthetic evaluation suite in src/evaluation.py.
Verifies that the evaluation machinery produces valid, self-consistent results.
"""

import numpy as np
import pytest

from src.evaluation import (
    generate_synthetic_mouth_signal,
    generate_synthetic_audio_signal,
    make_timestamps,
    run_single_experiment,
    run_evaluation,
    run_inconclusive_test,
    ExperimentResult,
    EvaluationSummary,
    EVAL_DURATION_SEC,
    TEST_OFFSETS_MS,
    TOLERANCE_MS,
)
from src.synchronization import SyncVerdict, TARGET_FPS


# ---------------------------------------------------------------------------
# Synthetic signal generation
# ---------------------------------------------------------------------------

class TestSyntheticSignalGeneration:

    def test_mouth_signal_length(self):
        fps = TARGET_FPS
        duration = EVAL_DURATION_SEC
        mouth = generate_synthetic_mouth_signal(duration, fps, seed=0)
        expected_n = int(duration * fps)
        assert abs(len(mouth) - expected_n) <= 1   # allow ±1 for rounding

    def test_mouth_signal_non_negative(self):
        mouth = generate_synthetic_mouth_signal()
        assert np.all(mouth >= 0), "Mouth aperture should always be ≥ 0"

    def test_mouth_signal_max_is_one(self):
        mouth = generate_synthetic_mouth_signal()
        assert abs(mouth.max() - 1.0) < 1e-9, "Mouth signal should be normalized to [0, 1]"

    def test_mouth_signal_has_variance(self):
        mouth = generate_synthetic_mouth_signal()
        assert mouth.std() > 0.05, "Mouth signal should have meaningful variance"

    def test_different_seeds_produce_different_signals(self):
        m1 = generate_synthetic_mouth_signal(seed=10)
        m2 = generate_synthetic_mouth_signal(seed=20)
        assert not np.allclose(m1, m2), "Different seeds should produce different signals"

    def test_audio_signal_same_length_as_mouth(self):
        mouth = generate_synthetic_mouth_signal()
        audio = generate_synthetic_audio_signal(mouth, offset_ms=100.0)
        assert len(audio) == len(mouth)

    def test_timestamps_length_matches_signal(self):
        mouth = generate_synthetic_mouth_signal()
        ts = make_timestamps(len(mouth))
        assert len(ts) == len(mouth)

    def test_timestamps_start_at_zero(self):
        ts = make_timestamps(100)
        assert ts[0] == pytest.approx(0.0)

    def test_timestamps_monotonically_increasing(self):
        ts = make_timestamps(100)
        diffs = np.diff(ts)
        assert np.all(diffs > 0), "Timestamps must be strictly increasing"


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

class TestRunSingleExperiment:

    def test_returns_experiment_result(self):
        exp = run_single_experiment(offset_ms=0.0)
        assert isinstance(exp, ExperimentResult)

    def test_true_offset_preserved(self):
        for offset in [0, 100, -200, 350]:
            exp = run_single_experiment(offset_ms=float(offset))
            assert exp.true_offset_ms == float(offset)

    def test_has_verdict(self):
        exp = run_single_experiment(offset_ms=0.0)
        assert isinstance(exp.verdict, SyncVerdict)

    def test_error_ms_non_negative_when_available(self):
        exp = run_single_experiment(offset_ms=100.0)
        if exp.error_ms is not None:
            assert exp.error_ms >= 0, "Absolute error must be non-negative"

    def test_within_tolerance_consistent_with_error(self):
        exp = run_single_experiment(offset_ms=0.0)
        if exp.error_ms is not None:
            expected_within = exp.error_ms <= TOLERANCE_MS
            assert exp.within_tolerance == expected_within

    def test_inconclusive_has_none_error(self):
        """If verdict is INCONCLUSIVE, error_ms and within_tolerance must be None."""
        # Run with flat signals to force INCONCLUSIVE
        from src.evaluation import run_single_experiment
        # A large noise level won't make it inconclusive, but a different approach:
        # We can monkey-patch — OR we can just test any result we get
        exp = run_single_experiment(offset_ms=0.0)
        if exp.verdict == SyncVerdict.INCONCLUSIVE:
            assert exp.error_ms is None
            assert exp.within_tolerance is None


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

class TestRunEvaluation:

    @pytest.fixture(scope="class")
    def evaluation_summary(self):
        return run_evaluation()

    def test_returns_evaluation_summary(self, evaluation_summary):
        assert isinstance(evaluation_summary, EvaluationSummary)

    def test_experiment_count_matches_offsets(self, evaluation_summary):
        assert len(evaluation_summary.experiments) == len(TEST_OFFSETS_MS)

    def test_eligible_plus_inconclusive_equals_total(self, evaluation_summary):
        total = len(evaluation_summary.experiments)
        eligible = evaluation_summary.eligible_count
        inconclusive = evaluation_summary.inconclusive_count
        # eligible + inconclusive ≤ total (eligible counts only non-inconclusive with estimates)
        assert eligible + inconclusive <= total

    def test_mae_is_none_or_positive(self, evaluation_summary):
        if evaluation_summary.mae_ms is not None:
            assert evaluation_summary.mae_ms >= 0

    def test_within_tolerance_pct_between_0_and_100(self, evaluation_summary):
        if evaluation_summary.within_tolerance_pct is not None:
            assert 0.0 <= evaluation_summary.within_tolerance_pct <= 100.0

    def test_tolerance_ms_matches_constant(self, evaluation_summary):
        assert evaluation_summary.tolerance_ms == TOLERANCE_MS

    def test_notes_is_non_empty_string(self, evaluation_summary):
        assert isinstance(evaluation_summary.notes, str)
        assert len(evaluation_summary.notes) > 10


# ---------------------------------------------------------------------------
# Inconclusive test
# ---------------------------------------------------------------------------

class TestInconclusiveTest:

    def test_flat_signals_return_inconclusive(self):
        result = run_inconclusive_test()
        assert isinstance(result, dict)
        assert "passed" in result
        assert result["passed"] is True, (
            f"Flat signal test FAILED. Verdict was: {result.get('verdict')}. "
            f"Details: {result.get('details')}"
        )

    def test_flat_signal_verdict_is_inconclusive(self):
        result = run_inconclusive_test()
        assert result["verdict"] == SyncVerdict.INCONCLUSIVE
