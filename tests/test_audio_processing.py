"""
test_audio_processing.py
Tests for audio extraction helpers and feature computation.
Tests do not require a real video; we test helpers independently.
"""

import os
import numpy as np
import pytest
import tempfile

# We test the compute_audio_features path using a real WAV if soundfile is available,
# or we test the error handling paths for missing FFmpeg / invalid files.

from src.audio_processing import (
    check_ffmpeg,
    compute_audio_features,
    AudioFeatures,
    DEFAULT_SAMPLE_RATE,
    MIN_AUDIO_DURATION,
    ENERGY_FRAME_LENGTH,
    ENERGY_HOP_LENGTH,
)


class TestFfmpegCheck:
    def test_check_ffmpeg_returns_bool(self):
        result = check_ffmpeg()
        assert isinstance(result, bool)

    def test_check_ffmpeg_does_not_raise(self):
        # Should return True or False, never raise
        try:
            check_ffmpeg()
        except Exception as exc:
            pytest.fail(f"check_ffmpeg() raised an unexpected exception: {exc}")


class TestComputeAudioFeatures:
    """
    These tests write a minimal synthetic WAV file and verify that
    compute_audio_features returns the expected structure.
    Requires soundfile or scipy.io.wavfile to create the test WAV.
    """

    def _write_sine_wav(self, filepath: str, duration_sec: float = 3.0, freq: float = 440.0):
        """Write a simple sine-wave WAV for testing."""
        import struct
        import wave

        sr = DEFAULT_SAMPLE_RATE
        n_samples = int(sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

    def test_valid_wav_returns_features(self, tmp_path):
        wav_file = str(tmp_path / "test_sine.wav")
        self._write_sine_wav(wav_file, duration_sec=3.0)

        features, messages = compute_audio_features(wav_file)
        assert features is not None, f"Expected features, got messages: {messages}"
        assert isinstance(features, AudioFeatures)

    def test_energy_array_non_negative(self, tmp_path):
        wav_file = str(tmp_path / "test_energy.wav")
        self._write_sine_wav(wav_file, duration_sec=3.0)
        features, _ = compute_audio_features(wav_file)
        assert features is not None
        assert np.all(features.energy >= 0), "Energy values must be non-negative"

    def test_energy_timestamps_match_energy_length(self, tmp_path):
        wav_file = str(tmp_path / "test_ts.wav")
        self._write_sine_wav(wav_file, duration_sec=4.0)
        features, _ = compute_audio_features(wav_file)
        assert features is not None
        assert len(features.energy) == len(features.energy_timestamps)

    def test_voice_activity_is_boolean_array(self, tmp_path):
        wav_file = str(tmp_path / "test_va.wav")
        self._write_sine_wav(wav_file, duration_sec=3.0)
        features, _ = compute_audio_features(wav_file)
        assert features is not None
        assert features.voice_activity.dtype == bool

    def test_duration_approximately_correct(self, tmp_path):
        duration = 5.0
        wav_file = str(tmp_path / "test_dur.wav")
        self._write_sine_wav(wav_file, duration_sec=duration)
        features, _ = compute_audio_features(wav_file)
        assert features is not None
        assert abs(features.duration_sec - duration) < 0.5, \
            f"Expected ~{duration}s, got {features.duration_sec:.2f}s"

    def test_too_short_audio_returns_error(self, tmp_path):
        wav_file = str(tmp_path / "test_short.wav")
        # Write 0.1 seconds — shorter than MIN_AUDIO_DURATION
        self._write_sine_wav(wav_file, duration_sec=0.1)
        features, messages = compute_audio_features(wav_file)
        assert features is None, "Expected None for too-short audio"
        assert len(messages) > 0

    def test_nonexistent_wav_returns_error(self):
        features, messages = compute_audio_features("/nonexistent/audio.wav")
        assert features is None
        assert len(messages) > 0

    def test_silent_audio_has_low_energy(self, tmp_path):
        """A WAV of near-silence should have very low energy values."""
        import wave

        wav_file = str(tmp_path / "test_silent.wav")
        sr = DEFAULT_SAMPLE_RATE
        n_samples = int(sr * 3.0)
        # All-zero samples = silence
        silence = np.zeros(n_samples, dtype=np.int16)

        with wave.open(wav_file, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(silence.tobytes())

        features, _ = compute_audio_features(wav_file)
        assert features is not None
        # Energy should be extremely small
        assert features.energy.mean() < 1e-3, \
            f"Expected near-zero energy for silence, got {features.energy.mean()}"
