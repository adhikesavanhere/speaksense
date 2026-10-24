"""
conftest.py — Shared pytest fixtures for SpeakSense tests.
"""

import numpy as np
import pytest

from src.synchronization import TARGET_FPS
from src.evaluation import (
    generate_synthetic_mouth_signal,
    generate_synthetic_audio_signal,
    make_timestamps,
    EVAL_DURATION_SEC,
)


@pytest.fixture
def default_duration():
    return EVAL_DURATION_SEC


@pytest.fixture
def default_fps():
    return TARGET_FPS


@pytest.fixture
def synthetic_zero_offset(default_duration, default_fps):
    """Mouth and audio signals with ZERO offset and timestamps."""
    mouth = generate_synthetic_mouth_signal(default_duration, default_fps, seed=1)
    audio = generate_synthetic_audio_signal(mouth, offset_ms=0.0, fps=default_fps, noise_std=0.05, seed=2)
    ts = make_timestamps(len(mouth), default_fps)
    return ts, mouth, audio


@pytest.fixture
def synthetic_positive_offset(default_duration, default_fps):
    """Signals where audio leads video by +250 ms."""
    mouth = generate_synthetic_mouth_signal(default_duration, default_fps, seed=3)
    audio = generate_synthetic_audio_signal(mouth, offset_ms=250.0, fps=default_fps, noise_std=0.08, seed=4)
    ts = make_timestamps(len(mouth), default_fps)
    return ts, mouth, audio, 250.0


@pytest.fixture
def synthetic_negative_offset(default_duration, default_fps):
    """Signals where video leads audio by -250 ms (offset = -250)."""
    mouth = generate_synthetic_mouth_signal(default_duration, default_fps, seed=5)
    audio = generate_synthetic_audio_signal(mouth, offset_ms=-250.0, fps=default_fps, noise_std=0.08, seed=6)
    ts = make_timestamps(len(mouth), default_fps)
    return ts, mouth, audio, -250.0


@pytest.fixture
def flat_signals(default_duration, default_fps):
    """Completely flat (zero-variance) signals."""
    n = int(default_duration * default_fps)
    ts = make_timestamps(n, default_fps)
    flat = np.zeros(n)
    return ts, flat
