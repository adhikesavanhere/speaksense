"""
audio_processing.py
-------------------
Extracts audio from a video file using FFmpeg, then computes audio features
(short-time energy envelope) using Librosa.

All heavy processing is independent of Streamlit.
"""

import os
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 22050       # Hz — librosa default
MIN_AUDIO_DURATION = 0.5          # seconds
ENERGY_FRAME_LENGTH = 2048        # samples per energy frame
ENERGY_HOP_LENGTH = 512           # hop between energy frames


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    """Container for computed audio features."""
    sample_rate: int
    duration_sec: float
    # Short-time energy envelope (one value per hop frame)
    energy: np.ndarray
    # Timestamps for each energy frame (seconds)
    energy_timestamps: np.ndarray
    # Boolean voice-activity array (same length as energy)
    voice_activity: np.ndarray


@dataclass
class AudioProcessingResult:
    """Return value from process_audio()."""
    features: Optional[AudioFeatures]
    audio_extracted: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FFmpeg audio extraction
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_audio_to_wav(video_filepath: str, output_wav: str) -> Tuple[bool, str]:
    """
    Use FFmpeg to extract the first audio stream from *video_filepath* and
    write a mono WAV at DEFAULT_SAMPLE_RATE to *output_wav*.

    Returns (success: bool, message: str).
    """
    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output without asking
        "-i", video_filepath,
        "-vn",                         # no video in output
        "-ac", "1",                    # mono
        "-ar", str(DEFAULT_SAMPLE_RATE),
        "-acodec", "pcm_s16le",        # standard WAV encoding
        output_wav,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            # Check for common "no audio" messages
            if "no audio" in stderr.lower() or "stream 0" not in stderr.lower():
                return False, "No audio stream found in the video."
            return False, f"FFmpeg error (code {result.returncode}): {stderr[-500:]}"
        if not os.path.exists(output_wav) or os.path.getsize(output_wav) < 100:
            return False, "FFmpeg ran but produced an empty or missing audio file."
        return True, "Audio extracted successfully."
    except FileNotFoundError:
        return False, (
            "FFmpeg not found. Please install FFmpeg and ensure it is on your PATH.\n"
            "Windows: https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timed out during audio extraction."


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_audio_features(wav_filepath: str) -> Tuple[Optional[AudioFeatures], List[str]]:
    """
    Load a WAV file with Librosa and compute short-time energy + voice activity.
    Returns (AudioFeatures or None, list of error/warning strings).
    """
    messages: List[str] = []

    if not LIBROSA_AVAILABLE:
        messages.append("Librosa is not installed. Cannot compute audio features.")
        return None, messages

    try:
        y, sr = librosa.load(wav_filepath, sr=DEFAULT_SAMPLE_RATE, mono=True)
    except Exception as exc:
        messages.append(f"Failed to load audio file: {exc}")
        return None, messages

    duration_sec = len(y) / sr
    if duration_sec < MIN_AUDIO_DURATION:
        messages.append(
            f"Audio duration ({duration_sec:.2f}s) is too short "
            f"(minimum {MIN_AUDIO_DURATION}s)."
        )
        return None, messages

    # Short-time energy: sum of squared samples in each frame
    energy_frames = librosa.feature.rms(
        y=y,
        frame_length=ENERGY_FRAME_LENGTH,
        hop_length=ENERGY_HOP_LENGTH,
    ).flatten()  # shape: (n_frames,)

    # Timestamps for each frame
    n_frames = len(energy_frames)
    energy_timestamps = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=ENERGY_HOP_LENGTH,
    )

    # Simple voice-activity detection: energy > mean + 0.5*std
    threshold = energy_frames.mean() + 0.5 * energy_frames.std()
    voice_activity = energy_frames > threshold

    return AudioFeatures(
        sample_rate=sr,
        duration_sec=duration_sec,
        energy=energy_frames,
        energy_timestamps=energy_timestamps,
        voice_activity=voice_activity,
    ), messages


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def process_audio(video_filepath: str) -> AudioProcessingResult:
    """
    Full audio processing pipeline for a given video file:
      1. Check FFmpeg availability
      2. Extract audio to a temporary WAV
      3. Compute audio features
      4. Clean up temporary file

    Returns AudioProcessingResult (check .errors before using .features).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not check_ffmpeg():
        return AudioProcessingResult(
            features=None,
            audio_extracted=False,
            errors=["FFmpeg is not installed or not on PATH. Audio extraction is unavailable."],
        )

    # Create a temporary WAV file
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="speaksense_audio_")
    os.close(tmp_fd)  # close file descriptor; FFmpeg will overwrite the file

    try:
        success, message = extract_audio_to_wav(video_filepath, tmp_wav)
        if not success:
            return AudioProcessingResult(
                features=None,
                audio_extracted=False,
                errors=[message],
            )

        features, feat_messages = compute_audio_features(tmp_wav)
        for msg in feat_messages:
            if "too short" in msg or "Failed" in msg or "not installed" in msg:
                errors.append(msg)
            else:
                warnings.append(msg)

        if features is None:
            return AudioProcessingResult(
                features=None,
                audio_extracted=True,
                errors=errors,
                warnings=warnings,
            )

        return AudioProcessingResult(
            features=features,
            audio_extracted=True,
            errors=errors,
            warnings=warnings,
        )

    finally:
        # Always clean up the temporary WAV
        try:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except OSError:
            pass
