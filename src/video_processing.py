"""
video_processing.py
-------------------
Handles video file validation, metadata extraction, and frame sampling.
Keeps processing logic independent of Streamlit so it can be unit-tested.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 500
DEFAULT_SAMPLE_FPS = 10          # frames per second to sample
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 300.0         # 5 minutes upper limit
MIN_RESOLUTION = (64, 64)        # width × height


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Structured container for video metadata."""
    filepath: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    file_size_mb: float
    codec: str = ""


@dataclass
class FrameSample:
    """One sampled video frame with its timestamp."""
    frame: np.ndarray       # BGR uint8 array (H, W, 3)
    timestamp_sec: float


@dataclass
class VideoProcessingResult:
    """Return value of process_video()."""
    metadata: VideoMetadata
    frames: List[FrameSample]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_video_file(filepath: str) -> List[str]:
    """
    Validate that the file exists, has a supported extension, and is within
    allowed size limits.  Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []

    if not os.path.exists(filepath):
        errors.append(f"File not found: {filepath}")
        return errors

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        errors.append(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File size {size_mb:.1f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB."
        )

    return errors


def validate_video_content(cap: cv2.VideoCapture) -> Tuple[List[str], List[str]]:
    """
    Validate an already-opened cv2.VideoCapture object.
    Returns (errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not cap.isOpened():
        errors.append("Could not open video file (it may be corrupt or unreadable).")
        return errors, warnings

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        errors.append(
            f"Video resolution {width}×{height} is too small "
            f"(minimum {MIN_RESOLUTION[0]}×{MIN_RESOLUTION[1]})."
        )

    if fps <= 0:
        warnings.append(
            "FPS metadata is missing or zero; frame timing will be estimated."
        )
        fps = 25.0  # fallback

    duration_sec = frame_count / fps if fps > 0 else 0.0
    if frame_count <= 0:
        # Some containers don't report frame count; we won't error on this alone
        warnings.append(
            "Frame count not available in metadata; duration estimation may be inaccurate."
        )

    if duration_sec < MIN_DURATION_SEC and frame_count > 0:
        errors.append(
            f"Video duration ({duration_sec:.2f}s) is too short "
            f"(minimum {MIN_DURATION_SEC}s required)."
        )

    if duration_sec > MAX_DURATION_SEC:
        warnings.append(
            f"Video duration ({duration_sec:.0f}s) exceeds recommended maximum "
            f"({MAX_DURATION_SEC:.0f}s). Processing may be slow."
        )

    return errors, warnings


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(filepath: str) -> Optional[VideoMetadata]:
    """
    Open the video file and return a VideoMetadata object, or None on failure.
    """
    if not os.path.exists(filepath):
        return None

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        cap.release()
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
    cap.release()

    duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    return VideoMetadata(
        filepath=filepath,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        file_size_mb=file_size_mb,
        codec=codec.strip("\x00"),
    )


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(
    filepath: str,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = 600,
) -> Tuple[List[FrameSample], List[str]]:
    """
    Sample frames from a video at `sample_fps` frames per second.
    Returns (list of FrameSample, list of warning strings).

    Keeps memory usage bounded by `max_frames`.
    """
    warnings: List[str] = []
    frames: List[FrameSample] = []

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return frames, ["Could not open video for frame sampling."]

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many native frames to skip between each sample
    frame_interval = max(1, int(round(native_fps / sample_fps)))
    estimated_samples = (total_frames // frame_interval) if total_frames > 0 else 0

    if estimated_samples > max_frames:
        # Widen the interval to stay within max_frames
        frame_interval = max(1, total_frames // max_frames)
        warnings.append(
            f"Video is long; sampling interval widened to every {frame_interval} frames "
            f"to stay within {max_frames}-frame budget."
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / native_fps
            frames.append(FrameSample(frame=frame.copy(), timestamp_sec=timestamp_sec))

            if len(frames) >= max_frames:
                warnings.append(
                    f"Reached frame budget ({max_frames}); remaining video not sampled."
                )
                break

        frame_idx += 1

    cap.release()
    return frames, warnings


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def process_video(
    filepath: str,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> VideoProcessingResult:
    """
    Full video processing pipeline:
      1. Validate file
      2. Extract metadata
      3. Validate content
      4. Sample frames

    Returns a VideoProcessingResult (check .errors before using the data).
    """
    all_errors: List[str] = []
    all_warnings: List[str] = []

    # Step 1 — file-level validation
    file_errors = validate_video_file(filepath)
    if file_errors:
        return VideoProcessingResult(
            metadata=VideoMetadata(
                filepath=filepath, width=0, height=0, fps=0,
                frame_count=0, duration_sec=0, file_size_mb=0,
            ),
            frames=[],
            errors=file_errors,
        )

    # Step 2 — metadata
    meta = extract_metadata(filepath)
    if meta is None:
        return VideoProcessingResult(
            metadata=VideoMetadata(
                filepath=filepath, width=0, height=0, fps=0,
                frame_count=0, duration_sec=0, file_size_mb=0,
            ),
            frames=[],
            errors=["Failed to read video metadata."],
        )

    # Step 3 — content validation
    cap = cv2.VideoCapture(filepath)
    content_errors, content_warnings = validate_video_content(cap)
    cap.release()
    all_errors.extend(content_errors)
    all_warnings.extend(content_warnings)

    if all_errors:
        return VideoProcessingResult(
            metadata=meta, frames=[], errors=all_errors, warnings=all_warnings
        )

    # Step 4 — sample frames
    frames, frame_warnings = sample_frames(filepath, sample_fps=sample_fps)
    all_warnings.extend(frame_warnings)

    if not frames:
        all_errors.append("No frames could be sampled from the video.")

    return VideoProcessingResult(
        metadata=meta,
        frames=frames,
        errors=all_errors,
        warnings=all_warnings,
    )
