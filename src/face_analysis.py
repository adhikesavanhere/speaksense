"""
face_analysis.py
----------------
Detects faces and extracts mouth-aperture measurements from video frames
using MediaPipe FaceMesh.

Mouth aperture = normalized vertical distance between upper and lower lips,
divided by face height (inter-eye distance) to be scale-invariant.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    _mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    _mp_face_mesh = None

import cv2

# Import our own type
from src.video_processing import FrameSample


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices (478-point model)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------

# Upper lip top center
UPPER_LIP_TOP = 13
# Lower lip bottom center
LOWER_LIP_BOTTOM = 14

# Alternative sets for robustness (outer mouth corners)
MOUTH_UPPER_INNER = [13, 312, 311, 310, 415, 308]   # upper inner lip
MOUTH_LOWER_INNER = [14, 317, 402, 318, 324, 78]    # lower inner lip

# Inter-eye distance landmarks (for normalization)
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FrameAnalysis:
    """Analysis result for one video frame."""
    timestamp_sec: float
    face_detected: bool
    mouth_aperture: Optional[float]   # Normalized [0, 1]
    face_confidence: float = 1.0
    warning: str = ""


@dataclass
class FaceAnalysisResult:
    """Aggregated result from analysing all frames."""
    frame_analyses: List[FrameAnalysis]
    # Filtered arrays (only frames where face was detected)
    timestamps: np.ndarray           # seconds
    apertures: np.ndarray            # normalized mouth openings
    face_detection_rate: float       # fraction of frames with a detected face
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _compute_mouth_aperture(landmarks, img_w: int, img_h: int) -> Optional[float]:
    """
    Compute normalized mouth aperture from MediaPipe NormalizedLandmarkList.

    Returns a float in roughly [0, 1], or None if landmarks are missing.
    """
    try:
        # Upper lip (several points) → mean y
        upper_y = np.mean([
            landmarks.landmark[i].y for i in MOUTH_UPPER_INNER
        ])
        # Lower lip (several points) → mean y
        lower_y = np.mean([
            landmarks.landmark[i].y for i in MOUTH_LOWER_INNER
        ])

        # Mouth opening in normalized image coordinates
        raw_aperture = abs(lower_y - upper_y)

        # Normalize by inter-eye distance (scale invariance)
        left_eye_x = landmarks.landmark[LEFT_EYE_INNER].x
        right_eye_x = landmarks.landmark[RIGHT_EYE_INNER].x
        inter_eye = abs(right_eye_x - left_eye_x)

        if inter_eye < 1e-6:
            return None  # degenerate face geometry

        aperture = raw_aperture / inter_eye
        return float(np.clip(aperture, 0.0, 2.0))  # clip extreme values

    except (IndexError, AttributeError):
        return None


def analyze_frames(
    frames: List[FrameSample],
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_num_faces: int = 1,
) -> FaceAnalysisResult:
    """
    Run MediaPipe FaceMesh on each sampled frame and compute mouth aperture.

    Parameters
    ----------
    frames : list of FrameSample
    min_detection_confidence : float
        MediaPipe detection threshold (0–1).
    min_tracking_confidence : float
        MediaPipe tracking threshold (0–1).
    max_num_faces : int
        If > 1, uses the face closest to the frame centre.

    Returns
    -------
    FaceAnalysisResult
    """
    errors: List[str] = []
    warnings: List[str] = []
    frame_analyses: List[FrameAnalysis] = []

    if not MEDIAPIPE_AVAILABLE:
        return FaceAnalysisResult(
            frame_analyses=[],
            timestamps=np.array([]),
            apertures=np.array([]),
            face_detection_rate=0.0,
            errors=["MediaPipe is not installed. Face analysis unavailable."],
        )

    if not frames:
        return FaceAnalysisResult(
            frame_analyses=[],
            timestamps=np.array([]),
            apertures=np.array([]),
            face_detection_rate=0.0,
            errors=["No frames provided for face analysis."],
        )

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,           # treat each frame independently
        max_num_faces=max(1, max_num_faces),
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:

        for fs in frames:
            frame_bgr = fs.frame
            img_h, img_w = frame_bgr.shape[:2]

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                frame_analyses.append(FrameAnalysis(
                    timestamp_sec=fs.timestamp_sec,
                    face_detected=False,
                    mouth_aperture=None,
                    warning="No face detected",
                ))
                continue

            # If multiple faces, pick the one whose nose tip is closest to centre
            if len(results.multi_face_landmarks) > 1:
                cx, cy = img_w / 2, img_h / 2
                best_lm = None
                best_dist = float("inf")
                for lm_set in results.multi_face_landmarks:
                    nose = lm_set.landmark[1]  # nose tip index
                    dx = nose.x * img_w - cx
                    dy = nose.y * img_h - cy
                    dist = dx * dx + dy * dy
                    if dist < best_dist:
                        best_dist = dist
                        best_lm = lm_set
                landmarks = best_lm
            else:
                landmarks = results.multi_face_landmarks[0]

            aperture = _compute_mouth_aperture(landmarks, img_w, img_h)
            frame_analyses.append(FrameAnalysis(
                timestamp_sec=fs.timestamp_sec,
                face_detected=True,
                mouth_aperture=aperture,
            ))

    # Aggregate
    detected = [fa for fa in frame_analyses if fa.face_detected and fa.mouth_aperture is not None]
    detection_rate = len(detected) / len(frame_analyses) if frame_analyses else 0.0

    if detection_rate < 0.3:
        warnings.append(
            f"Face detected in only {detection_rate * 100:.0f}% of sampled frames. "
            "Results may be unreliable — ensure the face is clearly visible throughout."
        )
    elif detection_rate < 0.6:
        warnings.append(
            f"Face detected in {detection_rate * 100:.0f}% of sampled frames. "
            "Some frames may have been missed."
        )

    timestamps = np.array([fa.timestamp_sec for fa in detected])
    apertures = np.array([fa.mouth_aperture for fa in detected])

    return FaceAnalysisResult(
        frame_analyses=frame_analyses,
        timestamps=timestamps,
        apertures=apertures,
        face_detection_rate=detection_rate,
        errors=errors,
        warnings=warnings,
    )
