# Architecture — SpeakSense

## Overview

SpeakSense is a single-process Python application with a Streamlit front-end.
All processing logic is separated from the UI layer so that modules can be
tested independently with pytest.

---

## Project Structure

```
speaksense/
├── app.py                    # Streamlit UI — entry point
├── requirements.txt
├── packages.txt              # Streamlit Cloud: ffmpeg
├── .streamlit/config.toml    # Upload size, theme
├── .gitignore
├── README.md
│
├── src/                      # Processing library (no Streamlit dependencies)
│   ├── __init__.py
│   ├── video_processing.py   # Video I/O and frame sampling
│   ├── audio_processing.py   # Audio extraction and STE features
│   ├── face_analysis.py      # MediaPipe FaceMesh, mouth aperture
│   ├── synchronization.py    # Cross-correlation, offset estimation
│   ├── evaluation.py         # Synthetic experiments and metrics
│   └── reporting.py          # Chart builders and result formatting
│
├── tests/                    # Pytest test suite
│   ├── conftest.py           # Shared fixtures
│   ├── test_video_processing.py
│   ├── test_audio_processing.py
│   ├── test_synchronization.py
│   └── test_evaluation.py
│
└── docs/
    ├── architecture.md       # This file
    ├── algorithm_explained.md
    └── interview_guide.md
```

---

## Module Descriptions

### `app.py` — Streamlit Application Layer

**Responsibility**: User interaction only. Collects inputs, calls `src/` modules,
displays results. Contains no business logic.

**Key sections**:
- Sidebar: configurable thresholds (alignment tolerance, peak quality, sample FPS)
- Tab 1 (Analyze): file upload → validation → pipeline → results
- Tab 2 (Evaluation): synthetic test suite with live results
- Tab 3 (About): documentation and privacy policy

**Dependencies**: all `src/` modules.

---

### `src/video_processing.py` — Video I/O

**Responsibility**: Validate video files, extract metadata, and sample frames
at a controlled rate without loading the entire video into memory.

**Key functions**:
- `validate_video_file(filepath)` → list of errors
- `extract_metadata(filepath)` → `VideoMetadata`
- `sample_frames(filepath, sample_fps, max_frames)` → `List[FrameSample]`
- `process_video(filepath, sample_fps)` → `VideoProcessingResult`

**External dependencies**: `opencv-python-headless` (cv2)

**Design decisions**:
- Uses `frame_interval` sampling to stay within a `max_frames` budget.
- Never loads all frames simultaneously; processes one at a time.
- Returns structured dataclasses, not raw tuples.

---

### `src/audio_processing.py` — Audio Features

**Responsibility**: Extract audio from a video using FFmpeg and compute the
Short-Time Energy (STE) envelope using Librosa.

**Key functions**:
- `check_ffmpeg()` → bool
- `extract_audio_to_wav(video, output_wav)` → (bool, message)
- `compute_audio_features(wav)` → `(AudioFeatures, messages)`
- `process_audio(video_filepath)` → `AudioProcessingResult`

**External dependencies**: `ffmpeg` (system), `librosa`, `soundfile`

**Design decisions**:
- Uses `subprocess.run` with `capture_output=True` for clean FFmpeg integration.
- Temp WAV is always deleted in a `finally` block.
- Voice activity detection uses a simple energy-threshold approach.

---

### `src/face_analysis.py` — Mouth Aperture Measurement

**Responsibility**: Run MediaPipe FaceMesh on each sampled frame, extract the
relevant landmark coordinates, and compute normalized mouth aperture.

**Key functions**:
- `analyze_frames(frames, ...)` → `FaceAnalysisResult`
- `_compute_mouth_aperture(landmarks, w, h)` → float or None

**External dependencies**: `mediapipe`, `opencv-python-headless`

**Design decisions**:
- Uses `static_image_mode=True` (each frame treated independently).
- Normalizes aperture by inter-eye distance for scale invariance.
- Handles multi-face frames by picking the face closest to frame centre.
- Tracks face detection rate; warns if < 60%.

**MediaPipe landmark indices used**:
- Upper inner lip: [13, 312, 311, 310, 415, 308]
- Lower inner lip: [14, 317, 402, 318, 324, 78]
- Inter-eye normalization: landmarks 133 (left inner) and 362 (right inner)

---

### `src/synchronization.py` — Core Algorithm

**Responsibility**: Estimate the temporal offset between the audio energy signal
and the mouth aperture signal using cross-correlation.

**Key functions**:
- `estimate_sync(...)` → `SyncResult`
- `_resample_to_grid(values, timestamps, fps)` → resampled array
- `_normalize(signal)` → zero-mean, unit-std array
- `_smooth(signal, sigma)` → Gaussian-smoothed array
- `_cross_correlate(a, b, fps)` → (correlation, lags_ms)
- `_peak_quality_ratio(corr, peak_idx)` → float

**External dependencies**: `numpy`, `scipy`

**Data flow**:
```
mouth_timestamps + mouth_apertures
audio_timestamps + audio_energy
         │
         ▼
Find temporal overlap window
         │
         ▼
Resample both to 25 Hz grid (np.interp)
         │
         ▼
Gaussian smooth (scipy.ndimage.gaussian_filter1d)
         │
         ▼
Zero-mean, unit-std normalize
         │
         ▼
Cross-correlate (np.correlate, mode='full')
         │
         ▼
Find peak lag → offset in ms
         │
         ▼
Compute peak quality ratio
         │
         ▼
Classify: ALIGNED / MISMATCH / INCONCLUSIVE
```

---

### `src/evaluation.py` — Synthetic Experiments

**Responsibility**: Generate synthetic signals with known offsets, run the
synchronization algorithm, and compute evaluation metrics. All numbers
displayed in the app come from actual execution.

**Key functions**:
- `generate_synthetic_mouth_signal(duration, fps, seed)` → array
- `generate_synthetic_audio_signal(mouth, offset_ms, fps, noise_std, seed)` → array
- `run_single_experiment(offset_ms, ...)` → `ExperimentResult`
- `run_evaluation(offsets_ms, ...)` → `EvaluationSummary`
- `run_inconclusive_test()` → dict

**Design decisions**:
- Synthetic mouth signal = sum of sine waves at 2, 3.5, 5 Hz (typical syllable rates).
- Audio = shifted mouth signal + Gaussian noise (σ=0.1).
- Circular shift (`np.roll`) used to preserve signal length.

---

### `src/reporting.py` — Visualizations and Formatting

**Responsibility**: Build Plotly charts and format result data for Streamlit display.
Contains no analysis logic.

**Key functions**:
- `build_signal_chart(sync_result)` → Plotly Figure
- `build_correlation_chart(sync_result)` → Plotly Figure
- `build_evaluation_chart(summary)` → Plotly Figure
- `verdict_display(result)` → dict with label, color, description
- `offset_explanation(offset_ms)` → human-readable string

---

## Data Flow Between Modules

```
app.py
  │
  ├── video_processing.process_video()
  │       returns: VideoProcessingResult
  │           .metadata (VideoMetadata)
  │           .frames   (List[FrameSample])
  │
  ├── audio_processing.process_audio()
  │       returns: AudioProcessingResult
  │           .features (AudioFeatures)
  │               .energy             [float array]
  │               .energy_timestamps  [float array]
  │
  ├── face_analysis.analyze_frames(frames)
  │       returns: FaceAnalysisResult
  │           .apertures   [float array]
  │           .timestamps  [float array]
  │
  └── synchronization.estimate_sync(
            mouth_timestamps, mouth_apertures,
            audio_timestamps, audio_energy
      )
          returns: SyncResult
              .verdict
              .estimated_offset_ms
              .peak_correlation
              .peak_quality_ratio
              .correlation  (for chart)
              .lags_ms      (for chart)
              .common_timestamps
              .mouth_signal_normalized
              .audio_signal_normalized
```

---

## Key Design Principles

1. **Independence**: `src/` modules have no Streamlit imports. They can be used
   as a library or tested without a browser.
2. **Structured errors**: Every module returns errors and warnings as plain lists
   of strings — never raises exceptions for expected failure cases.
3. **Memory efficiency**: Frames are sampled (not fully loaded), audio is
   processed via a temp file (not stored in RAM).
4. **Honesty**: INCONCLUSIVE is always preferred over a misleading verdict
   when signal quality is insufficient.
5. **Configurability**: Thresholds are module-level constants, not magic numbers
   buried in logic.
