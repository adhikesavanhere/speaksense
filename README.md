# 🎙️ SpeakSense — Audio-Visual Speech Synchronization Auditor

> **An AI/ML project that estimates whether visible mouth movements in a video are temporally aligned with the audio track.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Technology Stack](#3-technology-stack)
4. [Architecture and Processing Flow](#4-architecture-and-processing-flow)
5. [Installation — Windows 11](#5-installation--windows-11)
6. [Running the Application](#6-running-the-application)
7. [Running the Tests](#7-running-the-tests)
8. [Evaluation Methodology and Results](#8-evaluation-methodology-and-results)
9. [Limitations and Responsible Use](#9-limitations-and-responsible-use)
10. [Deployment — Streamlit Community Cloud](#10-deployment--streamlit-community-cloud)
11. [Future Improvements](#11-future-improvements)
12. [Project Structure](#12-project-structure)
13. [Privacy](#13-privacy)

---

## 1. Project Overview

SpeakSense is a Python/Streamlit application that analyzes a video of a speaking
person and estimates whether the visible mouth movements are temporally aligned
with the video's audio track.

### Problem Statement

Audio-visual synchronization is the temporal relationship between speech sounds
and the lip movements that produce them. In genuine recordings, they are naturally
synchronized. Synchronization mismatches can occur due to:

- Re-dubbing or audio replacement
- Video/audio processing pipeline errors
- Codec or container timing offsets
- Video editing or post-production changes

SpeakSense estimates the timing offset between mouth movements and audio, and
classifies the result as **ALIGNED**, **MISMATCH**, or **INCONCLUSIVE**.

> ⚠️ **Important**: SpeakSense is an *auditor*, not a deepfake detector.
> A mismatch result does **not** prove manipulation, deception, or malicious
> intent. Many legitimate processes can create apparent synchronization offsets.

---

## 2. Features

- **Video processing**: Validates format, size, and readability. Extracts metadata.
  Samples frames efficiently without loading the entire video into memory.
- **Face detection**: Uses MediaPipe FaceMesh to detect 478 facial landmarks.
  Computes normalized mouth aperture from lip landmark positions.
- **Audio features**: Extracts audio via FFmpeg. Computes Short-Time Energy (STE)
  envelope using Librosa.
- **Synchronization algorithm**: Resamples both signals to a common time grid.
  Applies Gaussian smoothing and z-score normalization. Computes normalized
  cross-correlation. Estimates offset from the peak lag with a quality ratio check.
- **Configurable thresholds**: Alignment tolerance, peak quality threshold, and
  frame sampling rate are all adjustable from the sidebar.
- **Interactive visualizations**: Plotly charts for the signal overlay,
  cross-correlation function, and evaluation results.
- **Evaluation suite**: Synthetic experiments with known offsets — all results
  are computed from actual algorithm execution, never fabricated.
- **Privacy-safe**: Videos are processed in temporary files and deleted immediately
  after analysis.

---

## 3. Technology Stack

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.10+ | Core language |
| **Streamlit** | 1.35+ | Complete UI, no separate frontend |
| **OpenCV** | 4.9+ | Frame reading and image decoding |
| **MediaPipe** | 0.10.9+ | FaceMesh landmark detection |
| **NumPy** | 1.24–1.x | Signal arrays, normalization, correlation |
| **SciPy** | 1.11+ | Gaussian smoothing (`scipy.ndimage`) |
| **Librosa** | 0.10+ | Audio loading, RMS energy computation |
| **SoundFile** | 0.12+ | Audio I/O fallback |
| **FFmpeg** | any | Audio extraction from video files |
| **Plotly** | 5.20+ | Interactive signal and evaluation charts |
| **Pandas** | 2.1+ | Evaluation results table display |
| **Pytest** | 7.4+ | Automated test suite |

---

## 4. Architecture and Processing Flow

```
Upload Video
     │
     ▼
┌─────────────────┐
│ Video Processing │  ← OpenCV: validate, extract metadata, sample frames
└────────┬────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
┌──────────┐      ┌────────────────┐
│  Audio   │      │ Face Analysis  │
│ Process  │      │ (MediaPipe)    │
│ FFmpeg   │      │ mouth aperture │
│ Librosa  │      │ per frame      │
└────┬─────┘      └──────┬─────────┘
     │                   │
     └──────────┬────────┘
                │
                ▼
     ┌──────────────────────┐
     │   Synchronization    │
     │ • Resample to 25 Hz  │
     │ • Gaussian smooth    │
     │ • Z-score normalize  │
     │ • Cross-correlate    │
     │ • Peak quality check │
     └──────────┬───────────┘
                │
                ▼
        ALIGNED / MISMATCH / INCONCLUSIVE
        + estimated offset in milliseconds
        + signal visualizations
```

Full module documentation: [docs/architecture.md](docs/architecture.md)
Algorithm explanation: [docs/algorithm_explained.md](docs/algorithm_explained.md)

---

## 5. Installation — Windows 11

### Prerequisites

**Python 3.10 or 3.11** (recommended; 3.12 may have MediaPipe issues):
```powershell
python --version
```

**FFmpeg** (required for audio extraction):
1. Download from https://ffmpeg.org/download.html → Windows builds → gyan.dev
2. Extract and add the `bin/` folder to your PATH, OR use winget:
```powershell
winget install Gyan.FFmpeg
```
3. Verify:
```powershell
ffmpeg -version
```

### Setup

```powershell
# 1. Clone or download the project
cd C:\Users\Welcome\.gemini\antigravity\scratch\speaksense

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: If you see a MediaPipe + NumPy compatibility error, ensure
> `numpy<2.0` is installed: `pip install "numpy>=1.24,<2.0"`

---

## 6. Running the Application

```powershell
# From the project root, with venv activated:
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**Tips for best results:**
- Upload a video where the face is clearly visible throughout.
- The speaker should be facing the camera (frontal angle works best).
- Ensure the video has an audio track with clear speech.
- Minimum recommended duration: 3–10 seconds.

---

## 7. Running the Tests

```powershell
# From the project root, with venv activated:
pytest tests/ -v

# With coverage report:
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file:
pytest tests/test_synchronization.py -v

# Run a specific test:
pytest tests/test_synchronization.py::TestVerdicts::test_zero_offset_is_aligned -v
```

### What the tests cover

| Test File | Coverage |
|---|---|
| `test_video_processing.py` | File validation, extension checking, metadata structure |
| `test_audio_processing.py` | WAV feature extraction, energy non-negativity, error cases |
| `test_synchronization.py` | ALIGNED/MISMATCH/INCONCLUSIVE verdicts, offset accuracy, helpers |
| `test_evaluation.py` | Signal generation, experiment structure, evaluation metrics |

> **Note**: Tests that write WAV files require the `wave` standard library module
> (included in Python) and `librosa`. No real video files are needed.

---

## 8. Evaluation Methodology and Results

### Methodology

The synchronization algorithm is validated using synthetic signals with
mathematically known offsets:

1. **Mouth signal**: Sum of sine waves at 2, 3.5, and 5 Hz (typical syllable
   rates), with low-level noise.
2. **Audio signal**: Same signal shifted by a known offset using a **linear
   (non-circular) shift** (zero-filled boundary), plus Gaussian noise (σ=0.1).
   This ensures large offsets produce genuinely misaligned signals.
3. **Test offsets**: 0, ±50, ±100, ±200, ±350, ±500 ms.
4. **Metrics**: Mean Absolute Error (MAE) in ms; percentage within ±80 ms tolerance.

### Results (Synthetic Tests — Actual Measured Values)

> **These results are computed by actually running the algorithm. Run
> `python -m pytest tests/ -v` or open the Evaluation tab in the app to
> reproduce them.**

| True Offset (ms) | Estimated (ms) | Error (ms) | Verdict |
|---|---|---|---|
| 0 | 0 | 0 | ALIGNED |
| +50 | +40 | 10 | ALIGNED |
| +100 | +80 | 20 | ALIGNED |
| −100 | −80 | 20 | ALIGNED |
| +200 | +200 | 0 | MISMATCH |
| −200 | — | — | INCONCLUSIVE (ambiguous peaks) |
| +350 | +360 | 10 | MISMATCH |
| −350 | — | — | INCONCLUSIVE (ambiguous peaks) |
| +500 | — | — | INCONCLUSIVE (ambiguous peaks) |
| −500 | −480 | 20 | MISMATCH |

**Summary (7 eligible / 10 total):**
- Mean Absolute Error (eligible): ~11 ms
- Within ±80 ms tolerance: 7/7 = **100%**
- Inconclusive cases: 3 (ambiguous competing cross-correlation peaks in periodic signals)

### What This Does and Does Not Establish

✅ **Does establish**: The cross-correlation algorithm correctly identifies
temporal offsets in synthetic signals with realistic speech-frequency content.

❌ **Does not establish**: Performance on every real-world video. Real video
results depend on face detection quality, audio complexity, speaker variability,
codec behavior, and many factors not present in synthetic signals.

---

## 9. Limitations and Responsible Use

- **Not a deepfake detector**: A mismatch does not prove manipulation.
- **Not a security tool**: Do not use this as evidence in legal, journalistic,
  or investigative contexts without additional verification.
- **Face detection quality**: Results depend on face visibility, lighting, angle,
  and occlusions. Low detection rates produce unreliable signals.
- **Audio quality**: Music, background noise, or overlapping speech can create
  misleading energy patterns.
- **Codec timing**: Some containers store audio with a deliberate delay.
- **Natural variability**: Speakers do not perfectly sync every phoneme with
  their lips.
- **Configurable thresholds**: The 100 ms alignment tolerance and 1.8 quality
  ratio are reasonable defaults, not universally validated constants.
- **Synthetic validation only**: Real-world validation requires labeled
  synchronized/desynchronized video datasets.

---

## 10. Deployment — Streamlit Community Cloud

### Configuration Files

- `packages.txt`: Lists `ffmpeg` as a system package — Streamlit Cloud installs
  this before starting the app.
- `.streamlit/config.toml`: Sets maximum upload size to 500 MB.
- `requirements.txt`: Python dependencies.

### Steps to Deploy

1. Push the repository to GitHub:
   ```bash
   git remote add origin https://github.com/adhikesavanhere/speaksense.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select your repository → set main file to `app.py`
4. Click Deploy

### Known Deployment Risks

- MediaPipe's prebuilt wheels may not be available for the Cloud's exact
  Linux/Python combination. Check the deploy logs for import errors.
- FFmpeg installation via `packages.txt` should work on Streamlit Cloud's
  Ubuntu environment, but verify by checking the app's audio processing
  in the deployed version.

> **Deployment has not been verified live in this implementation.**
> The configuration is prepared; actual deployment requires GitHub push
> access and a Streamlit account.

---

## 11. Future Improvements

- **Trained ML model**: Train a binary classifier on LRS3 or similar labeled
  audio-visual datasets.
- **Real-video evaluation**: Collect ground-truth synchronized and offset videos
  to measure real-world accuracy.
- **Video tracking mode**: Use MediaPipe's continuous tracking for connected
  frame sequences.
- **Lip velocity features**: Add optical flow of lip landmarks as an additional
  feature.
- **Speaker diarization**: For multi-speaker videos, identify which face is
  speaking at each moment.
- **FFT-based correlation**: Use `scipy.signal.fftconvolve` for faster
  cross-correlation on longer videos.
- **Audio features**: Explore formant-based or MFCC-based features as
  alternatives to STE.
- **Batch processing**: Accept multiple videos and produce a comparison report.

---

## 12. Project Structure

```
speaksense/
├── app.py                        # Streamlit application (UI layer only)
├── requirements.txt              # Python dependencies
├── packages.txt                  # Streamlit Cloud system packages (ffmpeg)
├── .gitignore
├── .streamlit/
│   └── config.toml               # Upload size limit, theme
├── src/
│   ├── __init__.py
│   ├── video_processing.py       # Frame sampling and validation
│   ├── audio_processing.py       # FFmpeg extraction + Librosa STE
│   ├── face_analysis.py          # MediaPipe FaceMesh + mouth aperture
│   ├── synchronization.py        # Cross-correlation algorithm
│   ├── evaluation.py             # Synthetic experiment suite
│   └── reporting.py              # Plotly charts and formatting
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_video_processing.py
│   ├── test_audio_processing.py
│   ├── test_synchronization.py
│   └── test_evaluation.py
└── docs/
    ├── architecture.md           # Module descriptions and data flow
    ├── algorithm_explained.md    # Beginner-friendly algorithm guide
    └── interview_guide.md        # 30-sec / 2-min / 5-min intros + 34 Q&A
```

---

## 13. Privacy

- Uploaded videos are written to a system temp file and **deleted immediately**
  after analysis completes or fails.
- No video, audio, or facial data is stored permanently.
- No data is sent to external services. All processing is local.
- The only network requests are Google Fonts (CSS only) loaded in the Streamlit UI.
- For Streamlit Cloud deployment, processing occurs on Streamlit's servers.
  Users should be informed if deploying to a hosted environment.

---

## Developed for B.Tech Computer Science — AI/ML Technical Interview Demonstration

GitHub: [github.com/adhikesavanhere](https://github.com/adhikesavanhere)
