# Interview Guide — SpeakSense

## 30-Second Introduction

> "I built SpeakSense — an Audio-Visual Speech Synchronization Auditor.
> It takes a video of a speaking person, extracts the mouth movement signal
> using MediaPipe facial landmarks, extracts the audio energy envelope using
> Librosa, and then applies cross-correlation to estimate whether the two
> signals are temporally aligned. The output is a verdict — aligned, possible
> mismatch, or inconclusive — with an estimated offset in milliseconds and
> interactive visualizations. It runs entirely as a Streamlit application."

---

## 2-Minute Explanation

> "SpeakSense addresses a real problem: when someone speaks on video, the mouth
> movements and the audio should happen simultaneously. If a video has been
> re-dubbed or if there is a processing error, they may be slightly out of sync.
>
> The pipeline has four stages. First, I sample video frames using OpenCV and
> run MediaPipe FaceMesh on each frame to extract 478 facial landmarks. From
> those landmarks, I compute the normalized mouth aperture — how open the mouth
> is — frame by frame, giving me a time-series signal.
>
> Second, I use FFmpeg to extract the audio track and Librosa to compute the
> Short-Time Energy envelope. This gives me another time-series that rises
> during speech and falls during silence.
>
> Third, I resample both signals to a common 25 Hz grid, apply Gaussian
> smoothing to reduce noise, and normalize them to zero mean and unit variance.
> Then I compute the normalized cross-correlation.
>
> The lag at the peak of the correlation tells me the estimated time offset.
> I also compute a peak quality ratio to assess whether the peak is
> statistically distinct. If the quality is too low, I return an inconclusive
> result instead of a misleading verdict.
>
> The algorithm was validated on synthetic signals with known offsets. On
> those tests, it achieves reasonable accuracy within about 80 ms tolerance.
> I'm careful to present this as an auditor, not a deepfake detector — a
> mismatch does not prove manipulation."

---

## 5-Minute Technical Walkthrough

> "Let me walk you through the code architecture and the key algorithmic
> decisions.
>
> **Architecture**: The project is organized into a `src/` library and a
> Streamlit `app.py`. The library has no Streamlit imports, so all modules can
> be tested with pytest independently. The app is purely the glue layer.
>
> **Video processing** (`src/video_processing.py`): I validate the file format,
> size, and readability using OpenCV. Frame sampling uses a frame interval:
> native FPS divided by target sample FPS. If the video is very long, the
> interval is widened to stay within a 600-frame budget, so large files don't
> crash the app on a laptop.
>
> **Audio processing** (`src/audio_processing.py`): FFmpeg is invoked as a
> subprocess. I call it with `-vn -ac 1 -ar 22050 -acodec pcm_s16le` flags to
> extract a mono 16-bit PCM WAV at 22050 Hz. Librosa then loads this WAV and
> computes RMS energy using `librosa.feature.rms` with a frame length of 2048
> samples and a hop of 512 samples. The temp WAV is always deleted in a finally
> block. Voice activity is estimated as energy > mean + 0.5*std.
>
> **Face analysis** (`src/face_analysis.py`): MediaPipe FaceMesh in static
> image mode detects 478 landmarks per frame. I use a specific subset of
> landmarks for the inner upper and lower lips, compute their mean y-coordinates,
> and take the absolute difference. I normalize by the inter-eye distance
> (landmark 133 to 362) to make the measurement scale-invariant. For videos
> with multiple faces, I pick the face closest to the frame centre.
>
> **Synchronization** (`src/synchronization.py`): Both signals are resampled to
> 25 Hz using np.interp over a common time grid. I chose 25 Hz because it's
> faster than audio frame rate but well above the 6 Hz maximum speech rate.
> Gaussian smoothing with sigma=2 samples suppresses frame noise. After
> z-score normalization, I compute the full cross-correlation with
> np.correlate(mode='full'). The peak lag in samples is converted to
> milliseconds using the 25 Hz rate. The peak quality ratio is the peak
> correlation divided by the mean of all other absolute values — if this is
> below 1.8, I return INCONCLUSIVE.
>
> **Evaluation** (`src/evaluation.py`): I generate synthetic mouth signals as
> sums of sine waves at 2, 3.5, and 5 Hz. The audio signal is the mouth signal
> circularly shifted by the known offset, with added Gaussian noise. I test
> offsets from ±500 ms and compute MAE and within-80ms accuracy.
>
> **Interface** (`app.py`): Streamlit with a dark custom CSS theme, three tabs,
> a sidebar for parameter tuning, Plotly charts for the signal overlay and
> cross-correlation function, and a live evaluation suite that runs and
> displays actual results."

---

## 30+ Technical Interview Questions and Answers

### Python and General Engineering

**Q1: Why did you separate the processing logic from the Streamlit UI?**

A: Streamlit's `st.*` calls can only run in the main thread and require a
browser session. If I mixed them with my processing logic, I couldn't write
meaningful unit tests — pytest doesn't have a Streamlit session. By keeping
`src/` free of Streamlit imports, every function can be tested with pytest
independently. The app layer is purely for display and orchestration.

---

**Q2: How did you handle temporary files safely?**

A: I used `tempfile.mkstemp()` to create a uniquely named temporary WAV file.
After the file descriptor is opened, I close it immediately and pass the path
to FFmpeg. After processing, whether or not it succeeded, I delete the file
in a `finally` block. For uploaded videos in Streamlit, the same pattern
applies — the temp video file is deleted in a `finally` block after analysis.

---

**Q3: What does the `@dataclass` decorator do and why did you use it?**

A: `@dataclass` automatically generates `__init__`, `__repr__`, and `__eq__`
methods from the class's annotated fields. I used it for all result containers
(`VideoProcessingResult`, `AudioFeatures`, `SyncResult`, etc.) because it
produces readable, self-documenting code with type annotations. It also makes
the results easy to inspect in tests.

---

**Q4: Why use `subprocess.run` for FFmpeg instead of a Python FFmpeg library?**

A: `subprocess.run` with `capture_output=True` gives me direct control over
the exact FFmpeg command flags, the return code, and stderr output. Python
FFmpeg wrapper libraries are often incomplete or have their own bugs.
Using subprocess directly means I can specify exactly `-ac 1 -ar 22050
-acodec pcm_s16le`, check the return code, parse stderr for "no audio"
errors, and the dependency tree stays minimal.

---

**Q5: How does your code handle the case where the video has no audio?**

A: In `extract_audio_to_wav`, if FFmpeg returns a non-zero exit code, I check
the stderr for phrases like "no audio". I return `(False, "No audio stream
found")`. In `process_audio`, the `audio_extracted` field is `False` and the
errors list explains the issue. In `app.py`, I check this condition before
calling `estimate_sync` and display a clear warning to the user.

---

**Q6: What is the difference between `numpy<2.0` and `numpy>=2.0` for your project?**

A: MediaPipe's prebuilt wheel has binary dependencies that were compiled
against NumPy 1.x's C API. NumPy 2.0 made breaking changes to the ABI
(Application Binary Interface). If you install NumPy 2.x alongside MediaPipe,
you get an import error like "module compiled against ABI version...". Pinning
`numpy>=1.24,<2.0` avoids this.

---

### Algorithm and Signal Processing

**Q7: What is Short-Time Energy and why is it a good proxy for speech activity?**

A: Short-Time Energy (STE) is the Root Mean Square (RMS) of audio samples
within a short window. Speech is louder than silence, so STE rises when someone
is speaking and falls during pauses. It's not perfect — music and background
noise also produce high STE — but it captures the amplitude envelope of the
signal, which correlates loosely with the rhythm of speech.

---

**Q8: Why did you choose cross-correlation instead of Dynamic Time Warping (DTW)?**

A: Cross-correlation assumes that the two signals are related by a constant
time shift. For the audio-visual sync problem, this is a reasonable model —
if the audio was dubbed or compressed, it shifts the entire audio track by a
fixed amount relative to the video. DTW finds a non-linear warping, which
would be appropriate if the speech rate varied, but for sync estimation a
linear shift is the right model. Cross-correlation is also `O(n log n)` with
FFT and has a clear interpretation (the lag at peak = the offset).

---

**Q9: What does the peak quality ratio measure and why is it important?**

A: The peak quality ratio is the peak correlation value divided by the mean
of all other absolute correlation values. It measures how "distinct" the peak
is relative to the background. If the ratio is low (close to 1.0), it means
many lags produce similar correlation values — there's no clear winner and any
estimate would be essentially random. A high ratio (e.g., 3.0) means one lag
is clearly the best match. Without this check, the algorithm would always
return a number even when the data is meaningless.

---

**Q10: What is the sign convention for your estimated offset?**

A: Positive offset means audio leads video — the audio energy event happens
before the corresponding mouth opening. Negative offset means video leads audio
— the mouth opens before the corresponding audio. This is documented in the
module docstring and displayed in the app UI.

---

**Q11: Why resample to 25 Hz instead of a higher rate?**

A: Speech movement rates are typically 2–8 Hz (syllables per second). Nyquist's
theorem says we need at least 16 Hz to represent 8 Hz signals. 25 Hz gives
us comfortable headroom, and it divides evenly into common video frame rates
(25 fps, 50 fps). Going higher (e.g., 100 Hz) would give us more temporal
resolution for offset estimation but at much greater computational cost
with diminishing returns for signals whose useful content is below 8 Hz.

---

**Q12: How does Gaussian smoothing affect the correlation result?**

A: Smoothing removes high-frequency noise and acts as a low-pass filter.
This makes the correlation function cleaner — the peak is easier to find and
the peak quality ratio is higher. Without smoothing, frame-level noise can
create many small spurious correlation peaks. The sigma=2 samples at 25 Hz
corresponds to a time constant of about 80 ms, which suppresses frame-level
noise while preserving syllable-scale patterns.

---

**Q13: What is `np.correlate(mode='full')` doing exactly?**

A: `np.correlate(a, b, mode='full')` computes:
```
C[k] = sum_n( a[n] * b[n - k] )
```
for all k from -(n-1) to +(n-1), where n = len(a) = len(b).
`mode='full'` returns the full 2n-1 length output. The center of this output
corresponds to zero lag. Values to the right correspond to positive lags
(a leads b), values to the left correspond to negative lags (b leads a).

---

**Q14: What is the Nyquist-Shannon sampling theorem and how does it apply here?**

A: The theorem states that to accurately represent a signal of maximum
frequency f_max, you need to sample at least at 2 * f_max. Speech mouth
movements have content up to about 8 Hz. We sample video at 10–25 fps, so
our frame rate exceeds Nyquist. However, when resampling to 25 Hz, we're
still above Nyquist for the relevant frequency range, so no useful information
is lost.

---

### Computer Vision

**Q15: What is MediaPipe FaceMesh and what does it output?**

A: MediaPipe FaceMesh is a convolutional neural network model that detects
and tracks 478 3D facial landmarks in real time. Each landmark is a point
on the face surface, expressed in normalized image coordinates (x, y ∈ [0,1])
and a depth coordinate. The 478-point mesh covers the face including the
eyes, nose, lips, jawline, and eyebrows. We use specific lip landmark indices
to extract mouth geometry.

---

**Q16: Why did you use `static_image_mode=True` for MediaPipe?**

A: In `static_image_mode=False` (the default for video), MediaPipe runs a
lightweight tracker after the initial detection to speed up processing.
However, this tracking can drift if frames are not contiguous. Since we're
sampling frames at irregular intervals (skipping many frames), the tracker
would be working with frames that are not consecutive. Using
`static_image_mode=True` treats each frame independently with full detection,
which is more accurate for our use case even if slower.

---

**Q17: How did you normalize the mouth aperture measurement?**

A: I compute the absolute difference between the mean y-coordinate of the
upper inner lip landmarks and the mean y-coordinate of the lower inner lip
landmarks. I then divide by the inter-eye distance (the x-distance between
landmarks 133 and 362). This normalization makes the measurement independent
of the camera distance and face size — a big face close to the camera and a
small face far away produce similar aperture values for the same degree of
mouth opening.

---

**Q18: How do you handle videos with multiple faces?**

A: If MediaPipe detects multiple faces, I select the face whose nose tip
(landmark 1) is closest to the centre of the frame, measured in Euclidean
distance. The assumption is that the primary speaker is usually in the centre
of the frame. This is a heuristic — for multiple speakers, the user would
need to crop the video to a single face first.

---

**Q19: What are the limitations of MediaPipe FaceMesh for this application?**

A: (1) It can fail at large head angles (> ~45° from frontal). (2) Lighting
conditions affect detection reliability — very dark, harsh side-lighting, or
low contrast can cause misses. (3) The lip landmark positions may be slightly
inaccurate for people whose facial features differ significantly from the
training data. (4) Glasses, facial hair, or face masks can interfere with lip
detection. (5) We use `static_image_mode` which is slower than video tracking.

---

### Testing and Evaluation

**Q20: How did you test the synchronization algorithm without real video data?**

A: I created synthetic signals with mathematically known offsets. The mouth
signal is a sum of sine waves at realistic speech frequencies. The audio
signal is the same signal shifted by a known number of samples, plus Gaussian
noise. By running the algorithm on these synthetic pairs and comparing the
estimated offset to the true offset, I can measure the algorithm's accuracy
independently of any face detection uncertainty or audio quality issues.

---

**Q21: What is Mean Absolute Error (MAE) and what does it tell you?**

A: MAE = mean(|estimated_offset - true_offset|) across all eligible experiments.
It tells you the average magnitude of the estimation error in milliseconds.
A lower MAE means the algorithm is more accurate on average. I chose MAE over
Mean Squared Error (MSE) because MAE is in the original units (ms) and treats
all errors linearly — MSE would disproportionately penalize large errors.

---

**Q22: What is the "within tolerance" metric and why 80 ms?**

A: "Within tolerance" counts the fraction of experiments where |error| ≤ 80 ms.
I chose 80 ms because perceptual research suggests humans can tolerate audio-
visual sync offsets of roughly 100–180 ms before noticing them. An estimation
error of 80 ms would leave room for the threshold comparison (the default
alignment threshold is 100 ms) to be meaningful. This is a practical choice,
not a universal constant.

---

**Q23: Why do you use `np.random.default_rng(seed)` instead of `np.random.seed()`?**

A: `np.random.default_rng(seed)` uses the new-style Generator API introduced
in NumPy 1.17. It creates an isolated random number generator that doesn't
affect the global random state. `np.random.seed()` sets the legacy global
state, which can interact unexpectedly with other code. The new API also has
better statistical properties and is thread-safe.

---

**Q24: What does `scope="class"` on a pytest fixture do?**

A: It means the fixture is created once per test class (not once per test
function). In my `TestRunEvaluation` class, I use `scope="class"` on the
`evaluation_summary` fixture so that the expensive `run_evaluation()` call
happens only once, and all test methods in the class share the same result.

---

**Q25: How would you test the face analysis module?**

A: Face analysis requires actual image data with faces. The cleanest approach
is to programmatically generate a synthetic face image (using OpenCV's drawing
functions to create a simple mouth-like geometry) or to include a small
publicly-licensed test image of a face in the repository. For the current
implementation, face analysis is tested implicitly through integration — the
module is exercised when the full pipeline runs on real video input. Isolated
unit testing of `_compute_mouth_aperture` can be done by constructing a mock
landmarks object.

---

### Design Decisions

**Q26: Why Streamlit and not Flask/FastAPI + React?**

A: For a student project focused on AI/ML algorithms, Streamlit provides the
complete UI in a single Python file without requiring JavaScript, React,
TypeScript, or a separate backend server. This keeps the tech stack minimal,
appropriate for demonstrating ML skills rather than web development skills.
The tradeoff is less control over the UI and no real-time updates — acceptable
for batch video analysis.

---

**Q27: Why Plotly instead of Matplotlib for charts?**

A: Plotly charts are interactive (pan, zoom, hover tooltips) within Streamlit
without any additional configuration. Matplotlib produces static images
unless paired with additional tools. For presenting signal visualizations where
the interviewer might want to zoom in on a specific time window, interactive
charts are more useful.

---

**Q28: What would you add if you had more time?**

A: (1) A trained classifier — use a dataset like LRS3 (Lip Reading Sentences 3)
to train a model that directly predicts alignment from audio-visual features.
(2) Better face tracking — use MediaPipe's video mode for connected frames.
(3) Optical flow of lips — instead of just aperture, track the velocity of lip
motion. (4) Speaker diarization — handle multiple speakers by identifying
which face is speaking at each moment. (5) Deployment validation — actually
deploy to Streamlit Cloud and verify the FFmpeg system package installs
correctly.

---

**Q29: What is a potential false positive in this system?**

A: A false positive (reporting MISMATCH for an actually-synchronized video)
could occur when: (1) Natural speaking variability — the speaker's mouth
movements are not perfectly synchronized with every phoneme. (2) Dubbing
with intentional timing offsets — professional dubbing sometimes shifts audio
by small amounts for perceived naturalness. (3) Codec or container differences
— some video containers store audio with a small intentional delay (typically
500 ms for Bluetooth audio). (4) Music, laughter, or environmental audio that
doesn't match mouth movements.

---

**Q30: What is a potential false negative in this system?**

A: A false negative (reporting ALIGNED for a manipulated video) could occur
when: (1) The manipulation offset is small (within the 100 ms tolerance).
(2) The mouth movements are very regular and periodic — the correlation might
find a false peak at the period of the signal rather than the true offset.
(3) The video has low face detection quality, causing the mouth signal to be
flat and the result to be INCONCLUSIVE (not a false negative, but not helpful
either). (4) Both audio and video are independently manipulated in ways that
happen to align.

---

**Q31: What is the computational complexity of the cross-correlation?**

A: Direct `np.correlate(mode='full')` on two signals of length n is O(n²)
in the worst case. NumPy typically uses FFT-based convolution for long signals,
making it O(n log n). For our signals at 25 Hz over 60 seconds, n ≈ 1500,
so performance is not a concern. If performance were critical (much longer
signals), I would explicitly use `scipy.signal.fftconvolve`.

---

**Q32: Why do you clip mouth aperture values to [0, 2.0]?**

A: In theory, the normalized aperture (raw opening / inter-eye distance)
should be in [0, ~0.5] for a normal face. Occasionally, landmark estimation
errors produce extreme values (very large numbers or negative numbers due to
landmark flips). Clipping to [0, 2.0] prevents these outliers from dominating
the signal mean and standard deviation, which would otherwise suppress the
useful variation during normalization.

---

**Q33: What is the `gcd` import in synchronization.py used for?**

A: It was imported as part of a planned rational-number approach to computing
resample factors (up_factor/down_factor = target_fps / source_fps, reduced
by GCD). The final implementation uses `np.interp` which doesn't need GCD.
The import is present but unused — a minor code quality issue that should be
cleaned up in a production codebase.

---

**Q34: How would you deploy this to Streamlit Community Cloud?**

A: (1) Push the code to a public GitHub repository. (2) Connect the repository
to share.streamlit.io. (3) The `packages.txt` file with `ffmpeg` instructs
Streamlit Cloud to install FFmpeg as a system package before starting the app.
(4) `requirements.txt` handles Python dependencies. The main risk is that
MediaPipe may not install on the Cloud's Linux environment without additional
configuration — this would need to be verified by actually deploying and
checking logs.

---

## What Was Implemented vs. Future Work

### Implemented
- Complete video processing pipeline (validation, sampling, metadata)
- Audio extraction via FFmpeg subprocess
- Short-Time Energy feature computation via Librosa
- MediaPipe FaceMesh mouth aperture measurement
- Cross-correlation synchronization algorithm with peak quality assessment
- ALIGNED / MISMATCH / INCONCLUSIVE verdict classification
- Configurable thresholds with documented sign convention
- Full Streamlit UI with dark theme, interactive Plotly charts, sidebar settings
- Synthetic evaluation suite with MAE and within-tolerance metrics
- Automated pytest test suite (35+ test cases)
- Complete documentation (architecture, algorithm, interview guide)
- Streamlit Cloud deployment configuration (packages.txt, config.toml)
- Privacy-safe temp file handling with cleanup

### Future Work
- Trained ML classifier on a labeled audio-visual dataset (e.g., LRS3)
- MediaPipe video mode for connected frame tracking
- Lip velocity / optical flow features
- Speaker diarization for multi-speaker videos
- Actual Streamlit Cloud deployment verification
- Real-video evaluation with ground truth synchronized samples
- Browser-based video preview in the Streamlit interface
