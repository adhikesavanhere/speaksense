# Algorithm Explained — SpeakSense

This document explains every step of the SpeakSense synchronization algorithm
in plain English, suitable for a student who is new to signal processing.

---

## The Core Question

Given a video of a person speaking, are the mouth movements happening at the
same time as the spoken audio?

If someone records a video and later replaces the audio, or if a video has been
manipulated, the audio and mouth movements may be slightly out of sync.

SpeakSense measures this timing difference.

---

## Step 1 — What Is a "Mouth Aperture Signal"?

Every time the speaker opens their mouth to say a syllable, the gap between
their upper and lower lips gets larger. When they pause or close their mouth,
the gap gets smaller.

MediaPipe FaceMesh detects 478 facial landmarks — tiny numbered points placed
on the face. Some of those points are on the upper and lower lips.

We measure the **vertical distance between the upper lip center and lower lip
center** in every video frame. We then divide by the **inter-eye distance**
(distance between the two inner eye corners) to make the measurement
independent of how close the camera is to the face.

This gives us one number per video frame — the **normalized mouth aperture**.

Plotting this over time produces a waveform that rises and falls as the speaker
talks: the **mouth-aperture signal**.

---

## Step 2 — What Is an "Audio Energy Signal"?

Audio is a waveform — a rapidly oscillating signal measured in samples (e.g.,
22050 samples per second). When someone speaks, the audio amplitude increases.
When they pause, it drops.

**Short-Time Energy (STE)** is a measure of how loud the audio is in a short
window of time. Specifically, it is the Root Mean Square (RMS) of the audio
samples within a small window:

```
STE(t) = sqrt( mean( samples[t]² ) )  over a window of ~2048 samples
```

We compute STE over a sliding window, moving it forward by 512 samples at a
time (the "hop length"). This gives us one energy value per hop — the
**audio energy signal**.

When speech occurs, STE rises. During silence, it drops. This makes STE a
rough proxy for "is speech happening right now?"

---

## Step 3 — Getting Both Signals on the Same Time Grid

The mouth-aperture signal has one value per video frame (e.g., 10 per second
if we sample at 10 fps). The audio energy signal has one value per hop
(approximately 43 per second at librosa defaults).

These two signals are sampled at **different rates** and may not start at
exactly the same time. We need to put them on a **common time grid** before
comparing them.

We choose a target rate of **25 Hz** (one value every 40 ms). We then use
**linear interpolation** (`np.interp`) to resample both signals to this grid,
spanning only the time window where both signals overlap.

---

## Step 4 — Smoothing

Both signals are noisy. The mouth doesn't open perfectly smoothly frame by
frame, and audio energy can fluctuate from one window to the next.

We apply **Gaussian smoothing** — a filter that replaces each sample with a
weighted average of its neighbours, with the weights following a bell curve.
Samples closer to the centre of the window get more weight.

This suppresses high-frequency noise while preserving the slower syllable-rate
pattern (typically 2–6 Hz for speech).

---

## Step 5 — Normalization

The two signals have different physical units and different ranges:

- Mouth aperture is a dimensionless ratio (typically 0.02 – 0.3).
- Audio energy is in units of amplitude (varies by microphone, distance, etc.).

We apply **zero-mean, unit-variance normalization** (also called z-score):

```
normalized(x) = (x - mean(x)) / std(x)
```

After normalization, both signals have mean ≈ 0 and standard deviation ≈ 1.
This makes the cross-correlation meaningful regardless of absolute scale.

If a signal is nearly flat (std ≈ 0), we cannot normalize it sensibly — we
return an INCONCLUSIVE result instead.

---

## Step 6 — Cross-Correlation

Cross-correlation measures how similar two signals are **as a function of
their relative time shift** (called the "lag").

Imagine sliding one signal forward or backward in time and measuring how well
it matches the other signal at each position. The position where the match is
best gives the estimated **temporal offset**.

Mathematically:

```
C(lag) = sum_over_t( audio(t) * mouth(t - lag) )
```

We use `numpy.correlate(mode='full')` which computes this for all lags from
-(n-1) to +(n-1) samples, where n is the signal length.

The result is a curve (the correlation function). The peak of this curve is
the lag at which audio and mouth are most similar.

---

## Step 7 — Estimating the Offset

We find the **index of the maximum value** in the correlation function:

```python
peak_idx = np.argmax(correlation)
```

We convert the sample index to milliseconds using the target sample rate
(25 Hz → 1 sample = 40 ms):

```
offset_ms = lag_samples / 25.0 * 1000.0
```

### Sign Convention (important!)

- **Positive offset** → audio leads video (audio events occur before the mouth opens)
- **Negative offset** → video leads audio (mouth opens before the audio event)

A perfectly synchronized video should have an offset near 0 ms.
Human perception can tolerate offsets up to about 100–180 ms.

---

## Step 8 — Peak Quality Ratio

Not all peaks are trustworthy. If the correlation curve is noisy and lumpy,
there may be many near-equal peaks, and picking the maximum is essentially
guessing.

We compute a **peak quality ratio**:

```
quality_ratio = correlation[peak_idx] / mean(|correlation| excluding the peak neighbourhood)
```

A high ratio (e.g., > 1.8) means the peak stands clearly above the rest —
the algorithm is confident. A low ratio (e.g., < 1.8) means many lags are
equally plausible — we return INCONCLUSIVE.

---

## Step 9 — Classification

Using configurable thresholds:

| Condition | Verdict |
|---|---|
| `peak_quality_ratio < 1.8` | **INCONCLUSIVE** — cannot determine alignment |
| `|offset_ms| < 100` and quality OK | **ALIGNED** — within tolerance |
| `|offset_ms| >= 100` and quality OK | **MISMATCH** — possible synchronization problem |

These thresholds are defaults. They can be adjusted in the Streamlit sidebar.

---

## Failure Cases

The algorithm returns INCONCLUSIVE in the following situations:

1. **Flat audio** — the audio track is silent. Without variation in the audio signal, there is nothing to correlate.
2. **Flat mouth signal** — the face is not detected in enough frames, or the speaker is not visibly moving their mouth (e.g., reading silently, face occluded).
3. **Too short** — the video is shorter than 1 second, or the temporal overlap between the signals is insufficient.
4. **Poor peak quality** — the correlation function has no clear peak (common with music, background noise, or irregular speaking patterns).
5. **Too few data points** — fewer than 5 measurements in either signal.

---

## Evaluation Metrics

For each synthetic test case:
- **True offset**: the offset we applied to create the test signal (ground truth).
- **Estimated offset**: the offset our algorithm estimated.
- **Error**: |estimated - true| in milliseconds.
- **Within tolerance**: whether the error is ≤ 80 ms.

Across all eligible experiments (those that returned a numeric estimate):
- **MAE (Mean Absolute Error)**: average of all individual errors.
- **Within-tolerance accuracy**: percentage of experiments where the error ≤ 80 ms.

---

## Important Limitations

1. **Not a trained model** — SpeakSense does not use machine learning. It uses classical signal processing. It has no concept of what "normal" looks like for different speakers, accents, or environments.
2. **Not a deepfake detector** — a mismatch could result from natural speaking variability, codec resampling, post-production audio editing, or compression artifacts — not just manipulation.
3. **Synthetic validation only** — the algorithm was validated on synthetic signals. Real-world performance on real videos depends on many additional factors.
4. **Landmark accuracy** — MediaPipe FaceMesh landmark positions can be inaccurate in poor lighting, at extreme angles, with occlusions, or for faces outside its training distribution.
5. **Thresholds are configurable defaults** — the 100 ms alignment threshold and 1.8 quality ratio are reasonable starting points, not universal constants.
