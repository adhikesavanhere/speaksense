"""
app.py — SpeakSense: Audio-Visual Speech Synchronization Auditor
================================================================
Complete Streamlit application.  Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Suppress matplotlib permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import streamlit as st
import numpy as np

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from the project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.video_processing import process_video, SUPPORTED_EXTENSIONS
from src.audio_processing import process_audio, check_ffmpeg
from src.face_analysis import analyze_frames
from src.synchronization import estimate_sync
from src.evaluation import run_evaluation, run_inconclusive_test
from src.reporting import (
    build_signal_chart,
    build_correlation_chart,
    build_evaluation_chart,
    verdict_display,
    offset_explanation,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SpeakSense — Speech Sync Auditor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, premium aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0a1a;
    --surface: #12122a;
    --surface2: #1a1a35;
    --accent: #7c6af0;
    --accent2: #4fc3f7;
    --accent3: #ffb74d;
    --green: #66bb6a;
    --red: #ef5350;
    --orange: #ffa726;
    --text: #e8e8f0;
    --subtext: #9090b0;
    --border: #2a2a4a;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: linear-gradient(135deg, #080818 0%, #0f0f25 50%, #0a1020 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d20 0%, #12122a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Cards */
.ss-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(124,106,240,0.08);
}

/* Verdict badge */
.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 14px 28px;
    border-radius: 50px;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 16px 0;
    border: 2px solid;
}
.verdict-aligned  { color: #66bb6a; border-color: #66bb6a; background: rgba(102,187,106,0.08); }
.verdict-mismatch { color: #ef5350; border-color: #ef5350; background: rgba(239,83,80,0.08); }
.verdict-inconclusive { color: #ffa726; border-color: #ffa726; background: rgba(255,167,38,0.08); }

/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-label { font-size: 0.78rem; color: var(--subtext); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: var(--accent2); }
.metric-unit  { font-size: 0.85rem; color: var(--subtext); }

/* Section title */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent2);
    letter-spacing: 0.5px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Warning / info boxes */
.ss-warning {
    background: rgba(255,167,38,0.08);
    border-left: 3px solid #ffa726;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ffd07a;
    font-size: 0.9rem;
}
.ss-info {
    background: rgba(79,195,247,0.08);
    border-left: 3px solid #4fc3f7;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #a0d8f5;
    font-size: 0.9rem;
}
.ss-error {
    background: rgba(239,83,80,0.08);
    border-left: 3px solid #ef5350;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ff9090;
    font-size: 0.9rem;
}
.ss-success {
    background: rgba(102,187,106,0.08);
    border-left: 3px solid #66bb6a;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #a0e0a3;
    font-size: 0.9rem;
}

/* Tag pill */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--subtext);
}

/* Upload area styling */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7c6af0 0%, #4fc3f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,106,240,0.4) !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c6af0, #4fc3f7) !important;
    border-radius: 4px !important;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--surface2) !important;
    color: #a0b4ff !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* Plotly charts background fix */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .bg {
    fill: transparent !important;
}

h1 { font-weight: 700 !important; }
h2, h3 { font-weight: 600 !important; }

/* Hide Streamlit hamburger/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:3rem;">🎙️</div>
        <div style="font-size:1.2rem; font-weight:700; color:#7c6af0;">SpeakSense</div>
        <div style="font-size:0.78rem; color:#9090b0; margin-top:4px;">Speech Sync Auditor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    sample_fps = st.slider(
        "Frame sampling rate (fps)",
        min_value=5, max_value=25, value=10,
        help="Frames sampled per second for face analysis. Higher = slower but more accurate.",
    )
    aligned_threshold = st.slider(
        "Alignment tolerance (ms)",
        min_value=50, max_value=300, value=100, step=25,
        help="Maximum offset considered 'aligned'. Below this threshold the result is ALIGNED.",
    )
    peak_quality_threshold = st.slider(
        "Peak quality threshold",
        min_value=1.2, max_value=3.0, value=1.8, step=0.1,
        help="Minimum cross-correlation peak quality ratio required to trust the result.",
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style="font-size:0.82rem; color:#9090b0; line-height:1.6;">
    SpeakSense estimates whether visible mouth movements are temporally aligned
    with a video's audio track.<br><br>
    <b style="color:#ffa726;">⚠ Important</b><br>
    A mismatch does <em>not</em> prove manipulation, deception, or malicious intent.
    This is an <em>auditor</em>, not a deepfake detector.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # FFmpeg status
    ffmpeg_ok = check_ffmpeg()
    if ffmpeg_ok:
        st.markdown("""<div class="ss-success">✅ FFmpeg detected</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="ss-error">❌ FFmpeg not found — audio analysis disabled</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content — tabs
# ---------------------------------------------------------------------------

tab_analyze, tab_evaluation, tab_about = st.tabs(["🔍 Analyze Video", "📊 Evaluation", "📖 About"])


# ===========================================================================
# TAB 1 — Analyze Video
# ===========================================================================

with tab_analyze:
    # Header
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h1 style="font-size:2.4rem; background:linear-gradient(90deg,#7c6af0,#4fc3f7);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin-bottom:8px;">
            SpeakSense
        </h1>
        <p style="font-size:1.05rem; color:#9090b0; max-width:650px; line-height:1.7;">
            Upload a video of a speaking person. SpeakSense will analyze whether the
            visible mouth movements are temporally aligned with the audio track using
            computer vision and signal processing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="section-title">📤 Upload Video</div>', unsafe_allow_html=True)
    ext_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    uploaded_file = st.file_uploader(
        f"Supported formats: {ext_str} (max 500 MB)",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        # Save to temp file for processing
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp_fd, tmp_video_path = tempfile.mkstemp(suffix=suffix, prefix="speaksense_video_")
        os.close(tmp_fd)

        with open(tmp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.markdown(f"""
        <div class="ss-info">
            📁 <b>{uploaded_file.name}</b> — {uploaded_file.size / (1024*1024):.1f} MB uploaded
        </div>
        """, unsafe_allow_html=True)

        # Quick metadata display
        from src.video_processing import extract_metadata
        meta = extract_metadata(tmp_video_path)
        if meta:
            cols = st.columns(4)
            meta_items = [
                ("Resolution", f"{meta.width}×{meta.height}", "px"),
                ("Duration", f"{meta.duration_sec:.1f}", "sec"),
                ("Frame Rate", f"{meta.fps:.1f}", "fps"),
                ("File Size", f"{meta.file_size_mb:.1f}", "MB"),
            ]
            for col, (label, value, unit) in zip(cols, meta_items):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-unit">{unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")

        # Analyze button
        run_analysis = st.button("🚀 Analyze Video", use_container_width=True)

        if run_analysis:
            st.markdown("---")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # ----------------------------------------------------------------
                # Step 1: Video processing
                # ----------------------------------------------------------------
                status_text.markdown("**Step 1/4 — Validating and sampling video frames…**")
                progress_bar.progress(10)

                video_result = process_video(tmp_video_path, sample_fps=sample_fps)

                if video_result.errors:
                    for err in video_result.errors:
                        st.markdown(f'<div class="ss-error">❌ {err}</div>', unsafe_allow_html=True)
                    st.stop()

                for warn in video_result.warnings:
                    st.markdown(f'<div class="ss-warning">⚠️ {warn}</div>', unsafe_allow_html=True)

                n_frames = len(video_result.frames)
                progress_bar.progress(25)
                status_text.markdown(f"**Step 1/4 — ✅ Sampled {n_frames} frames**")

                # ----------------------------------------------------------------
                # Step 2: Audio processing
                # ----------------------------------------------------------------
                status_text.markdown("**Step 2/4 — Extracting and processing audio…**")
                progress_bar.progress(35)

                audio_result = process_audio(tmp_video_path)
                audio_ok = audio_result.features is not None

                if audio_result.errors:
                    for err in audio_result.errors:
                        st.markdown(f'<div class="ss-warning">⚠️ Audio: {err}</div>', unsafe_allow_html=True)

                progress_bar.progress(50)

                # ----------------------------------------------------------------
                # Step 3: Face analysis
                # ----------------------------------------------------------------
                status_text.markdown(f"**Step 3/4 — Analyzing face & mouth movement across {n_frames} frames…**")

                face_result = analyze_frames(video_result.frames)

                if face_result.errors:
                    for err in face_result.errors:
                        st.markdown(f'<div class="ss-error">❌ Face analysis: {err}</div>', unsafe_allow_html=True)
                    st.stop()

                for warn in face_result.warnings:
                    st.markdown(f'<div class="ss-warning">⚠️ {warn}</div>', unsafe_allow_html=True)

                progress_bar.progress(75)
                status_text.markdown(
                    f"**Step 3/4 — ✅ Face detected in "
                    f"{face_result.face_detection_rate * 100:.0f}% of frames**"
                )

                # ----------------------------------------------------------------
                # Step 4: Synchronization
                # ----------------------------------------------------------------
                status_text.markdown("**Step 4/4 — Estimating audio-visual synchronization…**")

                if not audio_ok:
                    sync_result = None
                    st.markdown("""
                    <div class="ss-warning">
                        ⚠️ Audio could not be extracted or processed. Synchronization analysis
                        requires both audio and video signals.
                    </div>
                    """, unsafe_allow_html=True)
                elif len(face_result.apertures) < 5:
                    sync_result = None
                    st.markdown("""
                    <div class="ss-warning">
                        ⚠️ Insufficient face detection data for synchronization analysis.
                        Ensure the face is clearly visible throughout the video.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    sync_result = estimate_sync(
                        mouth_timestamps=face_result.timestamps,
                        mouth_apertures=face_result.apertures,
                        audio_timestamps=audio_result.features.energy_timestamps,
                        audio_energy=audio_result.features.energy,
                        aligned_threshold_ms=float(aligned_threshold),
                        peak_quality_threshold=float(peak_quality_threshold),
                    )

                progress_bar.progress(100)
                status_text.markdown("**✅ Analysis complete!**")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

                st.markdown("---")

                # ================================================================
                # RESULTS
                # ================================================================

                if sync_result is not None:
                    vd = verdict_display(sync_result)

                    # Verdict badge
                    verdict_class = {
                        "✅ LIKELY ALIGNED": "verdict-aligned",
                        "⚠️ POSSIBLE MISMATCH": "verdict-mismatch",
                        "❓ INCONCLUSIVE": "verdict-inconclusive",
                    }.get(vd["label"], "verdict-inconclusive")

                    st.markdown(f"""
                    <div style="text-align:center; padding: 20px 0;">
                        <div class="verdict-badge {verdict_class}">{vd["label"]}</div>
                        <p style="color:#9090b0; max-width:600px; margin:auto; line-height:1.6;">
                            {vd["description"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Key metrics
                    st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)
                    m_cols = st.columns(4)
                    metric_data = [
                        ("Estimated Offset",
                         f"{sync_result.estimated_offset_ms:+.0f}" if sync_result.estimated_offset_ms is not None else "N/A",
                         "ms"),
                        ("Alignment Metric",
                         f"{sync_result.peak_correlation:.3f}" if sync_result.peak_correlation is not None else "N/A",
                         "correlation"),
                        ("Peak Quality",
                         f"{sync_result.peak_quality_ratio:.2f}" if sync_result.peak_quality_ratio is not None else "N/A",
                         "ratio"),
                        ("Signal Overlap",
                         f"{sync_result.overlap_duration_sec:.1f}",
                         "sec"),
                    ]
                    for col, (label, value, unit) in zip(m_cols, metric_data):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value}</div>
                                <div class="metric-unit">{unit}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Offset explanation
                    st.markdown("")
                    st.markdown(f"""
                    <div class="ss-info">
                        🕐 <b>Timing interpretation:</b> {offset_explanation(sync_result.estimated_offset_ms)}
                    </div>
                    """, unsafe_allow_html=True)

                    # Reason
                    st.markdown(f"""
                    <div class="ss-card">
                        <div class="section-title">🧠 Algorithm Reasoning</div>
                        <p style="color:#c0c0d8; line-height:1.7;">{sync_result.reason}</p>
                        <p style="font-size:0.8rem; color:#9090b0; margin-top:12px;">
                            <b>Sign convention:</b> Positive offset = audio leads video.
                            Negative offset = video leads audio.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Signal visualizations
                    st.markdown('<div class="section-title">📈 Signal Visualizations</div>', unsafe_allow_html=True)

                    sig_chart = build_signal_chart(sync_result)
                    corr_chart = build_correlation_chart(sync_result)

                    if sig_chart:
                        st.plotly_chart(sig_chart, use_container_width=True)
                    if corr_chart:
                        st.plotly_chart(corr_chart, use_container_width=True)

                    # Warnings from sync
                    for warn in sync_result.warnings:
                        st.markdown(f'<div class="ss-warning">⚠️ {warn}</div>', unsafe_allow_html=True)

                # Face detection summary
                st.markdown('<div class="section-title">👤 Face Detection Summary</div>', unsafe_allow_html=True)
                fd_cols = st.columns(3)
                with fd_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Detection Rate</div>
                        <div class="metric-value">{face_result.face_detection_rate*100:.0f}%</div>
                        <div class="metric-unit">frames with face</div>
                    </div>
                    """, unsafe_allow_html=True)
                with fd_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Frames Analyzed</div>
                        <div class="metric-value">{len(face_result.frame_analyses)}</div>
                        <div class="metric-unit">total frames</div>
                    </div>
                    """, unsafe_allow_html=True)
                with fd_cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Valid Aperture Points</div>
                        <div class="metric-value">{len(face_result.apertures)}</div>
                        <div class="metric-unit">data points</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ================================================================
                # Limitations
                # ================================================================
                st.markdown("")
                st.markdown("""
                <div class="ss-card">
                    <div class="section-title">⚖️ Limitations & Responsible Use</div>
                    <ul style="color:#c0c0d8; line-height:2; font-size:0.92rem;">
                        <li>This tool is an <b>auditor</b>, not a trained deepfake detector.</li>
                        <li>A mismatch result does <b>not</b> imply manipulation, deception, or malicious intent.</li>
                        <li>An aligned result does <b>not</b> guarantee authenticity.</li>
                        <li>Results depend heavily on face visibility, lighting, and audio quality.</li>
                        <li>Natural speaking variability, codec resampling, and post-production edits
                            can all create apparent offsets without manipulation.</li>
                        <li>Thresholds are configurable defaults — they are not universally validated.</li>
                        <li>The algorithm has only been validated on synthetic test signals.
                            Real-world performance requires additional evaluation.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            finally:
                # Always clean up uploaded temp file
                try:
                    if os.path.exists(tmp_video_path):
                        os.remove(tmp_video_path)
                except OSError:
                    pass

    else:
        # Landing state when no file uploaded
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#9090b0;">
            <div style="font-size:4rem; margin-bottom:16px;">🎬</div>
            <div style="font-size:1.1rem; font-weight:500; margin-bottom:10px;">
                Upload a video to begin analysis
            </div>
            <div style="font-size:0.9rem; max-width:500px; margin:auto; line-height:1.7;">
                The video should contain a clearly visible face and an audio track.
                Optimal results require good lighting, a front-facing camera angle,
                and clearly audible speech.
            </div>
            <div style="margin-top:24px;">
                <span class="tag">MP4</span>
                <span class="tag">MOV</span>
                <span class="tag">AVI</span>
                <span class="tag">MKV</span>
                <span class="tag">WEBM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — Evaluation
# ===========================================================================

with tab_evaluation:
    st.markdown("""
    <h2 style="background:linear-gradient(90deg,#7c6af0,#4fc3f7);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Algorithm Evaluation
    </h2>
    <p style="color:#9090b0; max-width:700px; line-height:1.7;">
        The synchronization algorithm is validated using synthetic signals with
        <b>known, controlled offsets</b>. Results below are computed by running the
        actual algorithm — no numbers are fabricated.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ss-warning">
        ⚠️ These synthetic tests validate the algorithm's cross-correlation mechanics.
        They do not establish performance on every real-world video, which depends on
        face detection quality, audio complexity, speaker variability, and many other factors.
    </div>
    """, unsafe_allow_html=True)

    run_eval = st.button("▶️ Run Evaluation Suite", use_container_width=False)

    if run_eval:
        with st.spinner("Running synthetic evaluation experiments…"):
            summary = run_evaluation()
            inconclusive_test = run_inconclusive_test()

        # Summary metrics
        st.markdown('<div class="section-title">📊 Summary Metrics</div>', unsafe_allow_html=True)
        s_cols = st.columns(4)
        s_metrics = [
            ("Experiments Run", str(len(summary.experiments)), "total"),
            ("Eligible Results",
             str(summary.eligible_count),
             f"{summary.inconclusive_count} inconclusive"),
            ("Mean Abs. Error",
             f"{summary.mae_ms:.1f}" if summary.mae_ms is not None else "N/A",
             "ms (eligible only)"),
            (f"Within ±{summary.tolerance_ms:.0f} ms",
             f"{summary.within_tolerance_pct:.0f}%" if summary.within_tolerance_pct is not None else "N/A",
             f"{summary.within_tolerance_count}/{summary.eligible_count}"),
        ]
        for col, (label, value, unit) in zip(s_cols, s_metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-unit">{unit}</div>
                </div>
                """, unsafe_allow_html=True)

        # Inconclusive test
        st.markdown("")
        inc_status = "✅ PASSED" if inconclusive_test["passed"] else "❌ FAILED"
        inc_color = "ss-success" if inconclusive_test["passed"] else "ss-error"
        st.markdown(f"""
        <div class="{inc_color}">
            <b>Flat Signal Test:</b> {inc_status} — {inconclusive_test["details"]}
        </div>
        """, unsafe_allow_html=True)

        # Experiment table
        st.markdown("")
        st.markdown('<div class="section-title">🔬 Experiment Results</div>', unsafe_allow_html=True)

        # Render custom styled HTML table to avoid pyarrow/NumPy 2.x dependency issues
        headers = ["True Offset (ms)", "Estimated (ms)", "Error (ms)", "Within Tolerance", "Peak Quality", "Verdict"]
        table_html = [
            '<div style="overflow-x: auto; margin: 16px 0; border-radius: 12px; border: 1px solid var(--border);">',
            '<table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: left; background: var(--surface2);">',
            '  <thead>',
            '    <tr style="background: rgba(124, 106, 240, 0.15); border-bottom: 2px solid var(--border); color: var(--accent2);">'
        ]
        for h in headers:
            table_html.append(f'<th style="padding: 12px 16px; font-weight: 600;">{h}</th>')
        table_html.append('</tr></thead><tbody>')

        for exp in summary.experiments:
            verdict_str = exp.verdict.value
            true_off = f"{exp.true_offset_ms:+.0f}"
            est_off = f"{exp.estimated_offset_ms:+.0f}" if exp.estimated_offset_ms is not None else "—"
            err = f"{exp.error_ms:.0f}" if exp.error_ms is not None else "—"
            tol = "✅" if exp.within_tolerance else ("—" if exp.within_tolerance is None else "❌")
            pq = f"{exp.peak_quality_ratio:.2f}" if exp.peak_quality_ratio is not None else "—"
            v_color = "#66bb6a" if verdict_str == "ALIGNED" else ("#ef5350" if verdict_str == "MISMATCH" else "#ffa726")

            table_html.append(f'''
            <tr style="border-bottom: 1px solid var(--border);">
              <td style="padding: 10px 16px; font-weight: 500; color: #fff;">{true_off}</td>
              <td style="padding: 10px 16px; color: var(--text);">{est_off}</td>
              <td style="padding: 10px 16px; color: var(--text);">{err}</td>
              <td style="padding: 10px 16px;">{tol}</td>
              <td style="padding: 10px 16px; color: var(--text);">{pq}</td>
              <td style="padding: 10px 16px; font-weight: 600; color: {v_color};">{verdict_str}</td>
            </tr>
            ''')
        table_html.append('</tbody></table></div>')
        st.markdown("".join(table_html), unsafe_allow_html=True)

        # Chart
        eval_chart = build_evaluation_chart(summary)
        if eval_chart:
            st.plotly_chart(eval_chart, use_container_width=True)

        # Notes
        st.markdown(f"""
        <div class="ss-info">
            ℹ️ {summary.notes}
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 3 — About
# ===========================================================================

with tab_about:
    st.markdown("""
    <h2 style="background:linear-gradient(90deg,#7c6af0,#4fc3f7);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        About SpeakSense
    </h2>
    """, unsafe_allow_html=True)

    cols = st.columns(2)

    with cols[0]:
        st.markdown("""
        <div class="ss-card">
            <div class="section-title">🎯 What It Does</div>
            <p style="color:#c0c0d8; line-height:1.8; font-size:0.92rem;">
                SpeakSense analyzes a video of a speaking person and estimates whether
                the visible mouth movements are temporally aligned with the audio track.
                It uses computer vision to measure mouth aperture and signal processing
                to measure audio energy, then applies cross-correlation to estimate
                any timing offset.
            </p>
        </div>

        <div class="ss-card">
            <div class="section-title">🔧 Technology Stack</div>
            <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">Python</td>
                    <td style="padding:8px; color:#c0c0d8;">Core language</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">Streamlit</td>
                    <td style="padding:8px; color:#c0c0d8;">Complete UI</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">OpenCV</td>
                    <td style="padding:8px; color:#c0c0d8;">Frame reading</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">MediaPipe</td>
                    <td style="padding:8px; color:#c0c0d8;">FaceMesh landmarks</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">Librosa</td>
                    <td style="padding:8px; color:#c0c0d8;">Audio features (STE)</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">FFmpeg</td>
                    <td style="padding:8px; color:#c0c0d8;">Audio extraction</td>
                </tr>
                <tr style="border-bottom:1px solid #2a2a4a;">
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">NumPy / SciPy</td>
                    <td style="padding:8px; color:#c0c0d8;">Signal processing</td>
                </tr>
                <tr>
                    <td style="padding:8px; color:#7c6af0; font-weight:600;">Plotly</td>
                    <td style="padding:8px; color:#c0c0d8;">Interactive charts</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
        <div class="ss-card">
            <div class="section-title">⚙️ Pipeline Flow</div>
            <ol style="color:#c0c0d8; line-height:2.2; font-size:0.9rem;">
                <li><b>Upload & Validate</b> — format, size, readability checks</li>
                <li><b>Frame Sampling</b> — uniform interval sampling via OpenCV</li>
                <li><b>Audio Extraction</b> — FFmpeg → WAV → Librosa STE</li>
                <li><b>Face Analysis</b> — MediaPipe FaceMesh → mouth aperture</li>
                <li><b>Resampling</b> — both signals to 25 Hz common grid</li>
                <li><b>Normalization</b> — Gaussian smoothing + zero-mean, unit-std</li>
                <li><b>Cross-Correlation</b> — full normalized correlation</li>
                <li><b>Peak Analysis</b> — offset estimation + quality ratio</li>
                <li><b>Verdict</b> — ALIGNED / MISMATCH / INCONCLUSIVE</li>
            </ol>
        </div>

        <div class="ss-card">
            <div class="section-title">🔒 Privacy</div>
            <ul style="color:#c0c0d8; line-height:2; font-size:0.9rem;">
                <li>Videos are processed locally in temporary files</li>
                <li>Uploaded files are deleted immediately after analysis</li>
                <li>No video data is sent to external services</li>
                <li>No personal media is stored or logged</li>
            </ul>
        </div>

        <div class="ss-card">
            <div class="section-title">📚 Documentation</div>
            <ul style="color:#c0c0d8; line-height:2; font-size:0.9rem;">
                <li><code>docs/algorithm_explained.md</code> — how it works</li>
                <li><code>docs/architecture.md</code> — module map</li>
                <li><code>docs/interview_guide.md</code> — 30+ Q&amp;A for interviews</li>
                <li><code>tests/</code> — automated pytest test suite</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
