"""
reporting.py
------------
Builds visualizations and formats result summaries for display in Streamlit.

All charts use actual computed values from the analysis pipeline.
No hardcoded or fabricated data is ever displayed.
"""

import numpy as np
from typing import Optional, List

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.synchronization import SyncResult, SyncVerdict
from src.evaluation import EvaluationSummary, ExperimentResult


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "mouth": "#4FC3F7",       # sky blue
    "audio": "#FFB74D",       # warm amber
    "aligned": "#66BB6A",     # green
    "mismatch": "#EF5350",    # red
    "inconclusive": "#FFA726", # orange
    "corr": "#CE93D8",        # light purple
    "peak": "#F06292",        # pink accent
    "grid": "#2A2A3E",        # dark grid
    "background": "#0F0F23",  # near-black
    "surface": "#1A1A2E",
    "text": "#E0E0E0",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["surface"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#2A2A3E", zeroline=False),
    yaxis=dict(gridcolor="#2A2A3E", zeroline=False),
    margin=dict(l=50, r=30, t=50, b=50),
)


# ---------------------------------------------------------------------------
# Signal timeline chart
# ---------------------------------------------------------------------------

def build_signal_chart(sync_result: SyncResult) -> Optional[object]:
    """
    Build a Plotly figure showing the normalized mouth-aperture and audio-energy
    signals on the same time axis.
    Returns None if Plotly is unavailable or signals are missing.
    """
    if not PLOTLY_AVAILABLE:
        return None
    if sync_result.common_timestamps is None:
        return None

    t = sync_result.common_timestamps
    mouth = sync_result.mouth_signal_normalized
    audio = sync_result.audio_signal_normalized

    fig = go.Figure()

    if mouth is not None:
        fig.add_trace(go.Scatter(
            x=t, y=mouth,
            name="Mouth Aperture (normalized)",
            line=dict(color=COLORS["mouth"], width=2),
            mode="lines",
        ))

    if audio is not None:
        fig.add_trace(go.Scatter(
            x=t, y=audio,
            name="Audio Energy (normalized)",
            line=dict(color=COLORS["audio"], width=2),
            mode="lines",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Normalized Signal Comparison",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized Value",
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="#444",
            borderwidth=1,
        ),
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Cross-correlation chart
# ---------------------------------------------------------------------------

def build_correlation_chart(sync_result: SyncResult) -> Optional[object]:
    """
    Build a Plotly figure showing the full cross-correlation function and
    marking the estimated peak lag.
    """
    if not PLOTLY_AVAILABLE:
        return None
    if sync_result.correlation is None or sync_result.lags_ms is None:
        return None

    corr = sync_result.correlation
    lags = sync_result.lags_ms
    peak_ms = sync_result.estimated_offset_ms

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lags, y=corr,
        name="Cross-Correlation",
        line=dict(color=COLORS["corr"], width=1.5),
        mode="lines",
    ))

    if peak_ms is not None:
        fig.add_vline(
            x=peak_ms,
            line=dict(color=COLORS["peak"], width=2, dash="dash"),
            annotation_text=f"Peak: {peak_ms:+.0f} ms",
            annotation_font_color=COLORS["peak"],
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Cross-Correlation Function",
        xaxis_title="Lag (milliseconds)",
        yaxis_title="Correlation",
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Evaluation chart
# ---------------------------------------------------------------------------

def build_evaluation_chart(summary: EvaluationSummary) -> Optional[object]:
    """
    Build a Plotly figure showing true vs. estimated offsets for all experiments.
    """
    if not PLOTLY_AVAILABLE:
        return None

    experiments = summary.experiments
    true_offsets = [e.true_offset_ms for e in experiments]
    est_offsets = [e.estimated_offset_ms for e in experiments]
    verdicts = [e.verdict for e in experiments]

    # Colour by verdict
    colors_per_exp = []
    for v in verdicts:
        if v == SyncVerdict.ALIGNED:
            colors_per_exp.append(COLORS["aligned"])
        elif v == SyncVerdict.MISMATCH:
            colors_per_exp.append(COLORS["mismatch"])
        else:
            colors_per_exp.append(COLORS["inconclusive"])

    fig = go.Figure()

    # Diagonal ideal line
    if true_offsets:
        lo = min(true_offsets) - 50
        hi = max(true_offsets) + 50
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color="#555", dash="dot", width=1),
            name="Ideal (no error)",
            showlegend=True,
        ))

    # Tolerance band
    fig.add_hrect(
        y0=-summary.tolerance_ms, y1=summary.tolerance_ms,
        fillcolor="rgba(100,200,100,0.07)",
        line_width=0,
        annotation_text=f"±{summary.tolerance_ms:.0f} ms tolerance",
        annotation_font_color="#888",
    )

    # Scatter points
    valid_true = [t for t, e in zip(true_offsets, est_offsets) if e is not None]
    valid_est = [e for e in est_offsets if e is not None]
    valid_colors = [c for c, e in zip(colors_per_exp, est_offsets) if e is not None]

    fig.add_trace(go.Scatter(
        x=valid_true,
        y=valid_est,
        mode="markers",
        marker=dict(color=valid_colors, size=12, line=dict(color="#fff", width=1)),
        name="Experiments",
        text=[f"True: {t:.0f} ms<br>Est: {e:.0f} ms" for t, e in zip(valid_true, valid_est)],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Synthetic Evaluation: True vs. Estimated Offset",
        xaxis_title="True Offset (ms)",
        yaxis_title="Estimated Offset (ms)",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Verdict badge helper
# ---------------------------------------------------------------------------

def verdict_display(result: SyncResult) -> dict:
    """
    Return a dict with display strings and colour for the verdict.
    Used by the Streamlit app.
    """
    if result.verdict == SyncVerdict.ALIGNED:
        return {
            "label": "✅ LIKELY ALIGNED",
            "color": COLORS["aligned"],
            "description": "The estimated timing offset is within the configured tolerance.",
        }
    elif result.verdict == SyncVerdict.MISMATCH:
        return {
            "label": "⚠️ POSSIBLE MISMATCH",
            "color": COLORS["mismatch"],
            "description": "The estimated timing offset exceeds the configured tolerance.",
        }
    else:
        return {
            "label": "❓ INCONCLUSIVE",
            "color": COLORS["inconclusive"],
            "description": "Insufficient signal quality to make a reliable determination.",
        }


# ---------------------------------------------------------------------------
# Offset direction explanation
# ---------------------------------------------------------------------------

def offset_explanation(offset_ms: Optional[float]) -> str:
    """Return a human-readable explanation of what the estimated offset means."""
    if offset_ms is None:
        return "No offset could be estimated."
    abs_ms = abs(offset_ms)
    if abs_ms < 10:
        return f"The estimated offset is approximately {abs_ms:.0f} ms — essentially zero."
    elif offset_ms > 0:
        return (
            f"The audio signal appears to lead the mouth movements by approximately "
            f"{abs_ms:.0f} ms. This means audio events occur about {abs_ms:.0f} ms "
            f"before the corresponding mouth opening."
        )
    else:
        return (
            f"The mouth movements appear to lead the audio by approximately "
            f"{abs_ms:.0f} ms. This means the mouth opens about {abs_ms:.0f} ms "
            f"before the corresponding audio event."
        )
