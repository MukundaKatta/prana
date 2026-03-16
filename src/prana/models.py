"""Pydantic domain models for the Prana pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class _NumpyArbitraryConfig(BaseModel):
    """Base that allows numpy arrays in pydantic v2 models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------


class ROI(_NumpyArbitraryConfig):
    """Region-of-interest extracted from a video frame.

    Attributes:
        label: Human-readable name (e.g. ``"forehead"``, ``"left_cheek"``).
        x, y, w, h: Bounding box in pixel coordinates.
        pixels: Raw pixel values inside the ROI, shape ``(H, W, 3)`` BGR.
    """

    label: str
    x: int
    y: int
    w: int
    h: int
    pixels: np.ndarray


class VideoFrame(_NumpyArbitraryConfig):
    """A single frame grabbed from the video source.

    Attributes:
        index: Zero-based frame counter.
        timestamp_s: Wall-clock time since recording start, in seconds.
        image: The raw BGR image as a numpy array ``(H, W, 3)``.
        rois: Skin ROIs detected in this frame (populated by ``FaceDetector``).
    """

    index: int
    timestamp_s: float
    image: np.ndarray
    rois: list[ROI] = Field(default_factory=list)


class PPGSignal(_NumpyArbitraryConfig):
    """Remote-PPG waveform extracted from a sequence of video frames.

    Attributes:
        signal: 1-D array of the cleaned rPPG amplitude values.
        timestamps: Corresponding time stamps in seconds.
        fps: Effective sampling rate after extraction.
        channel: Which extraction was used (``"POS"``, ``"CHROM"``, etc.).
    """

    signal: np.ndarray
    timestamps: np.ndarray
    fps: float
    channel: str = "POS"


class ConfidenceInterval(BaseModel):
    """A point estimate with a symmetric confidence band."""

    value: float
    lower: float
    upper: float
    unit: str


class StressLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class VitalSigns(BaseModel):
    """Complete set of vital signs produced by a single measurement session."""

    heart_rate_bpm: ConfidenceInterval
    hrv_sdnn_ms: ConfidenceInterval
    hrv_rmssd_ms: ConfidenceInterval
    stress_level: StressLevel
    respiratory_rate_brpm: ConfidenceInterval
    spo2_percent: ConfidenceInterval
    systolic_bp_mmhg: ConfidenceInterval
    diastolic_bp_mmhg: ConfidenceInterval
    quality_score: float = Field(ge=0.0, le=1.0)


class MeasurementSession(BaseModel):
    """Metadata and results for one measurement run."""

    session_id: str
    started_at: datetime
    duration_s: float
    frame_count: int
    fps: float
    source: str
    vitals: Optional[VitalSigns] = None
