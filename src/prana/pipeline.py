"""End-to-end VitalsPipeline: video -> face -> rPPG -> five vital signs."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from prana.models import (
    MeasurementSession,
    PPGSignal,
    ROI,
    VideoFrame,
    VitalSigns,
)
from prana.rppg.face_detector import FaceDetector
from prana.rppg.filters import bandpass_filter, remove_motion_artifacts
from prana.rppg.peak_detector import PeakDetector
from prana.rppg.signal_extractor import Algorithm, SignalExtractor
from prana.vitals.blood_pressure import BPEstimator
from prana.vitals.heart_rate import HeartRateEstimator
from prana.vitals.hrv import HRVAnalyzer
from prana.vitals.respiratory import RespiratoryEstimator
from prana.vitals.spo2 import SpO2Estimator


@dataclass
class VitalsPipeline:
    """Orchestrate the full measurement pipeline.

    Parameters:
        source: Video source -- an integer (webcam index), a file path, or
            the literal string ``"webcam"`` (equivalent to 0).
        duration_s: Recording length in seconds (ignored for files).
        algorithm: rPPG extraction algorithm.
        window_size: Sliding-window length for POS/CHROM.
    """

    source: str | int = 0
    duration_s: float = 30.0
    algorithm: Algorithm = Algorithm.POS
    window_size: int = 32

    # Sub-components (created on first run).
    _face_detector: FaceDetector = field(default_factory=FaceDetector, init=False)
    _signal_extractor: SignalExtractor = field(default=None, init=False)  # type: ignore[assignment]
    _peak_detector: PeakDetector = field(default_factory=PeakDetector, init=False)
    _hr_estimator: HeartRateEstimator = field(default_factory=HeartRateEstimator, init=False)
    _hrv_analyzer: HRVAnalyzer = field(default_factory=HRVAnalyzer, init=False)
    _rr_estimator: RespiratoryEstimator = field(default_factory=RespiratoryEstimator, init=False)
    _spo2_estimator: SpO2Estimator = field(default_factory=SpO2Estimator, init=False)
    _bp_estimator: BPEstimator = field(default_factory=BPEstimator, init=False)

    def __post_init__(self) -> None:
        self._signal_extractor = SignalExtractor(
            algorithm=self.algorithm, window_size=self.window_size
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, show_preview: bool = False) -> MeasurementSession:
        """Execute the full pipeline: capture -> extract -> estimate.

        Parameters:
            show_preview: If ``True``, display a live OpenCV window with
                ROI overlays during capture.

        Returns:
            A ``MeasurementSession`` populated with vital signs.
        """
        session_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(tz=timezone.utc)

        frames, fps = self._capture(show_preview=show_preview)

        if not frames:
            return MeasurementSession(
                session_id=session_id,
                started_at=started_at,
                duration_s=0.0,
                frame_count=0,
                fps=0.0,
                source=str(self.source),
            )

        roi_series = [f.rois for f in frames]
        timestamps = np.array([f.timestamp_s for f in frames])
        duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

        # rPPG extraction.
        ppg = self._signal_extractor.extract(roi_series, timestamps, fps)

        # Filtering.
        ppg.signal = remove_motion_artifacts(ppg.signal, fps)
        ppg.signal = bandpass_filter(ppg.signal, fps)

        # Peak detection.
        peaks, ibi = self._peak_detector.detect(ppg.signal, fps)
        quality = self._peak_detector.signal_quality(ibi)

        # Vital-sign estimation.
        hr_ci = self._hr_estimator.estimate(ppg)
        sdnn_ci, rmssd_ci, stress = self._hrv_analyzer.analyze(ibi)
        rr_ci = self._rr_estimator.estimate(ppg, peaks, ibi)
        spo2_ci = self._spo2_estimator.estimate(roi_series, fps)
        sbp_ci, dbp_ci = self._bp_estimator.estimate(ppg.signal, peaks, ibi, fps)

        vitals = VitalSigns(
            heart_rate_bpm=hr_ci,
            hrv_sdnn_ms=sdnn_ci,
            hrv_rmssd_ms=rmssd_ci,
            stress_level=stress,
            respiratory_rate_brpm=rr_ci,
            spo2_percent=spo2_ci,
            systolic_bp_mmhg=sbp_ci,
            diastolic_bp_mmhg=dbp_ci,
            quality_score=quality,
        )

        return MeasurementSession(
            session_id=session_id,
            started_at=started_at,
            duration_s=duration,
            frame_count=len(frames),
            fps=fps,
            source=str(self.source),
            vitals=vitals,
        )

    # ------------------------------------------------------------------
    # Video capture
    # ------------------------------------------------------------------

    def _capture(
        self, show_preview: bool = False
    ) -> tuple[list[VideoFrame], float]:
        """Grab frames from the video source, detecting faces in each."""
        src = 0 if self.source == "webcam" else self.source
        cap = cv2.VideoCapture(src)  # type: ignore[arg-type]
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frames: list[VideoFrame] = []
        start = time.monotonic()

        try:
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                elapsed = time.monotonic() - start

                # For file sources, honour duration limit too.
                if isinstance(self.source, str) and self.source != "webcam":
                    pass  # read entire file
                elif elapsed > self.duration_s:
                    break

                frame = VideoFrame(
                    index=len(frames),
                    timestamp_s=elapsed,
                    image=image,
                )
                self._face_detector.detect(frame)
                frames.append(frame)

                if show_preview:
                    self._draw_preview(image, frame.rois)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            self._face_detector.close()

        return frames, fps

    @staticmethod
    def _draw_preview(image: np.ndarray, rois: list[ROI]) -> None:
        display = image.copy()
        for roi in rois:
            cv2.rectangle(
                display,
                (roi.x, roi.y),
                (roi.x + roi.w, roi.y + roi.h),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display,
                roi.label,
                (roi.x, roi.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        cv2.imshow("Prana - press q to stop", display)

    # ------------------------------------------------------------------
    # Convenience: run from raw numpy frames (for testing)
    # ------------------------------------------------------------------

    def run_from_frames(
        self,
        images: list[np.ndarray],
        fps: float = 30.0,
    ) -> MeasurementSession:
        """Run the pipeline on pre-loaded frames (no camera needed)."""
        session_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(tz=timezone.utc)

        frames: list[VideoFrame] = []
        for i, img in enumerate(images):
            frame = VideoFrame(
                index=i,
                timestamp_s=i / fps,
                image=img,
            )
            self._face_detector.detect(frame)
            frames.append(frame)

        self._face_detector.close()

        roi_series = [f.rois for f in frames]
        timestamps = np.array([f.timestamp_s for f in frames])

        ppg = self._signal_extractor.extract(roi_series, timestamps, fps)
        ppg.signal = remove_motion_artifacts(ppg.signal, fps)
        ppg.signal = bandpass_filter(ppg.signal, fps)

        peaks, ibi = self._peak_detector.detect(ppg.signal, fps)
        quality = self._peak_detector.signal_quality(ibi)

        hr_ci = self._hr_estimator.estimate(ppg)
        sdnn_ci, rmssd_ci, stress = self._hrv_analyzer.analyze(ibi)
        rr_ci = self._rr_estimator.estimate(ppg, peaks, ibi)
        spo2_ci = self._spo2_estimator.estimate(roi_series, fps)
        sbp_ci, dbp_ci = self._bp_estimator.estimate(ppg.signal, peaks, ibi, fps)

        vitals = VitalSigns(
            heart_rate_bpm=hr_ci,
            hrv_sdnn_ms=sdnn_ci,
            hrv_rmssd_ms=rmssd_ci,
            stress_level=stress,
            respiratory_rate_brpm=rr_ci,
            spo2_percent=spo2_ci,
            systolic_bp_mmhg=sbp_ci,
            diastolic_bp_mmhg=dbp_ci,
            quality_score=quality,
        )

        duration = float(timestamps[-1]) if len(timestamps) else 0.0
        return MeasurementSession(
            session_id=session_id,
            started_at=started_at,
            duration_s=duration,
            frame_count=len(frames),
            fps=fps,
            source="memory",
            vitals=vitals,
        )
