"""Heart-rate variability analysis and stress-level estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prana.models import ConfidenceInterval, StressLevel


@dataclass
class HRVAnalyzer:
    """Compute HRV time-domain metrics and infer a stress level.

    Parameters:
        sdnn_stress_low: SDNN threshold (ms) below which stress is HIGH.
        sdnn_stress_high: SDNN threshold (ms) above which stress is LOW.
        rmssd_stress_low: RMSSD threshold (ms) below which stress is HIGH.
        rmssd_stress_high: RMSSD threshold (ms) above which stress is LOW.
    """

    sdnn_stress_low: float = 30.0
    sdnn_stress_high: float = 60.0
    rmssd_stress_low: float = 20.0
    rmssd_stress_high: float = 45.0

    # ------------------------------------------------------------------
    # Time-domain metrics
    # ------------------------------------------------------------------

    @staticmethod
    def sdnn(ibi_ms: np.ndarray) -> float:
        """Standard deviation of NN (normal-to-normal) intervals in ms."""
        if len(ibi_ms) < 2:
            return 0.0
        return float(np.std(ibi_ms, ddof=1))

    @staticmethod
    def rmssd(ibi_ms: np.ndarray) -> float:
        """Root mean square of successive differences (ms)."""
        if len(ibi_ms) < 2:
            return 0.0
        diffs = np.diff(ibi_ms)
        return float(np.sqrt(np.mean(diffs ** 2)))

    @staticmethod
    def pnn50(ibi_ms: np.ndarray) -> float:
        """Percentage of successive differences > 50 ms."""
        if len(ibi_ms) < 2:
            return 0.0
        diffs = np.abs(np.diff(ibi_ms))
        return float(np.sum(diffs > 50) / len(diffs) * 100)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def analyze(
        self, ibi_seconds: np.ndarray
    ) -> tuple[ConfidenceInterval, ConfidenceInterval, StressLevel]:
        """Compute SDNN, RMSSD and stress level from IBI series.

        Parameters:
            ibi_seconds: Inter-beat intervals in **seconds**.

        Returns:
            ``(sdnn_ci, rmssd_ci, stress_level)``
        """
        ibi_ms = ibi_seconds * 1000.0

        sdnn_val = self.sdnn(ibi_ms)
        rmssd_val = self.rmssd(ibi_ms)

        # Bootstrap-style rough CI: use standard error of the metric.
        n = len(ibi_ms)
        se_factor = 1.96 / max(np.sqrt(n), 1)

        sdnn_ci = ConfidenceInterval(
            value=round(sdnn_val, 1),
            lower=round(max(sdnn_val * (1 - se_factor), 0), 1),
            upper=round(sdnn_val * (1 + se_factor), 1),
            unit="ms",
        )
        rmssd_ci = ConfidenceInterval(
            value=round(rmssd_val, 1),
            lower=round(max(rmssd_val * (1 - se_factor), 0), 1),
            upper=round(rmssd_val * (1 + se_factor), 1),
            unit="ms",
        )

        stress = self._stress_level(sdnn_val, rmssd_val)
        return sdnn_ci, rmssd_ci, stress

    def _stress_level(self, sdnn_val: float, rmssd_val: float) -> StressLevel:
        score = 0
        if sdnn_val < self.sdnn_stress_low:
            score += 2
        elif sdnn_val < self.sdnn_stress_high:
            score += 1

        if rmssd_val < self.rmssd_stress_low:
            score += 2
        elif rmssd_val < self.rmssd_stress_high:
            score += 1

        if score >= 3:
            return StressLevel.HIGH
        elif score >= 1:
            return StressLevel.MODERATE
        return StressLevel.LOW
