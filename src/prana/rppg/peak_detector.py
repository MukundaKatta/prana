"""Pulse-peak detection from a cleaned rPPG signal.

Locates systolic peaks so that inter-beat intervals (IBI) can be computed
for heart-rate and HRV analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


@dataclass
class PeakDetector:
    """Detect pulse peaks in an rPPG waveform.

    Parameters:
        min_hr_bpm: Minimum plausible heart rate (sets maximum peak spacing).
        max_hr_bpm: Maximum plausible heart rate (sets minimum peak spacing).
        prominence_factor: Fraction of signal amplitude used as the minimum
            peak prominence.
    """

    min_hr_bpm: float = 40.0
    max_hr_bpm: float = 200.0
    prominence_factor: float = 0.3

    def detect(
        self,
        signal: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find peaks and return their indices and the inter-beat intervals.

        Parameters:
            signal: 1-D cleaned rPPG waveform.
            fps: Sampling rate in Hz.

        Returns:
            ``(peak_indices, ibi_seconds)`` where ``ibi_seconds`` has length
            ``len(peak_indices) - 1``.
        """
        min_distance = int(fps * 60.0 / self.max_hr_bpm)
        max_distance = int(fps * 60.0 / self.min_hr_bpm)

        # Adaptive prominence: fraction of peak-to-trough range.
        amp = np.ptp(signal)
        prominence = max(amp * self.prominence_factor, 1e-6)

        peaks, properties = find_peaks(
            signal,
            distance=max(min_distance, 1),
            prominence=prominence,
        )

        # Remove peaks that are too far apart (likely false positives around
        # edges) -- keep only those within plausible IBI range.
        if len(peaks) > 1:
            ibi_samples = np.diff(peaks)
            valid_mask = (ibi_samples >= min_distance) & (ibi_samples <= max_distance)
            # Keep peaks that border at least one valid interval.
            keep = np.zeros(len(peaks), dtype=bool)
            keep[:-1] |= valid_mask
            keep[1:] |= valid_mask
            if keep.sum() >= 2:
                peaks = peaks[keep]

        ibi_seconds = np.diff(peaks) / fps
        return peaks, ibi_seconds

    def signal_quality(self, ibi_seconds: np.ndarray) -> float:
        """Heuristic signal-quality score in ``[0, 1]``.

        Based on the coefficient of variation of the IBI series -- a clean
        resting signal should have low variability.
        """
        if len(ibi_seconds) < 2:
            return 0.0
        cv = np.std(ibi_seconds) / np.mean(ibi_seconds)
        # Map CV in [0, 0.5] to quality in [1, 0].
        quality = float(np.clip(1.0 - 2.0 * cv, 0.0, 1.0))
        return quality
