"""Blood-pressure proxy estimation from pulse waveform features.

True cuff-less blood-pressure measurement from camera video alone is an
open research problem.  This module provides a *proxy* estimate based on
pulse-wave features that correlate with BP:

* **Pulse Transit Time (PTT)** -- approximated here as the time between
  the foot (onset) and the systolic peak of the rPPG waveform.
* **Augmentation Index (AI)** -- ratio of the reflected-wave amplitude to
  the primary systolic peak.
* **Heart Rate** -- higher resting HR is weakly associated with higher BP.

The default regression coefficients are *illustrative* and MUST be
calibrated per-user against a reference sphygmomanometer for clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import argrelmin

from prana.models import ConfidenceInterval


@dataclass
class BPEstimator:
    """Estimate systolic and diastolic blood pressure from rPPG features.

    Parameters:
        sbp_intercept: Baseline systolic BP (mmHg) for the regression.
        dbp_intercept: Baseline diastolic BP (mmHg).
        ptt_coeff: Coefficient mapping PTT (ms) to BP change.
        hr_coeff: Coefficient mapping HR (bpm) to BP change.
    """

    sbp_intercept: float = 120.0
    dbp_intercept: float = 80.0
    ptt_coeff: float = -0.5
    hr_coeff: float = 0.15

    def estimate(
        self,
        signal: np.ndarray,
        peak_indices: np.ndarray,
        ibi_seconds: np.ndarray,
        fps: float,
    ) -> tuple[ConfidenceInterval, ConfidenceInterval]:
        """Compute systolic and diastolic BP proxy values.

        Parameters:
            signal: Cleaned rPPG waveform.
            peak_indices: Systolic peak locations.
            ibi_seconds: Inter-beat intervals.
            fps: Sampling rate.

        Returns:
            ``(systolic_ci, diastolic_ci)``
        """
        if len(peak_indices) < 3 or len(ibi_seconds) < 2:
            return (
                ConfidenceInterval(value=0, lower=0, upper=0, unit="mmHg"),
                ConfidenceInterval(value=0, lower=0, upper=0, unit="mmHg"),
            )

        ptt_ms = self._mean_ptt(signal, peak_indices, fps)
        mean_hr = 60.0 / np.mean(ibi_seconds) if np.mean(ibi_seconds) > 0 else 75

        sbp = self.sbp_intercept + self.ptt_coeff * ptt_ms + self.hr_coeff * mean_hr
        dbp = self.dbp_intercept + self.ptt_coeff * ptt_ms * 0.6 + self.hr_coeff * mean_hr * 0.5

        sbp = float(np.clip(sbp, 80, 200))
        dbp = float(np.clip(dbp, 50, 130))

        # Wide CI to reflect the inherently limited accuracy.
        sbp_width = 12.0
        dbp_width = 8.0

        sbp_ci = ConfidenceInterval(
            value=round(sbp, 0),
            lower=round(sbp - sbp_width, 0),
            upper=round(sbp + sbp_width, 0),
            unit="mmHg",
        )
        dbp_ci = ConfidenceInterval(
            value=round(dbp, 0),
            lower=round(dbp - dbp_width, 0),
            upper=round(dbp + dbp_width, 0),
            unit="mmHg",
        )
        return sbp_ci, dbp_ci

    @staticmethod
    def _mean_ptt(
        signal: np.ndarray,
        peak_indices: np.ndarray,
        fps: float,
    ) -> float:
        """Mean pulse transit time approximation (onset-to-peak) in ms."""
        ptts: list[float] = []
        for pk in peak_indices:
            # Search backwards from the peak for the preceding local minimum
            # (pulse onset / foot).
            search_start = max(0, pk - int(fps * 0.4))
            segment = signal[search_start : pk]
            if len(segment) < 3:
                continue
            local_mins = argrelmin(segment, order=2)[0]
            if len(local_mins) == 0:
                # Fall back to absolute minimum.
                foot_local = int(np.argmin(segment))
            else:
                foot_local = int(local_mins[-1])
            foot_global = search_start + foot_local
            ptt_samples = pk - foot_global
            ptts.append(ptt_samples / fps * 1000.0)

        return float(np.mean(ptts)) if ptts else 200.0
