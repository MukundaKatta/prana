"""Respiratory-rate estimation from rPPG pulse modulation.

Breathing modulates the pulse waveform in three ways:

1. **Baseline wander** -- the DC level of the rPPG signal oscillates at the
   respiratory frequency.
2. **Amplitude modulation** -- pulse amplitude varies with inspiration /
   expiration.
3. **Frequency modulation** -- IBI varies slightly with respiration
   (respiratory sinus arrhythmia, RSA).

This module extracts all three modulatory signals and fuses them via
spectral voting to estimate breathing rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, welch

from prana.models import ConfidenceInterval, PPGSignal


@dataclass
class RespiratoryEstimator:
    """Estimate respiratory rate from the rPPG signal.

    Parameters:
        min_rr_brpm: Minimum plausible breathing rate (breaths/min).
        max_rr_brpm: Maximum plausible breathing rate.
    """

    min_rr_brpm: float = 6.0
    max_rr_brpm: float = 40.0

    def estimate(
        self,
        ppg: PPGSignal,
        peak_indices: np.ndarray,
        ibi_seconds: np.ndarray,
    ) -> ConfidenceInterval:
        """Fuse baseline-wander, amplitude-modulation and frequency-modulation
        respiratory estimates.

        Parameters:
            ppg: The cleaned rPPG signal.
            peak_indices: Pulse peak locations (sample indices).
            ibi_seconds: Inter-beat intervals in seconds.

        Returns:
            Respiratory rate with confidence interval, in breaths per minute.
        """
        estimates: list[float] = []

        bw = self._baseline_wander(ppg.signal, ppg.fps)
        if bw is not None:
            estimates.append(bw)

        am = self._amplitude_modulation(ppg.signal, peak_indices, ppg.fps)
        if am is not None:
            estimates.append(am)

        fm = self._frequency_modulation(ibi_seconds)
        if fm is not None:
            estimates.append(fm)

        if not estimates:
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="brpm")

        median_rr = float(np.median(estimates))
        spread = float(np.std(estimates)) if len(estimates) > 1 else 2.0
        return ConfidenceInterval(
            value=round(median_rr, 1),
            lower=round(max(median_rr - spread, self.min_rr_brpm), 1),
            upper=round(min(median_rr + spread, self.max_rr_brpm), 1),
            unit="brpm",
        )

    # ------------------------------------------------------------------
    # Modulatory signal extraction
    # ------------------------------------------------------------------

    def _baseline_wander(self, signal: np.ndarray, fps: float) -> float | None:
        """Extract respiratory frequency from the low-frequency baseline."""
        low = self.min_rr_brpm / 60.0
        high = self.max_rr_brpm / 60.0
        nyq = fps / 2.0
        if high >= nyq:
            high = nyq - 0.01
        if low >= high:
            return None
        b, a = butter(3, [low / nyq, high / nyq], btype="band")
        pad = min(3 * max(len(b), len(a)), len(signal) - 1)
        if pad < 1:
            return None
        filtered = filtfilt(b, a, signal, padlen=pad)
        return self._dominant_freq_bpm(filtered, fps)

    def _amplitude_modulation(
        self, signal: np.ndarray, peaks: np.ndarray, fps: float
    ) -> float | None:
        if len(peaks) < 4:
            return None
        amplitudes = signal[peaks]
        # Resample amplitudes at uniform spacing using linear interpolation.
        peak_times = peaks / fps
        uniform_times = np.linspace(peak_times[0], peak_times[-1], len(peaks) * 2)
        uniform_amps = np.interp(uniform_times, peak_times, amplitudes)
        effective_fs = len(uniform_amps) / (uniform_times[-1] - uniform_times[0])
        return self._dominant_freq_bpm(uniform_amps, effective_fs)

    def _frequency_modulation(self, ibi_seconds: np.ndarray) -> float | None:
        if len(ibi_seconds) < 4:
            return None
        # IBI series is already sampled once per beat; approximate fs.
        mean_ibi = np.mean(ibi_seconds)
        ibi_fs = 1.0 / mean_ibi if mean_ibi > 0 else 1.0
        return self._dominant_freq_bpm(ibi_seconds, ibi_fs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dominant_freq_bpm(self, signal: np.ndarray, fs: float) -> float | None:
        """Welch PSD -> dominant frequency in breaths/min."""
        sig = signal - np.mean(signal)
        if np.std(sig) < 1e-10:
            return None
        nperseg = min(len(sig), 256)
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        low = self.min_rr_brpm / 60.0
        high = self.max_rr_brpm / 60.0
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return None
        peak_idx = np.argmax(psd[mask])
        return float(freqs[mask][peak_idx] * 60.0)
