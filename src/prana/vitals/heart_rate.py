"""FFT-based heart-rate estimation from an rPPG signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prana.models import ConfidenceInterval, PPGSignal


@dataclass
class HeartRateEstimator:
    """Estimate heart rate using spectral (FFT) analysis.

    Parameters:
        min_hr_bpm: Lower bound of the valid HR range.
        max_hr_bpm: Upper bound of the valid HR range.
        confidence_width_bpm: Half-width of the reported confidence interval
            (adaptive, but floored at this value).
    """

    min_hr_bpm: float = 40.0
    max_hr_bpm: float = 200.0
    confidence_width_bpm: float = 3.0

    def estimate(self, ppg: PPGSignal) -> ConfidenceInterval:
        """Return the dominant heart rate in BPM with a confidence band.

        The dominant frequency is found via zero-padded FFT of the rPPG
        signal.  The confidence interval width is scaled by the spectral
        sharpness (ratio of peak power to total power in the cardiac band).
        """
        signal = ppg.signal
        fps = ppg.fps
        n = len(signal)
        if n < 4:
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="bpm")

        # Window and zero-pad for resolution.
        windowed = signal * np.hanning(n)
        nfft = max(2048, int(2 ** np.ceil(np.log2(n * 4))))
        spectrum = np.abs(np.fft.rfft(windowed, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)

        # Restrict to cardiac band.
        low = self.min_hr_bpm / 60.0
        high = self.max_hr_bpm / 60.0
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="bpm")

        cardiac_spectrum = spectrum[mask]
        cardiac_freqs = freqs[mask]

        peak_idx = np.argmax(cardiac_spectrum)
        dominant_freq = cardiac_freqs[peak_idx]
        hr_bpm = dominant_freq * 60.0

        # Spectral concentration ratio -> confidence width.
        peak_power = cardiac_spectrum[peak_idx]
        total_power = cardiac_spectrum.sum()
        concentration = peak_power / total_power if total_power > 0 else 0
        # Higher concentration -> narrower CI.
        width = self.confidence_width_bpm / max(concentration * 5, 0.5)

        return ConfidenceInterval(
            value=round(hr_bpm, 1),
            lower=round(max(hr_bpm - width, self.min_hr_bpm), 1),
            upper=round(min(hr_bpm + width, self.max_hr_bpm), 1),
            unit="bpm",
        )

    def estimate_from_ibi(self, ibi_seconds: np.ndarray) -> ConfidenceInterval:
        """Alternative: estimate HR directly from inter-beat intervals."""
        if len(ibi_seconds) < 1:
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="bpm")
        hr_values = 60.0 / ibi_seconds
        hr_values = hr_values[
            (hr_values >= self.min_hr_bpm) & (hr_values <= self.max_hr_bpm)
        ]
        if len(hr_values) == 0:
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="bpm")
        mean_hr = float(np.mean(hr_values))
        std_hr = float(np.std(hr_values))
        return ConfidenceInterval(
            value=round(mean_hr, 1),
            lower=round(max(mean_hr - 1.96 * std_hr, self.min_hr_bpm), 1),
            upper=round(min(mean_hr + 1.96 * std_hr, self.max_hr_bpm), 1),
            unit="bpm",
        )
