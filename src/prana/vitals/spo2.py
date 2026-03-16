"""Blood-oxygen saturation (SpO2) estimation from multi-wavelength rPPG.

Standard pulse oximetry uses the ratio-of-ratios (R) of pulsatile (AC) to
steady-state (DC) components in red and infrared channels.  With a
smartphone camera we approximate this using the red and blue channels of
the RGB Bayer sensor:

    R = (AC_red / DC_red) / (AC_blue / DC_blue)
    SpO2 = a - b * R

The empirical calibration constants ``a`` and ``b`` are device-dependent
and should ideally be determined against a reference pulse oximeter (see
``prana calibrate``).  The defaults here are rough starting points derived
from published smartphone rPPG studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from prana.models import ConfidenceInterval, ROI
from prana.rppg.filters import bandpass_filter


@dataclass
class SpO2Estimator:
    """Estimate SpO2 from per-channel rPPG pulsatile amplitude.

    Parameters:
        calibration_a: Intercept of the linear SpO2-vs-R model.
        calibration_b: Slope of the linear SpO2-vs-R model.
        fps: Sampling rate used for bandpass filtering the channel traces.
    """

    calibration_a: float = 110.0
    calibration_b: float = 25.0
    fps: float = 30.0

    def estimate(
        self,
        roi_series: Sequence[list[ROI]],
        fps: float | None = None,
    ) -> ConfidenceInterval:
        """Compute SpO2 from the ratio-of-ratios of RGB channels.

        Parameters:
            roi_series: Per-frame list of skin ROIs (same format as
                ``SignalExtractor.extract``).
            fps: Override sampling rate if different from construction-time.

        Returns:
            SpO2 percentage with confidence interval.
        """
        if fps is not None:
            self.fps = fps

        # Extract per-channel spatial mean traces.
        red_trace: list[float] = []
        green_trace: list[float] = []
        blue_trace: list[float] = []

        for rois in roi_series:
            if not rois:
                red_trace.append(red_trace[-1] if red_trace else 0.0)
                green_trace.append(green_trace[-1] if green_trace else 0.0)
                blue_trace.append(blue_trace[-1] if blue_trace else 0.0)
                continue
            pixels = np.vstack([roi.pixels.reshape(-1, 3) for roi in rois])
            # OpenCV BGR order.
            blue_trace.append(float(pixels[:, 0].mean()))
            green_trace.append(float(pixels[:, 1].mean()))
            red_trace.append(float(pixels[:, 2].mean()))

        red = np.array(red_trace, dtype=np.float64)
        blue = np.array(blue_trace, dtype=np.float64)

        if len(red) < 10:
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="%")

        # Bandpass to isolate pulsatile (AC) component.
        red_ac = bandpass_filter(red, self.fps)
        blue_ac = bandpass_filter(blue, self.fps)

        # DC component: temporal mean.
        red_dc = np.mean(red) if np.mean(red) != 0 else 1.0
        blue_dc = np.mean(blue) if np.mean(blue) != 0 else 1.0

        # RMS of AC as pulsatile amplitude proxy.
        red_ac_rms = np.sqrt(np.mean(red_ac ** 2))
        blue_ac_rms = np.sqrt(np.mean(blue_ac ** 2))

        if blue_ac_rms == 0:
            return ConfidenceInterval(value=0, lower=0, upper=0, unit="%")

        ratio_of_ratios = (red_ac_rms / red_dc) / (blue_ac_rms / blue_dc)

        spo2 = self.calibration_a - self.calibration_b * ratio_of_ratios
        spo2 = float(np.clip(spo2, 70, 100))

        # Confidence: narrower when pulsatile SNR is higher.
        snr = (red_ac_rms + blue_ac_rms) / 2.0
        width = max(2.0, 5.0 / (1.0 + 10.0 * snr))

        return ConfidenceInterval(
            value=round(spo2, 1),
            lower=round(max(spo2 - width, 70.0), 1),
            upper=round(min(spo2 + width, 100.0), 1),
            unit="%",
        )
