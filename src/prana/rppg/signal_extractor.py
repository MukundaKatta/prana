"""rPPG signal extraction from skin-color changes.

Implements the POS (Plane-Orthogonal-to-Skin) algorithm and the CHROM
(Chrominance-based) algorithm for recovering the blood-volume pulse from
temporal variations in skin pixel color.

Reference (POS):
    Wang, W. et al. "Algorithmic Principles of Remote PPG", IEEE TBME 2017.

Reference (CHROM):
    De Haan, G. & Jeanne, V. "Robust Pulse Rate from Chrominance-Based rPPG",
    IEEE TBME 2013.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np

from prana.models import PPGSignal, ROI


class Algorithm(str, Enum):
    POS = "POS"
    CHROM = "CHROM"
    GREEN = "GREEN"


class SignalExtractor:
    """Extract an rPPG signal from a time-series of skin ROIs.

    Parameters:
        algorithm: Which colour-subspace projection to use.
        window_size: Sliding-window length (in frames) for POS / CHROM.
    """

    def __init__(
        self,
        algorithm: Algorithm = Algorithm.POS,
        window_size: int = 32,
    ) -> None:
        self.algorithm = algorithm
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(
        self,
        roi_series: Sequence[list[ROI]],
        timestamps: np.ndarray,
        fps: float,
    ) -> PPGSignal:
        """Extract rPPG signal from a sequence of per-frame ROI lists.

        Parameters:
            roi_series: One list of ``ROI`` objects per frame.  Within each
                list the ROIs are averaged (spatial mean of all skin patches).
            timestamps: 1-D array of frame timestamps (seconds).
            fps: Effective video frame rate.

        Returns:
            A ``PPGSignal`` containing the cleaned pulse waveform.
        """
        # Step 1: compute spatial average RGB per frame.
        rgb_trace = self._spatial_average(roi_series)  # (N, 3)

        # Step 2: apply selected algorithm.
        if self.algorithm == Algorithm.GREEN:
            signal = self._green_channel(rgb_trace)
        elif self.algorithm == Algorithm.CHROM:
            signal = self._chrom(rgb_trace)
        else:
            signal = self._pos(rgb_trace)

        return PPGSignal(
            signal=signal,
            timestamps=timestamps[: len(signal)],
            fps=fps,
            channel=self.algorithm.value,
        )

    # ------------------------------------------------------------------
    # Spatial averaging
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_average(roi_series: Sequence[list[ROI]]) -> np.ndarray:
        """Compute mean BGR -> RGB per frame across all ROIs."""
        rgb_means: list[np.ndarray] = []
        for rois in roi_series:
            if not rois:
                # No face detected -- repeat last good value or zeros.
                rgb_means.append(rgb_means[-1] if rgb_means else np.zeros(3))
                continue
            frame_pixels: list[np.ndarray] = []
            for roi in rois:
                mean_bgr = roi.pixels.reshape(-1, 3).mean(axis=0)
                frame_pixels.append(mean_bgr)
            mean_bgr_all = np.mean(frame_pixels, axis=0)
            # Convert BGR -> RGB.
            rgb_means.append(mean_bgr_all[::-1].copy())
        return np.array(rgb_means, dtype=np.float64)

    # ------------------------------------------------------------------
    # Algorithm implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _green_channel(rgb: np.ndarray) -> np.ndarray:
        """Simplest rPPG: just the green channel, de-trended."""
        g = rgb[:, 1].copy()
        g -= np.mean(g)
        return g

    def _pos(self, rgb: np.ndarray) -> np.ndarray:
        """POS (Plane-Orthogonal-to-Skin) algorithm.

        Projects temporally normalised RGB onto a plane orthogonal to the
        skin-tone vector, combining two orthogonal chrominance signals with
        an adaptive weight based on their standard deviations.
        """
        n = len(rgb)
        ws = self.window_size
        pulse = np.zeros(n)

        for t in range(ws, n):
            window = rgb[t - ws : t, :]  # (ws, 3)
            # Temporal normalisation: divide each channel by its mean.
            means = window.mean(axis=0)
            means[means == 0] = 1.0
            normalised = window / means  # (ws, 3)

            # Projection onto orthogonal plane.
            s1 = normalised[:, 1] - normalised[:, 2]  # G - B
            s2 = (
                normalised[:, 1]
                + normalised[:, 2]
                - 2.0 * normalised[:, 0]
            )  # G + B - 2R

            # Adaptive combination.
            std1 = np.std(s1)
            std2 = np.std(s2) if np.std(s2) != 0 else 1e-8
            alpha = std1 / std2
            h = s1 + alpha * s2

            # Overlap-add.
            pulse[t - ws : t] += h - np.mean(h)

        return pulse

    def _chrom(self, rgb: np.ndarray) -> np.ndarray:
        """CHROM (Chrominance-based) algorithm.

        Uses fixed linear combinations of normalised colour channels and
        adaptively weights them within a sliding window.
        """
        n = len(rgb)
        ws = self.window_size
        pulse = np.zeros(n)

        for t in range(ws, n):
            window = rgb[t - ws : t, :]
            means = window.mean(axis=0)
            means[means == 0] = 1.0
            c = window / means

            xs = 3.0 * c[:, 0] - 2.0 * c[:, 1]
            ys = 1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2]

            std_x = np.std(xs)
            std_y = np.std(ys) if np.std(ys) != 0 else 1e-8
            alpha = std_x / std_y
            h = xs - alpha * ys

            pulse[t - ws : t] += h - np.mean(h)

        return pulse
