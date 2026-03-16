"""Signal filtering utilities for rPPG waveforms.

Provides bandpass filtering, ICA-based denoising, and motion-artifact
removal so that downstream vital-sign estimators receive a clean pulse.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Default pass-band 0.7 -- 3.5 Hz covers heart rates of 42 -- 210 bpm.

    Parameters:
        signal: 1-D rPPG waveform.
        fps: Sampling rate in Hz.
        low_hz: Lower cut-off frequency.
        high_hz: Upper cut-off frequency.
        order: Filter order (applied twice via ``filtfilt``).

    Returns:
        Filtered signal of the same length.
    """
    nyq = fps / 2.0
    if high_hz >= nyq:
        high_hz = nyq - 0.01
    if low_hz <= 0:
        low_hz = 0.01
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    # Pad to avoid edge artefacts when the signal is short.
    pad_len = min(3 * max(len(b), len(a)), len(signal) - 1)
    if pad_len < 1:
        return signal
    return filtfilt(b, a, signal, padlen=pad_len).astype(signal.dtype)


def ica_denoise(
    signals: np.ndarray,
    n_components: int | None = None,
) -> np.ndarray:
    """Blind-source separation via FastICA.

    Expects a 2-D array where each *row* is a signal channel (e.g. the three
    colour-channel rPPG traces).  Returns the independent components; the
    component with the strongest spectral peak in the cardiac band is
    conventionally selected by the caller.

    Parameters:
        signals: Shape ``(n_channels, n_samples)``.
        n_components: Number of ICA components (default: same as channels).

    Returns:
        Separated sources, shape ``(n_components, n_samples)``.
    """
    from sklearn.decomposition import FastICA

    if n_components is None:
        n_components = signals.shape[0]
    ica = FastICA(n_components=n_components, max_iter=500, random_state=42)
    sources = ica.fit_transform(signals.T).T  # (n_components, n_samples)
    return sources


def remove_motion_artifacts(
    signal: np.ndarray,
    fps: float,
    threshold_factor: float = 3.0,
    kernel_size: int = 5,
) -> np.ndarray:
    """Suppress motion-artifact spikes using median filtering and clipping.

    1. Compute the first derivative of the signal.
    2. Identify samples whose derivative exceeds ``threshold_factor`` times
       the median absolute deviation (MAD).
    3. Replace flagged samples via median interpolation.

    Parameters:
        signal: 1-D rPPG waveform.
        fps: Sampling rate (used only for future extensions).
        threshold_factor: MAD multiplier for spike detection.
        kernel_size: Median filter kernel (must be odd).

    Returns:
        Cleaned signal.
    """
    sig = signal.copy()
    diff = np.diff(sig, prepend=sig[0])
    mad = np.median(np.abs(diff - np.median(diff)))
    if mad == 0:
        mad = 1e-8
    outliers = np.abs(diff - np.median(diff)) > threshold_factor * mad

    if np.any(outliers):
        smoothed = medfilt(sig, kernel_size=kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        sig[outliers] = smoothed[outliers]

    return sig
