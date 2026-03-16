"""Shared fixtures: synthetic PPG signal generation."""

from __future__ import annotations

import numpy as np
import pytest


def generate_synthetic_ppg(
    hr_bpm: float = 72.0,
    respiratory_rate_brpm: float = 15.0,
    duration_s: float = 30.0,
    fps: float = 30.0,
    noise_level: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a realistic synthetic PPG waveform.

    The signal is composed of:
    * A fundamental cardiac frequency and its first two harmonics.
    * Respiratory modulation (amplitude + baseline).
    * Additive Gaussian noise.

    Returns:
        ``(signal, timestamps)``
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fps)
    t = np.arange(n_samples) / fps

    f_hr = hr_bpm / 60.0  # cardiac fundamental (Hz)
    f_rr = respiratory_rate_brpm / 60.0  # respiratory (Hz)

    # Cardiac harmonics (approximate PPG morphology).
    cardiac = (
        np.sin(2 * np.pi * f_hr * t)
        + 0.5 * np.sin(2 * np.pi * 2 * f_hr * t - np.pi / 4)
        + 0.25 * np.sin(2 * np.pi * 3 * f_hr * t - np.pi / 3)
    )

    # Respiratory modulation.
    resp_mod = 1.0 + 0.15 * np.sin(2 * np.pi * f_rr * t)
    baseline_wander = 0.3 * np.sin(2 * np.pi * f_rr * t + np.pi / 6)

    signal = cardiac * resp_mod + baseline_wander
    signal += rng.normal(0, noise_level, n_samples)
    signal = signal / np.max(np.abs(signal))  # normalise to [-1, 1]

    return signal, t


@pytest.fixture()
def synthetic_ppg():
    """Fixture returning ``(signal, timestamps, fps, true_hr, true_rr)``."""
    hr = 72.0
    rr = 15.0
    fps = 30.0
    signal, timestamps = generate_synthetic_ppg(hr_bpm=hr, respiratory_rate_brpm=rr, fps=fps)
    return signal, timestamps, fps, hr, rr
