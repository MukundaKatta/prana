"""Tests for the rPPG sub-package: filters, peak detection, signal extraction."""

from __future__ import annotations

import numpy as np
import pytest

from prana.models import PPGSignal, ROI
from prana.rppg.filters import bandpass_filter, remove_motion_artifacts
from prana.rppg.peak_detector import PeakDetector
from prana.rppg.signal_extractor import Algorithm, SignalExtractor
from tests.conftest import generate_synthetic_ppg


# -----------------------------------------------------------------------
# Bandpass filter
# -----------------------------------------------------------------------


class TestBandpassFilter:
    def test_preserves_cardiac_frequency(self, synthetic_ppg):
        signal, _, fps, true_hr, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps, low_hz=0.7, high_hz=3.5)

        # The dominant frequency should still match the true HR.
        spectrum = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
        cardiac_mask = (freqs >= 0.7) & (freqs <= 3.5)
        dominant_freq = freqs[cardiac_mask][np.argmax(spectrum[cardiac_mask])]
        estimated_hr = dominant_freq * 60
        assert abs(estimated_hr - true_hr) < 5, f"Expected ~{true_hr}, got {estimated_hr}"

    def test_attenuates_out_of_band(self, synthetic_ppg):
        signal, _, fps, _, _ = synthetic_ppg
        # Inject a 10 Hz component that should be removed.
        t = np.arange(len(signal)) / fps
        contaminated = signal + 0.5 * np.sin(2 * np.pi * 10 * t)
        filtered = bandpass_filter(contaminated, fps)

        spectrum = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
        idx_10hz = np.argmin(np.abs(freqs - 10))
        # Power at 10 Hz should be heavily attenuated.
        assert spectrum[idx_10hz] < 0.1 * np.max(spectrum)

    def test_short_signal(self):
        """Filter should not crash on very short inputs."""
        sig = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = bandpass_filter(sig, fps=30.0)
        assert len(result) == len(sig)


# -----------------------------------------------------------------------
# Motion artifact removal
# -----------------------------------------------------------------------


class TestMotionArtifactRemoval:
    def test_removes_spikes(self, synthetic_ppg):
        signal, _, fps, _, _ = synthetic_ppg
        spiked = signal.copy()
        spiked[100] += 10.0  # huge spike
        spiked[200] -= 8.0
        cleaned = remove_motion_artifacts(spiked, fps)
        # Spike should be substantially reduced.
        assert abs(cleaned[100] - signal[100]) < abs(spiked[100] - signal[100])
        assert abs(cleaned[200] - signal[200]) < abs(spiked[200] - signal[200])

    def test_clean_signal_unchanged(self, synthetic_ppg):
        signal, _, fps, _, _ = synthetic_ppg
        cleaned = remove_motion_artifacts(signal, fps)
        # Should be very close to the original when no artifacts exist.
        correlation = np.corrcoef(signal, cleaned)[0, 1]
        assert correlation > 0.95


# -----------------------------------------------------------------------
# Peak detector
# -----------------------------------------------------------------------


class TestPeakDetector:
    def test_detects_correct_number_of_peaks(self, synthetic_ppg):
        signal, timestamps, fps, true_hr, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        detector = PeakDetector()
        peaks, ibi = detector.detect(filtered, fps)

        duration = timestamps[-1]
        expected_beats = int(true_hr / 60 * duration)
        # Allow +/- 15% tolerance.
        assert abs(len(peaks) - expected_beats) < expected_beats * 0.15

    def test_ibi_matches_hr(self, synthetic_ppg):
        signal, _, fps, true_hr, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        detector = PeakDetector()
        _, ibi = detector.detect(filtered, fps)

        mean_hr = 60.0 / np.mean(ibi) if len(ibi) > 0 else 0
        assert abs(mean_hr - true_hr) < 5, f"Expected ~{true_hr}, got {mean_hr}"

    def test_signal_quality_good_for_clean_signal(self, synthetic_ppg):
        signal, _, fps, _, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        detector = PeakDetector()
        _, ibi = detector.detect(filtered, fps)
        quality = detector.signal_quality(ibi)
        assert quality > 0.5


# -----------------------------------------------------------------------
# Signal extractor (POS algorithm on synthetic ROIs)
# -----------------------------------------------------------------------


class TestSignalExtractor:
    @staticmethod
    def _make_roi_series(n_frames: int, hr_bpm: float, fps: float) -> list[list[ROI]]:
        """Build a series of synthetic ROIs whose green channel encodes a PPG."""
        t = np.arange(n_frames) / fps
        f = hr_bpm / 60.0
        green_mod = 0.02 * np.sin(2 * np.pi * f * t)

        roi_series: list[list[ROI]] = []
        for i in range(n_frames):
            # Skin-like base colour + tiny green modulation.
            base_bgr = np.array([120, 160 + green_mod[i] * 255, 180], dtype=np.float32)
            pixels = np.tile(base_bgr, (10, 10, 1)).astype(np.uint8)
            roi = ROI(label="forehead", x=0, y=0, w=10, h=10, pixels=pixels)
            roi_series.append([roi])
        return roi_series

    def test_pos_extracts_signal(self):
        hr = 75.0
        fps = 30.0
        n = int(30 * fps)
        roi_series = self._make_roi_series(n, hr, fps)
        timestamps = np.arange(n) / fps

        extractor = SignalExtractor(algorithm=Algorithm.POS, window_size=32)
        ppg = extractor.extract(roi_series, timestamps, fps)

        assert len(ppg.signal) == n
        assert ppg.channel == "POS"
        # The signal should have content (not all zeros).
        assert np.std(ppg.signal) > 0

    def test_green_channel(self):
        hr = 70.0
        fps = 30.0
        n = int(10 * fps)
        roi_series = self._make_roi_series(n, hr, fps)
        timestamps = np.arange(n) / fps

        extractor = SignalExtractor(algorithm=Algorithm.GREEN)
        ppg = extractor.extract(roi_series, timestamps, fps)
        assert ppg.channel == "GREEN"
        assert np.std(ppg.signal) > 0
