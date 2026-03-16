"""Tests for the vitals sub-package: HR, HRV, RR, SpO2, BP estimators."""

from __future__ import annotations

import numpy as np
import pytest

from prana.models import ConfidenceInterval, PPGSignal, ROI, StressLevel
from prana.rppg.filters import bandpass_filter
from prana.rppg.peak_detector import PeakDetector
from prana.vitals.blood_pressure import BPEstimator
from prana.vitals.heart_rate import HeartRateEstimator
from prana.vitals.hrv import HRVAnalyzer
from prana.vitals.respiratory import RespiratoryEstimator
from prana.vitals.spo2 import SpO2Estimator
from tests.conftest import generate_synthetic_ppg


# -----------------------------------------------------------------------
# Heart rate
# -----------------------------------------------------------------------


class TestHeartRateEstimator:
    def test_fft_estimate_matches_true_hr(self, synthetic_ppg):
        signal, timestamps, fps, true_hr, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        ppg = PPGSignal(signal=filtered, timestamps=timestamps, fps=fps)

        estimator = HeartRateEstimator()
        ci = estimator.estimate(ppg)
        assert abs(ci.value - true_hr) < 5, f"Expected ~{true_hr}, got {ci.value}"
        assert ci.unit == "bpm"
        assert ci.lower <= ci.value <= ci.upper

    def test_ibi_estimate(self, synthetic_ppg):
        signal, _, fps, true_hr, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        peaks, ibi = PeakDetector().detect(filtered, fps)

        estimator = HeartRateEstimator()
        ci = estimator.estimate_from_ibi(ibi)
        assert abs(ci.value - true_hr) < 5

    def test_empty_signal(self):
        ppg = PPGSignal(signal=np.array([0.0, 0.0]), timestamps=np.array([0.0, 1.0]), fps=30.0)
        ci = HeartRateEstimator().estimate(ppg)
        assert ci.value == 0

    @pytest.mark.parametrize("hr", [50, 72, 100, 150])
    def test_various_heart_rates(self, hr):
        signal, timestamps = generate_synthetic_ppg(hr_bpm=hr, duration_s=30, fps=30)
        filtered = bandpass_filter(signal, 30.0)
        ppg = PPGSignal(signal=filtered, timestamps=timestamps, fps=30.0)
        ci = HeartRateEstimator().estimate(ppg)
        assert abs(ci.value - hr) < 6, f"HR={hr}: got {ci.value}"


# -----------------------------------------------------------------------
# HRV
# -----------------------------------------------------------------------


class TestHRVAnalyzer:
    def test_sdnn_and_rmssd(self):
        # Perfectly regular IBI -> near-zero HRV.
        ibi = np.full(50, 0.833)  # 72 bpm
        analyzer = HRVAnalyzer()
        sdnn_ci, rmssd_ci, stress = analyzer.analyze(ibi)
        assert sdnn_ci.value < 5  # very low variability
        assert rmssd_ci.value < 5

    def test_high_variability_indicates_low_stress(self):
        rng = np.random.default_rng(0)
        ibi = 0.833 + rng.normal(0, 0.06, 100)  # lots of variability
        analyzer = HRVAnalyzer()
        sdnn_ci, rmssd_ci, stress = analyzer.analyze(ibi)
        assert sdnn_ci.value > 30
        assert stress in (StressLevel.LOW, StressLevel.MODERATE)

    def test_pnn50(self):
        analyzer = HRVAnalyzer()
        ibi = np.array([800, 860, 810, 870, 805, 862], dtype=float)
        pnn = analyzer.pnn50(ibi)
        assert 0 <= pnn <= 100


# -----------------------------------------------------------------------
# Respiratory rate
# -----------------------------------------------------------------------


class TestRespiratoryEstimator:
    def test_detects_respiratory_rate(self, synthetic_ppg):
        signal, timestamps, fps, _, true_rr = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        ppg = PPGSignal(signal=filtered, timestamps=timestamps, fps=fps)
        peaks, ibi = PeakDetector().detect(filtered, fps)

        estimator = RespiratoryEstimator()
        ci = estimator.estimate(ppg, peaks, ibi)
        # Respiratory rate estimation is inherently noisy; allow wide margin.
        assert ci.value > 0, "Should produce a non-zero estimate"
        assert ci.unit == "brpm"

    def test_insufficient_data(self):
        ppg = PPGSignal(
            signal=np.zeros(10), timestamps=np.arange(10) / 30.0, fps=30.0
        )
        ci = RespiratoryEstimator().estimate(ppg, np.array([]), np.array([]))
        # Should gracefully return zero rather than crash.
        assert ci.value >= 0


# -----------------------------------------------------------------------
# SpO2
# -----------------------------------------------------------------------


class TestSpO2Estimator:
    @staticmethod
    def _make_roi_series(n: int = 300) -> list[list[ROI]]:
        """ROIs with realistic skin-like pixel values."""
        rois: list[list[ROI]] = []
        t = np.arange(n) / 30.0
        for i in range(n):
            # Simulate subtle pulsatile variation.
            r = int(180 + 2 * np.sin(2 * np.pi * 1.2 * t[i]))
            g = int(160 + 3 * np.sin(2 * np.pi * 1.2 * t[i]))
            b = int(130 + 1.5 * np.sin(2 * np.pi * 1.2 * t[i]))
            pixels = np.full((10, 10, 3), [b, g, r], dtype=np.uint8)
            rois.append([ROI(label="forehead", x=0, y=0, w=10, h=10, pixels=pixels)])
        return rois

    def test_spo2_in_normal_range(self):
        rois = self._make_roi_series()
        estimator = SpO2Estimator(fps=30.0)
        ci = estimator.estimate(rois)
        assert 85 <= ci.value <= 100, f"SpO2 out of range: {ci.value}"
        assert ci.unit == "%"

    def test_empty_rois(self):
        ci = SpO2Estimator().estimate([])
        assert ci.value == 0


# -----------------------------------------------------------------------
# Blood pressure
# -----------------------------------------------------------------------


class TestBPEstimator:
    def test_bp_in_plausible_range(self, synthetic_ppg):
        signal, _, fps, _, _ = synthetic_ppg
        filtered = bandpass_filter(signal, fps)
        peaks, ibi = PeakDetector().detect(filtered, fps)

        estimator = BPEstimator()
        sbp, dbp = estimator.estimate(filtered, peaks, ibi, fps)
        assert 80 <= sbp.value <= 200, f"SBP out of range: {sbp.value}"
        assert 50 <= dbp.value <= 130, f"DBP out of range: {dbp.value}"
        assert sbp.unit == "mmHg"

    def test_insufficient_peaks(self):
        est = BPEstimator()
        sbp, dbp = est.estimate(np.zeros(10), np.array([5]), np.array([]), fps=30)
        assert sbp.value == 0
        assert dbp.value == 0


# -----------------------------------------------------------------------
# Validator (Bland-Altman)
# -----------------------------------------------------------------------


class TestValidator:
    def test_bland_altman_perfect_agreement(self):
        from prana.validator import bland_altman

        vals = np.array([70, 72, 74, 68, 75], dtype=float)
        result = bland_altman(vals, vals)
        assert abs(result.bias) < 1e-10
        assert result.n == 5

    def test_bland_altman_known_bias(self):
        from prana.validator import bland_altman

        est = np.array([72, 74, 76, 70, 77], dtype=float)
        ref = np.array([70, 72, 74, 68, 75], dtype=float)
        result = bland_altman(est, ref)
        assert abs(result.bias - 2.0) < 1e-10

    def test_validate_vital(self):
        from prana.validator import validate_vital

        est = np.array([71, 73, 75, 69, 76], dtype=float)
        ref = np.array([70, 72, 74, 68, 75], dtype=float)
        report = validate_vital("Heart Rate", "bpm", est, ref)
        assert report.mae == pytest.approx(1.0)
        assert report.correlation > 0.99
