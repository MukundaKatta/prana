"""Validation utilities: compare estimated vitals against ground truth.

Provides Bland-Altman analysis for assessing measurement agreement between
Prana estimates and reference devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlandAltmanResult:
    """Result of a Bland-Altman agreement analysis."""

    bias: float
    lower_loa: float  # lower limit of agreement (bias - 1.96*SD)
    upper_loa: float  # upper limit of agreement (bias + 1.96*SD)
    sd: float
    n: int

    @property
    def loa_range(self) -> float:
        return self.upper_loa - self.lower_loa

    def __repr__(self) -> str:
        return (
            f"BlandAltman(bias={self.bias:+.2f}, "
            f"LoA=[{self.lower_loa:.2f}, {self.upper_loa:.2f}], n={self.n})"
        )


def bland_altman(
    estimates: np.ndarray,
    references: np.ndarray,
) -> BlandAltmanResult:
    """Compute Bland-Altman statistics.

    Parameters:
        estimates: 1-D array of values from the method under test (Prana).
        references: 1-D array of ground-truth / reference values.

    Returns:
        A ``BlandAltmanResult`` with bias, limits of agreement, and SD.
    """
    if len(estimates) != len(references):
        raise ValueError("Arrays must have the same length.")
    if len(estimates) < 2:
        raise ValueError("At least 2 paired observations required.")

    diff = estimates - references
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return BlandAltmanResult(
        bias=bias,
        lower_loa=bias - 1.96 * sd,
        upper_loa=bias + 1.96 * sd,
        sd=sd,
        n=len(estimates),
    )


@dataclass
class ValidationReport:
    """Aggregate validation metrics for one vital sign."""

    vital_name: str
    unit: str
    mae: float
    rmse: float
    correlation: float
    bland_altman: BlandAltmanResult


def validate_vital(
    vital_name: str,
    unit: str,
    estimates: np.ndarray,
    references: np.ndarray,
) -> ValidationReport:
    """Full validation report for a single vital sign.

    Parameters:
        vital_name: Human-readable name (e.g. ``"Heart Rate"``).
        unit: Unit string (e.g. ``"bpm"``).
        estimates: Prana values.
        references: Ground-truth values.

    Returns:
        ``ValidationReport`` with MAE, RMSE, Pearson r, and Bland-Altman.
    """
    est = np.asarray(estimates, dtype=np.float64)
    ref = np.asarray(references, dtype=np.float64)

    mae = float(np.mean(np.abs(est - ref)))
    rmse = float(np.sqrt(np.mean((est - ref) ** 2)))
    if np.std(est) > 0 and np.std(ref) > 0:
        correlation = float(np.corrcoef(est, ref)[0, 1])
    else:
        correlation = 0.0

    ba = bland_altman(est, ref)

    return ValidationReport(
        vital_name=vital_name,
        unit=unit,
        mae=mae,
        rmse=rmse,
        correlation=correlation,
        bland_altman=ba,
    )
