"""Rich terminal display of vital signs with confidence intervals."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prana.models import MeasurementSession, VitalSigns


def _ci_str(ci) -> str:  # noqa: ANN001
    """Format a ConfidenceInterval as  ``value [lower -- upper] unit``."""
    return f"{ci.value} [{ci.lower} -- {ci.upper}] {ci.unit}"


def print_vitals(session: MeasurementSession, console: Console | None = None) -> None:
    """Print a rich table of vital signs to the terminal.

    Parameters:
        session: Completed ``MeasurementSession`` with vitals populated.
        console: Optional ``rich.Console`` instance (default: new console).
    """
    if console is None:
        console = Console()

    if session.vitals is None:
        console.print("[bold red]No vital signs recorded.[/bold red]")
        return

    v: VitalSigns = session.vitals

    table = Table(
        title="Prana Vital Signs",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
        title_style="bold white on blue",
    )
    table.add_column("Vital", style="bold")
    table.add_column("Estimate", justify="right")
    table.add_column("95% CI", justify="right", style="dim")

    table.add_row(
        "Heart Rate",
        f"{v.heart_rate_bpm.value} {v.heart_rate_bpm.unit}",
        f"[{v.heart_rate_bpm.lower} -- {v.heart_rate_bpm.upper}]",
    )
    table.add_row(
        "HRV (SDNN)",
        f"{v.hrv_sdnn_ms.value} {v.hrv_sdnn_ms.unit}",
        f"[{v.hrv_sdnn_ms.lower} -- {v.hrv_sdnn_ms.upper}]",
    )
    table.add_row(
        "HRV (RMSSD)",
        f"{v.hrv_rmssd_ms.value} {v.hrv_rmssd_ms.unit}",
        f"[{v.hrv_rmssd_ms.lower} -- {v.hrv_rmssd_ms.upper}]",
    )

    stress_color = {
        "low": "green",
        "moderate": "yellow",
        "high": "red",
    }.get(v.stress_level.value, "white")
    table.add_row(
        "Stress Level",
        Text(v.stress_level.value.upper(), style=f"bold {stress_color}"),
        "",
    )

    table.add_row(
        "Respiratory Rate",
        f"{v.respiratory_rate_brpm.value} {v.respiratory_rate_brpm.unit}",
        f"[{v.respiratory_rate_brpm.lower} -- {v.respiratory_rate_brpm.upper}]",
    )
    table.add_row(
        "SpO2",
        f"{v.spo2_percent.value} {v.spo2_percent.unit}",
        f"[{v.spo2_percent.lower} -- {v.spo2_percent.upper}]",
    )
    table.add_row(
        "Blood Pressure (sys)",
        f"{v.systolic_bp_mmhg.value} {v.systolic_bp_mmhg.unit}",
        f"[{v.systolic_bp_mmhg.lower} -- {v.systolic_bp_mmhg.upper}]",
    )
    table.add_row(
        "Blood Pressure (dia)",
        f"{v.diastolic_bp_mmhg.value} {v.diastolic_bp_mmhg.unit}",
        f"[{v.diastolic_bp_mmhg.lower} -- {v.diastolic_bp_mmhg.upper}]",
    )

    quality_color = "green" if v.quality_score > 0.7 else "yellow" if v.quality_score > 0.4 else "red"
    table.add_row(
        "Signal Quality",
        Text(f"{v.quality_score:.0%}", style=f"bold {quality_color}"),
        "",
    )

    panel = Panel(
        table,
        subtitle=(
            f"Session {session.session_id} | "
            f"{session.frame_count} frames @ {session.fps:.1f} fps | "
            f"{session.duration_s:.1f}s"
        ),
        subtitle_align="right",
    )
    console.print(panel)
