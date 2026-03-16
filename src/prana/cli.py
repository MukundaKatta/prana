"""Click CLI for Prana.

Commands:
    prana measure   -- run a vital-sign measurement session
    prana calibrate -- calibrate SpO2/BP against reference values
    prana report    -- display results of a previous session
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console

from prana import __version__

SESSIONS_DIR = Path.home() / ".prana" / "sessions"

console = Console()


@click.group()
@click.version_option(__version__, prog_name="prana")
def cli() -> None:
    """Prana -- Wearable-Free Vital Sign Estimation from Smartphone Camera."""


# -----------------------------------------------------------------------
# measure
# -----------------------------------------------------------------------


@cli.command()
@click.option(
    "--source",
    default="webcam",
    help="Video source: 'webcam', camera index (0,1,...), or path to video file.",
)
@click.option("--duration", default=30.0, type=float, help="Recording duration in seconds.")
@click.option(
    "--algorithm",
    type=click.Choice(["POS", "CHROM", "GREEN"], case_sensitive=False),
    default="POS",
    help="rPPG extraction algorithm.",
)
@click.option("--preview/--no-preview", default=False, help="Show live camera preview.")
@click.option("--save/--no-save", default=True, help="Persist session to ~/.prana/sessions/.")
def measure(source: str, duration: float, algorithm: str, preview: bool, save: bool) -> None:
    """Run a vital-sign measurement session."""
    from prana.pipeline import VitalsPipeline
    from prana.report import print_vitals
    from prana.rppg.signal_extractor import Algorithm

    src: str | int = source
    if source.isdigit():
        src = int(source)

    algo = Algorithm[algorithm.upper()]
    pipeline = VitalsPipeline(source=src, duration_s=duration, algorithm=algo)

    console.print(f"[bold]Starting measurement[/bold] (source={source}, {duration}s, {algorithm})")
    session = pipeline.run(show_preview=preview)

    print_vitals(session, console=console)

    if save and session.vitals is not None:
        _save_session(session)
        console.print(f"[dim]Session saved: {session.session_id}[/dim]")


# -----------------------------------------------------------------------
# calibrate
# -----------------------------------------------------------------------


@cli.command()
@click.option("--reference-hr", type=float, default=None, help="Reference heart rate (bpm).")
@click.option("--reference-spo2", type=float, default=None, help="Reference SpO2 (%).")
@click.option("--reference-sbp", type=float, default=None, help="Reference systolic BP (mmHg).")
@click.option("--reference-dbp", type=float, default=None, help="Reference diastolic BP (mmHg).")
def calibrate(
    reference_hr: float | None,
    reference_spo2: float | None,
    reference_sbp: float | None,
    reference_dbp: float | None,
) -> None:
    """Calibrate estimation parameters against reference measurements.

    Provide known reference values so Prana can adjust its internal
    coefficients for better accuracy on your device / skin tone.
    """
    cal_path = Path.home() / ".prana" / "calibration.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)

    cal: dict = {}
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())

    if reference_hr is not None:
        cal["reference_hr_bpm"] = reference_hr
    if reference_spo2 is not None:
        cal["reference_spo2_pct"] = reference_spo2
    if reference_sbp is not None:
        cal["reference_sbp_mmhg"] = reference_sbp
    if reference_dbp is not None:
        cal["reference_dbp_mmhg"] = reference_dbp

    cal_path.write_text(json.dumps(cal, indent=2))
    console.print(f"[green]Calibration saved to {cal_path}[/green]")
    for k, v in cal.items():
        console.print(f"  {k}: {v}")


# -----------------------------------------------------------------------
# report
# -----------------------------------------------------------------------


@cli.command()
@click.option(
    "--session",
    "session_id",
    default="latest",
    help="Session ID or 'latest'.",
)
def report(session_id: str) -> None:
    """Display results from a saved session."""
    from prana.models import MeasurementSession
    from prana.report import print_vitals

    sessions_dir = SESSIONS_DIR
    if not sessions_dir.exists():
        console.print("[red]No saved sessions found.[/red]")
        return

    if session_id == "latest":
        files = sorted(sessions_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not files:
            console.print("[red]No saved sessions found.[/red]")
            return
        path = files[0]
    else:
        path = sessions_dir / f"{session_id}.json"

    if not path.exists():
        console.print(f"[red]Session not found: {session_id}[/red]")
        return

    data = json.loads(path.read_text())
    session = MeasurementSession(**data)
    print_vitals(session, console=console)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _save_session(session) -> None:  # noqa: ANN001
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{session.session_id}.json"
    path.write_text(session.model_dump_json(indent=2))
