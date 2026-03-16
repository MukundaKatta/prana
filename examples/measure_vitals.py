#!/usr/bin/env python3
"""Example: run a full vital-sign measurement from the webcam or a video file.

Usage:
    python examples/measure_vitals.py                   # webcam, 30 s
    python examples/measure_vitals.py --file video.mp4  # from file
    python examples/measure_vitals.py --synthetic       # synthetic demo
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def run_synthetic_demo() -> None:
    """Run the pipeline on a synthetically generated PPG signal.

    This path requires NO camera and NO video file -- it constructs fake
    video frames whose pixel values embed a known PPG waveform, then runs
    the full pipeline to demonstrate the estimation outputs.
    """
    from prana.models import ROI, VideoFrame
    from prana.pipeline import VitalsPipeline
    from prana.report import print_vitals

    fps = 30.0
    duration_s = 30.0
    n_frames = int(fps * duration_s)
    hr_bpm = 72.0
    rr_brpm = 15.0

    t = np.arange(n_frames) / fps
    f_hr = hr_bpm / 60.0
    f_rr = rr_brpm / 60.0

    # Simulate skin-colour changes encoding a PPG.
    images: list[np.ndarray] = []
    for i in range(n_frames):
        # Subtle green-channel modulation around a skin-like base.
        g_mod = 2 * np.sin(2 * np.pi * f_hr * t[i])
        r_mod = 1 * np.sin(2 * np.pi * f_hr * t[i] + 0.3)
        b_mod = 0.5 * np.sin(2 * np.pi * f_hr * t[i] + 0.6)
        # Respiratory baseline wander.
        resp = 0.8 * np.sin(2 * np.pi * f_rr * t[i])

        r = int(np.clip(180 + r_mod + resp, 0, 255))
        g = int(np.clip(160 + g_mod + resp, 0, 255))
        b = int(np.clip(130 + b_mod + resp, 0, 255))

        # 64x64 "face" image.
        img = np.full((64, 64, 3), [b, g, r], dtype=np.uint8)
        images.append(img)

    print(f"Running synthetic demo: {n_frames} frames, {fps} fps, "
          f"true HR={hr_bpm} bpm, true RR={rr_brpm} brpm")

    pipeline = VitalsPipeline(algorithm="POS")  # type: ignore[arg-type]
    session = pipeline.run_from_frames(images, fps=fps)
    print_vitals(session)


def run_live(source, duration: float) -> None:
    """Run from webcam or video file."""
    from prana.pipeline import VitalsPipeline
    from prana.report import print_vitals

    pipeline = VitalsPipeline(source=source, duration_s=duration)
    session = pipeline.run(show_preview=True)
    print_vitals(session)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prana vital-sign measurement")
    parser.add_argument("--file", type=str, default=None, help="Path to video file")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic demo (no camera)")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_demo()
    elif args.file:
        run_live(args.file, args.duration)
    else:
        run_live("webcam", args.duration)


if __name__ == "__main__":
    main()
