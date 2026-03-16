# Prana -- Wearable-Free Vital Sign Estimation from Smartphone Camera

Prana is a Python library and CLI tool for estimating vital signs from video
captured by an ordinary smartphone camera. It uses remote photoplethysmography
(rPPG) techniques to extract pulse signals from subtle skin-color changes
visible in the face, then derives five key vital signs:

| Vital | Method |
|---|---|
| Heart Rate (HR) | FFT-based spectral analysis of rPPG signal |
| Heart Rate Variability (HRV) | Inter-beat interval statistics and stress estimation |
| Respiratory Rate (RR) | Pulse amplitude / baseline modulation + chest motion |
| SpO2 | Multi-wavelength (R/G/B) ratio-of-ratios analysis |
| Blood Pressure (proxy) | Pulse Transit Time and waveform feature regression |

## Quick start

```bash
pip install -e .
prana measure --source webcam --duration 30
prana calibrate --reference-hr 72
prana report --session latest
```

## Architecture

```
VideoFrames -> FaceDetector -> ROI pixels
    -> SignalExtractor (POS / CHROM) -> raw rPPG
    -> Filters (bandpass + ICA) -> clean rPPG
    -> PeakDetector -> IBI series
    -> VitalEstimators -> VitalSigns
```

## Testing

```bash
pytest tests/
```

Tests use synthetically generated PPG waveforms so no camera or video files
are required.

## Author

Mukunda Katta

## License

MIT
