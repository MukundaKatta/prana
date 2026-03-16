"""Face detection and skin-ROI extraction.

Uses MediaPipe Face Mesh to locate the face, then extracts forehead and
cheek regions suitable for rPPG signal recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from prana.models import ROI, VideoFrame

# MediaPipe Face Mesh landmark indices for skin ROIs.
# These are approximate convex-hull groups on the canonical 468-point mesh.
_FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                     67, 109]
_LEFT_CHEEK_INDICES = [36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 100,
                       142, 203, 206]
_RIGHT_CHEEK_INDICES = [266, 425, 426, 427, 411, 352, 345, 346, 347, 348,
                        329, 371, 423, 426]


@dataclass
class FaceDetector:
    """Detect face landmarks and extract skin ROIs from video frames.

    Parameters:
        min_detection_confidence: MediaPipe detection confidence threshold.
        min_tracking_confidence: MediaPipe tracking confidence threshold.
        roi_padding: Fractional padding applied around each ROI bounding box.
    """

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    roi_padding: float = 0.05
    _face_mesh: object = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_mesh(self) -> None:
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except ImportError:
            # Fallback: use OpenCV Haar cascade for basic face detection.
            self._face_mesh = None

    def close(self) -> None:
        if self._face_mesh is not None and hasattr(self._face_mesh, "close"):
            self._face_mesh.close()
            self._face_mesh = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: VideoFrame) -> VideoFrame:
        """Detect face and populate ``frame.rois`` with skin regions.

        Returns the *same* ``VideoFrame`` instance with its ``rois`` list
        populated.
        """
        self._ensure_mesh()
        image = frame.image
        h, w = image.shape[:2]

        rois: list[ROI] = []

        if self._face_mesh is not None:
            rois = self._detect_mediapipe(image, h, w)
        else:
            rois = self._detect_haar(image, h, w)

        frame.rois = rois
        return frame

    # ------------------------------------------------------------------
    # MediaPipe path
    # ------------------------------------------------------------------

    def _landmarks_to_roi(
        self,
        landmarks: list,
        indices: list[int],
        h: int,
        w: int,
        image: np.ndarray,
        label: str,
    ) -> Optional[ROI]:
        pts = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices
             if i < len(landmarks)]
        )
        if len(pts) < 3:
            return None
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        pad_x = int((x_max - x_min) * self.roi_padding)
        pad_y = int((y_max - y_min) * self.roi_padding)
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        roi_w = x_max - x_min
        roi_h = y_max - y_min
        if roi_w < 4 or roi_h < 4:
            return None

        pixels = image[y_min:y_max, x_min:x_max].copy()
        return ROI(label=label, x=x_min, y=y_min, w=roi_w, h=roi_h, pixels=pixels)

    def _detect_mediapipe(self, image: np.ndarray, h: int, w: int) -> list[ROI]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []

        landmarks = results.multi_face_landmarks[0].landmark
        rois: list[ROI] = []
        for indices, label in [
            (_FOREHEAD_INDICES, "forehead"),
            (_LEFT_CHEEK_INDICES, "left_cheek"),
            (_RIGHT_CHEEK_INDICES, "right_cheek"),
        ]:
            roi = self._landmarks_to_roi(landmarks, indices, h, w, image, label)
            if roi is not None:
                rois.append(roi)
        return rois

    # ------------------------------------------------------------------
    # Haar-cascade fallback (no MediaPipe)
    # ------------------------------------------------------------------

    def _detect_haar(self, image: np.ndarray, h: int, w: int) -> list[ROI]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return []

        fx, fy, fw, fh = faces[0]
        # Approximate forehead: top 30 % of face box.
        fh_top = int(fh * 0.30)
        forehead_pixels = image[fy : fy + fh_top, fx : fx + fw].copy()
        rois = [
            ROI(label="forehead", x=int(fx), y=int(fy), w=int(fw), h=fh_top, pixels=forehead_pixels)
        ]
        # Approximate cheeks: middle horizontal band, left/right halves.
        cy = fy + int(fh * 0.45)
        ch = int(fh * 0.25)
        half_w = fw // 2
        for label, cx in [("left_cheek", fx), ("right_cheek", fx + half_w)]:
            pixels = image[cy : cy + ch, cx : cx + half_w].copy()
            rois.append(ROI(label=label, x=int(cx), y=int(cy), w=int(half_w), h=int(ch), pixels=pixels))
        return rois
