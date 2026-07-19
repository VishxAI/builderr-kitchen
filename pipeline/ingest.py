"""Video metadata probing via OpenCV (no system ffmpeg dependency)."""

from pathlib import Path

import cv2


def probe(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": frame_count / fps if fps else 0.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()
