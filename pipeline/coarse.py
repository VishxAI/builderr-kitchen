"""Coarse activity index: the cheap full-video pass.

Samples one frame every `interval_s` seconds (default 4 s ≈ 60% of the
25-frames/min cap, leaving budget for fine passes), computes a motion score
against the previous sample, and runs YOLO person detection in batches.
Everything is local — zero API cost.

Output: a list of records
    {"t": seconds, "activity": float, "persons": int, "person_boxes": [[x1,y1,x2,y2], ...]}

CLI:  python -m pipeline.coarse videos/sample_19min.mp4 --out index.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .ingest import probe

MOTION_WIDTH = 320          # downscale width for motion differencing
YOLO_BATCH = 32
YOLO_MODEL = "yolo11n.pt"   # nano model, ~5 MB, auto-downloaded by ultralytics


def sample_frames(video_path: Path, interval_s: float):
    """Yield (t_seconds, frame_bgr) sequentially — no random seeking."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * interval_s))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx / fps, frame
            idx += 1
    finally:
        cap.release()


def motion_score(prev_gray, gray) -> float:
    """Mean absolute pixel difference between consecutive samples (0-255)."""
    if prev_gray is None:
        return 0.0
    return float(np.mean(cv2.absdiff(prev_gray, gray)))


def build_index(video_path: Path, interval_s: float = 4.0, device: str | None = None) -> dict:
    from ultralytics import YOLO

    meta = probe(video_path)
    model = YOLO(YOLO_MODEL)

    records = []
    frames_batch, times_batch = [], []
    prev_gray = None
    started = time.monotonic()

    def flush():
        if not frames_batch:
            return
        results = model(frames_batch, classes=[0], verbose=False, device=device)
        for t, res in zip(times_batch, results):
            boxes = res.boxes.xyxy.cpu().numpy().round(1).tolist() if res.boxes is not None else []
            rec = next(r for r in records if r["t"] == t)
            rec["persons"] = len(boxes)
            rec["person_boxes"] = boxes
        frames_batch.clear()
        times_batch.clear()

    for t, frame in sample_frames(video_path, interval_s):
        small = cv2.resize(frame, (MOTION_WIDTH, int(MOTION_WIDTH * frame.shape[0] / frame.shape[1])))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        records.append({"t": round(t, 2), "activity": round(motion_score(prev_gray, gray), 3)})
        prev_gray = gray

        frames_batch.append(frame)
        times_batch.append(round(t, 2))
        if len(frames_batch) >= YOLO_BATCH:
            flush()
    flush()

    return {
        "video": meta,
        "interval_s": interval_s,
        "frames_sampled": len(records),
        "build_seconds": round(time.monotonic() - started, 1),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build coarse activity index for a video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    index = build_index(args.video, args.interval)
    out = args.out or args.video.with_suffix(".index.json")
    out.write_text(json.dumps(index), encoding="utf-8")

    recs = index["records"]
    active = [r for r in recs if r["activity"] > 2.0]
    occupied = [r for r in recs if r.get("persons", 0) > 0]
    print(f"video: {index['video']['duration_s']:.0f}s @ {index['video']['fps']:.1f}fps "
          f"{index['video']['width']}x{index['video']['height']}")
    print(f"sampled {index['frames_sampled']} frames (interval {index['interval_s']}s) "
          f"in {index['build_seconds']}s")
    print(f"activity>2.0 in {len(active)}/{len(recs)} samples; "
          f"person visible in {len(occupied)}/{len(recs)} samples")
    print(f"index -> {out}")


if __name__ == "__main__":
    main()
