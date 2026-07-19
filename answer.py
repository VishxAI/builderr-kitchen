"""Kitchen CCTV monitor agent — builderr Round 1.

Entry point matching the challenge contract:

    python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json

Flow per video: build a coarse activity index (motion + YOLO person detection,
local, zero cost), then answer that video's questions through the Engine
(targeted local-VLM inspection with bisection for timestamps). Budget caps
from SPEC.md are tracked and enforced; frames are the scarce resource.
"""

import argparse
import json
import time
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".ts"}

# Hard caps from SPEC.md, per 60 minutes of source video (scale proportionally).
COST_CAP_PER_60MIN_USD = 0.30
FRAME_CAP_PER_60MIN = 1500
WALLCLOCK_CAP_SECONDS = 25 * 60

# Fraction of the frame budget the coarse index may consume.
COARSE_BUDGET_FRACTION = 0.55


class Budget:
    """Tracks frames, model calls, and estimated cost against the scaled caps."""

    def __init__(self, total_video_seconds: float):
        self.total_video_seconds = total_video_seconds
        scale = max(total_video_seconds, 1.0) / 3600.0
        self.frame_cap = int(FRAME_CAP_PER_60MIN * scale)
        self.cost_cap_usd = COST_CAP_PER_60MIN_USD * scale
        self.frames_processed = 0
        self.model_calls = 0
        self.estimated_cost_usd = 0.0  # stays 0.0: all models run locally
        self.started = time.monotonic()

    @property
    def runtime_seconds(self) -> float:
        return time.monotonic() - self.started

    @property
    def frames_remaining(self) -> int:
        return max(0, self.frame_cap - self.frames_processed)

    @property
    def time_exceeded(self) -> bool:
        scale = max(self.total_video_seconds, 1.0) / 3600.0
        return self.runtime_seconds > 0.85 * WALLCLOCK_CAP_SECONDS * scale

    @property
    def normalized_cost_per_60min(self) -> float:
        hours = max(self.total_video_seconds, 1.0) / 3600.0
        return self.estimated_cost_usd / hours

    def to_run_log(self) -> dict:
        return {
            "runtime_seconds": round(self.runtime_seconds, 1),
            "frames_processed": self.frames_processed,
            "model_calls": self.model_calls,
            "estimated_model_api_cost_usd": round(self.estimated_cost_usd, 4),
            "normalized_model_api_cost_per_60min_usd": round(self.normalized_cost_per_60min, 4),
        }


def discover_videos(videos_dir: Path) -> dict:
    """Map video_id (stem) -> path for every video file in the directory."""
    return {
        p.stem: p
        for p in sorted(videos_dir.iterdir())
        if p.suffix.lower() in VIDEO_EXTS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kitchen CCTV question-answering agent")
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()

    from pipeline.coarse import build_index
    from pipeline.engine import Engine
    from pipeline.ingest import probe

    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    videos = discover_videos(args.videos)

    # Only videos actually referenced by questions count toward the budget.
    referenced = {q.get("video_id") for q in questions}
    active = {vid: p for vid, p in videos.items() if vid in referenced}
    total_seconds = sum(probe(p)["duration_s"] for p in active.values()) or 3600.0
    budget = Budget(total_video_seconds=total_seconds)

    engines: dict[str, Engine] = {}
    answers = []
    for q in questions:
        vid = q.get("video_id")
        if vid not in active:
            answers.append({"id": q["id"], "answer": "not_visible",
                            "confidence": 0.0, "evidence": []})
            continue
        if vid not in engines:
            path = active[vid]
            duration = probe(path)["duration_s"]
            # size the coarse interval so the index stays inside its budget share
            coarse_frames = max(1, int(budget.frame_cap
                                       * (duration / budget.total_video_seconds)
                                       * COARSE_BUDGET_FRACTION))
            interval = max(4.0, duration / coarse_frames)
            index = build_index(path, interval_s=interval)
            budget.frames_processed += index["frames_sampled"]
            engines[vid] = Engine(path, index, budget)
        answers.append(engines[vid].answer(q))

    args.out.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    args.log.write_text(json.dumps(budget.to_run_log(), indent=2), encoding="utf-8")
    print(f"Answered {len(answers)} questions over {len(active)} videos -> {args.out}")
    print(json.dumps(budget.to_run_log(), indent=2))


if __name__ == "__main__":
    main()
