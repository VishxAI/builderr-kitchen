"""Local scoring harness mirroring the challenge rubric.

Compares answers.json against a ground-truth file of the same shape
(list of {"id", "answer", optional "tolerance_s"}). Scoring:
- yes_no / multiple choice / string answers: exact match (case-insensitive)
- counts: exact match
- timestamps: full credit within ±2 s, half credit within ±5 s
  (or the question's own "tolerance_s" when the event spans longer)
- not_visible: exact match against a ground-truth "not_visible"

Usage:
    python scripts/score.py answers.json truth.json
"""

import argparse
import json
from pathlib import Path


def score_pair(pred, truth) -> tuple[float, str]:
    t_ans = truth["answer"]
    p_ans = pred.get("answer")

    if isinstance(t_ans, str):
        ok = isinstance(p_ans, str) and p_ans.strip().lower() == t_ans.strip().lower()
        return (1.0 if ok else 0.0), ("exact" if ok else f"want {t_ans!r} got {p_ans!r}")

    if isinstance(t_ans, bool):
        return (1.0 if p_ans == t_ans else 0.0), f"want {t_ans} got {p_ans}"

    if isinstance(t_ans, int):
        return (1.0 if p_ans == t_ans else 0.0), f"want {t_ans} got {p_ans}"

    if isinstance(t_ans, float):  # timestamp or duration in seconds
        if not isinstance(p_ans, (int, float)):
            return 0.0, f"want ~{t_ans}s got {p_ans!r}"
        diff = abs(float(p_ans) - t_ans)
        full = truth.get("tolerance_s", 2.0)
        partial = max(5.0, full)
        if diff <= full:
            return 1.0, f"off by {diff:.1f}s (full credit)"
        if diff <= partial:
            return 0.5, f"off by {diff:.1f}s (partial credit)"
        return 0.0, f"off by {diff:.1f}s"

    return 0.0, f"unhandled truth type {type(t_ans).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score answers against ground truth")
    parser.add_argument("answers", type=Path)
    parser.add_argument("truth", type=Path)
    args = parser.parse_args()

    preds = {a["id"]: a for a in json.loads(args.answers.read_text(encoding="utf-8-sig"))}
    truths = json.loads(args.truth.read_text(encoding="utf-8-sig"))

    total = 0.0
    for t in truths:
        pred = preds.get(t["id"], {})
        pts, detail = score_pair(pred, t)
        total += pts
        conf = pred.get("confidence", "-")
        print(f"{t['id']}: {pts:.1f}  ({detail})  conf={conf}")

    print(f"\nscore: {total:.1f}/{len(truths)} = {100 * total / max(len(truths), 1):.0f}%")


if __name__ == "__main__":
    main()
