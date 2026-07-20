"""Targeted regression probes for organizer gold-set feedback (bff1ca2 -> next).

Soham's second review of commit bff1ca2 reported three specific failures:
  1. attribute count ("how many wearing X") returned wrong counts on a
     color-attribute predicate ("red shirt")
  2. a contact/touch question returned a false "yes"
  3. an unreadable label returned "no" instead of "not_visible"

Each function below reproduces the exact scenario against real footage and
states the expected answer, so a regression shows up as a printed mismatch
rather than requiring a full pipeline run. Requires cached *.index.json
files (built once by pipeline.coarse, see README) for sample_19min and
sample_22min.

    python scripts/probe_gold_feedback.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from answer import Budget
from pipeline.engine import Engine


def load(video_id: str) -> tuple[Engine, Budget]:
    idx = json.load(open(f"videos/{video_id}.index.json", encoding="utf-8-sig"))
    b = Budget(total_video_seconds=idx["video"]["duration_s"])
    return Engine(Path(f"videos/{video_id}.mp4"), idx, b), b


def check(label: str, got, expect) -> bool:
    ok = got == expect
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got={got!r} expect={expect!r}")
    return ok


def main() -> None:
    results = []

    # --- bug 1: color-attribute count must bind color to the right garment,
    # not just "some detected color word matches" (the frame at 11:00 shows
    # exactly one black-shirt cook and one red-shirt person)
    e, _ = load("sample_19min")
    e._qtype = "count"
    r = e.grounded_person_count(660.0, "How many people are wearing a black shirt at 11:00?")
    results.append(check("black-shirt count @11:00", r["answer"], 1))
    r = e.grounded_person_count(660.0, "How many people are wearing a red shirt at 11:00?")
    results.append(check("red-shirt count @11:00", r["answer"], 1))

    # --- bug 2: contact question must not false-positive on proximity, and
    # must still catch genuine bare-hand contact (portrait close-up video)
    e, _ = load("sample_19min")
    e._qtype = "yes_no"
    r = e.touch_burst(660.0, "Did the cook touch the wok handle with bare hands at 11:00?")
    results.append(check("bare-hand wok touch @11:00 (tool use, not bare hands)", r["answer"], "no"))

    e, _ = load("sample_22min")
    e._qtype = "yes_no"
    r = e.touch_burst(240.0, "Is a hand touching the diced meat with bare hands at 4:00?")
    results.append(check("bare-hand contact @4:00 (genuine positive)", r["answer"], "yes"))

    # --- bug 3: illegible-but-present text must abstain, not deny
    e, _ = load("sample_22min")
    e._qtype = "yes_no"
    r = e.ocr_question(60.0, "Is the brand name on the metal gas stove readable at 1:00?")
    results.append(check("illegible stove brand @1:00", r["answer"], "not_visible"))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} probes passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
