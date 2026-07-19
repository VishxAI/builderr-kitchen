"""Question-answering engine: routes each question to a strategy.

Strategies by question type:
- yes_no / count / state at a specific timestamp -> frames around t -> VLM
  (count questions cross-check the YOLO index).
- timestamp ("when did X first happen") -> probe scan biased to active
  windows, then bisect the transition to within the +/-2 s tolerance.
- duration ("how long was X on Y") -> locate state start, then state end.
- multiple choice / event order -> one shared spread of frames, one VLM
  call per frame covering all candidate events, first-yes per event.
- OCR questions (order number / label / receipt / screen text) -> EasyOCR
  on frames near t or the most active moments.
- Anything below confidence threshold -> not_visible.

All perception is local (YOLO index + Qwen3-VL + EasyOCR). No API calls.
The frame budget is the scarce resource: every strategy checks
budget.frames_remaining and degrades before exceeding it.
"""

import os
import re
from pathlib import Path

import cv2

TRACE = bool(os.environ.get("ENGINE_TRACE"))

OCR_KEYWORDS = ("order number", "receipt", "ticket", "label", "screen", "display",
                "number visible", "text visible", "written")

# events about people appearing/leaving can be localized via the YOLO index
PERSON_EVENT = re.compile(
    r"\b(person|people|man|woman|delivery|courier|customer|visitor|staff|worker|someone|anyone)\b",
    re.I)

# container words that a scan candidate must be verified against - the VLM
# happily calls food in a frying pan "a bowl of food" unless asked directly
CONTAINER_WORDS = ("bowl", "plate", "pan", "pot", "box", "bag", "tray",
                   "container", "barrel", "bucket", "basket", "jug")


# ---------------------------------------------------------------- utilities

def parse_time_ref(text: str) -> float | None:
    """Extract a timestamp reference like '12:34', '1:02:03' or '754s' from text."""
    m = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", text)
    if m:
        h, mnt, s = map(int, m.groups())
        return h * 3600 + mnt * 60 + s
    m = re.search(r"\b(\d{1,3}):(\d{2})\b", text)
    if m:
        mnt, s = map(int, m.groups())
        return mnt * 60 + s
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*s(?:ec(?:onds)?)?\b", text, re.I)
    if m:
        return float(m.group(1))
    return None


def parse_options(question: dict) -> list[str]:
    """Options from an explicit field, 'A) .. B) ..' lettering, or a trailing list."""
    if isinstance(question.get("options"), list) and question["options"]:
        return [str(o) for o in question["options"]]
    text = question.get("question", "")
    lettered = re.findall(r"[A-Fa-f][\)\.]\s*([^A-Fa-f\)\.][^;A-F]*?)(?=\s+[A-Fa-f][\)\.]|\s*$)", text)
    if len(lettered) >= 2:
        return [o.strip(" .,;") for o in lettered]
    m = re.search(r":\s*(.+?)\??$", text)
    if m:
        parts = [p.strip(" .?") for p in re.split(r",| then | and ", m.group(1)) if p.strip(" .?")]
        if len(parts) >= 2:
            return parts
    return []


def normalize_state(desc: str) -> str:
    """Rewrite event phrasing as state phrasing so single-frame probes work.

    'delivery person first entered the kitchen' asked at mid-presence reads as
    'is entering right now?' -> no. State phrasing ('is present in') is what a
    single frame can actually answer.
    """
    d = re.sub(r"\bfirst\b|\binitially\b", "", desc, flags=re.I)
    d = re.sub(r"\b(enters?|entered|arrives?|arrived|appears?|appeared|"
               r"walk(?:s|ed)? in(?:to)?|c[ao]mes? in(?:to)?)\b",
               "is present in", d, flags=re.I)
    d = re.sub(r"\bis present in\s+(the\s+)?(kitchen|room|frame|view)\b",
               r"is present in \1\2", d, flags=re.I)
    d = re.sub(r"\b(?:was|were|is|being)?\s*placed\s+(on|at|in|near)\b",
               r"is \1", d, flags=re.I)
    return re.sub(r"\s+", " ", d).strip()


def grab_frame(video_path: Path, t: float):
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def yes_no_from_text(answer: str) -> tuple[str | None, float]:
    """Map a VLM free-text answer to yes/no with a crude confidence."""
    a = answer.lower()
    if re.match(r"^\s*(yes|yeah|correct|true)\b", a):
        return "yes", 0.85
    if re.match(r"^\s*(no|not|false)\b", a):
        return "no", 0.85
    if "yes" in a and "no" not in a:
        return "yes", 0.6
    if "no" in a and "yes" not in a:
        return "no", 0.6
    return None, 0.0


def count_from_text(answer: str) -> int | None:
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "no": 0}
    m = re.search(r"\b(\d+)\b", answer)
    if m:
        return int(m.group(1))
    for w, v in words.items():
        if re.search(rf"\b{w}\b", answer.lower()):
            return v
    return None


# ---------------------------------------------------------------- engine

class Engine:
    """Answers questions for one video using its coarse index + local VLM/OCR."""

    _qtype = ""  # set per-question in answer(); default for direct strategy calls

    def __init__(self, video_path: Path, index: dict, budget):
        self.video_path = video_path
        self.index = index
        self.budget = budget
        self.duration = index["video"]["duration_s"]

    # -- frame access with budget accounting

    def frames_at(self, times: list[float]):
        frames, kept_times = [], []
        for t in times:
            if self.budget.frames_remaining <= 0:
                break
            f = grab_frame(self.video_path, min(max(t, 0), self.duration - 0.1))
            if f is not None:
                frames.append(f)
                kept_times.append(t)
                self.budget.frames_processed += 1
        return kept_times, frames

    def vlm_ask(self, frames, question: str, max_new_tokens: int = 96) -> str:
        from . import vlm

        self.budget.model_calls += 1
        return vlm.ask(frames, question, max_new_tokens=max_new_tokens)

    def active_times(self, n: int) -> list[float]:
        """n timestamps spread over the video, biased to high-activity samples."""
        recs = self.index["records"]
        if not recs:
            return [self.duration * (i + 0.5) / n for i in range(n)]
        # split video into n equal bins, take the most active sample in each
        out = []
        bin_len = self.duration / n
        for i in range(n):
            lo, hi = i * bin_len, (i + 1) * bin_len
            in_bin = [r for r in recs if lo <= r["t"] < hi]
            if in_bin:
                out.append(max(in_bin, key=lambda r: r["activity"])["t"])
            else:
                out.append((lo + hi) / 2)
        return out

    # -- region cropping: zoom into the area a question names

    _REGIONS = {
        "left": (0.0, 0.55, 0.0, 1.0),
        "right": (0.45, 1.0, 0.0, 1.0),
        "back": (0.0, 1.0, 0.0, 0.6),
        "top": (0.0, 1.0, 0.0, 0.6),
        "front": (0.0, 1.0, 0.4, 1.0),
        "bottom": (0.0, 1.0, 0.4, 1.0),
        "center": (0.2, 0.8, 0.2, 0.8),
        "middle": (0.2, 0.8, 0.2, 0.8),
    }

    def _with_zoom(self, frames: list, desc: str) -> list:
        """If desc names a region, append an upscaled crop of it to the frame list."""
        d = desc.lower()
        for key, (x0, x1, y0, y1) in self._REGIONS.items():
            if re.search(rf"\b{key}\b", d):
                out = list(frames)
                for f in frames:
                    h, w = f.shape[:2]
                    crop = f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
                    out.append(cv2.resize(crop, None, fx=2.0, fy=2.0,
                                          interpolation=cv2.INTER_CUBIC))
                return out
        return frames

    # -- strategies

    def at_time_yes_no(self, t: float, question: str) -> dict:
        _, frames = self.frames_at([t])
        if not frames:
            return {"answer": "not_visible", "confidence": 0.0, "evidence": []}
        # no 'unclear' escape here: it flips negation questions ("is X empty?");
        # an unparseable answer already falls through to the retry frame
        prompt = (f"Look at this kitchen CCTV frame. {question} "
                  "Answer yes or no, then one short sentence of justification.")
        raw = self.vlm_ask(self._with_zoom(frames, question), prompt)
        ans, conf = yes_no_from_text(raw)
        if ans is None or "unclear" in raw.lower():
            # second opinion on a nearby frame before giving up
            _, frames2 = self.frames_at([t + 1.5])
            if frames2:
                raw = self.vlm_ask(self._with_zoom(frames2, question), prompt)
                ans, conf = yes_no_from_text(raw)
        if ans is None or "unclear" in raw.lower():
            return {"answer": "not_visible", "confidence": 0.3, "evidence": self._ev(t - 1, t + 1)}
        return {"answer": ans, "confidence": conf, "evidence": self._ev(t - 1, t + 1)}

    def at_time_count(self, t: float, question: str) -> dict:
        _, frames = self.frames_at([t])
        if not frames:
            return {"answer": "not_visible", "confidence": 0.0, "evidence": []}
        raw = self.vlm_ask(frames, f"{question} Answer with a single number only.", max_new_tokens=16)
        n = count_from_text(raw)
        yolo_n = self._yolo_persons_near(t)
        if n is None:
            n = yolo_n
        conf = 0.85 if (yolo_n is not None and n == yolo_n) else 0.6
        if n is None:
            return {"answer": "not_visible", "confidence": 0.2, "evidence": self._ev(t, t)}
        return {"answer": n, "confidence": conf, "evidence": self._ev(t, t)}

    def _probe(self, t: float, probe_q: str, desc: str = "",
               confirm_yes: bool = False) -> str | None:
        _, frames = self.frames_at([t])
        if not frames:
            return None
        ans, _ = yes_no_from_text(self.vlm_ask(self._with_zoom(frames, desc),
                                               probe_q, max_new_tokens=8))
        if TRACE:
            print(f"    probe t={t:7.1f} -> {ans}")
        if ans == "yes" and confirm_yes:
            # a false yes wrecks bisection; verify on a neighboring frame
            _, frames2 = self.frames_at([t + 1.5])
            if frames2:
                ans2, _ = yes_no_from_text(self.vlm_ask(self._with_zoom(frames2, desc),
                                                        probe_q, max_new_tokens=8))
                if ans2 != "yes":
                    return "no"
        return ans

    def _bisect(self, lo: float, hi: float, probe_q: str, desc: str = "") -> float:
        """Narrow the lo(no)->hi(yes) transition to <2 s.

        Returns the bracket midpoint: the true transition lies in (lo, hi], and
        detection lags onset slightly, so the midpoint beats the upper bound.
        """
        while hi - lo > 2.0 and self.budget.frames_remaining > 0:
            mid = (lo + hi) / 2
            ans = self._probe(mid, probe_q, desc)
            if ans is None:
                break
            if ans == "yes":
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def _person_windows(self) -> list[tuple[float, float]]:
        """Windows where the YOLO person count exceeds the video's modal count."""
        recs = self.index["records"]
        counts = [r.get("persons") for r in recs if r.get("persons") is not None]
        if not counts:
            return []
        modal = max(set(counts), key=counts.count)
        iv = self.index["interval_s"]
        wins: list[list[float]] = []
        for r in recs:
            p = r.get("persons")
            if p is not None and p > modal:
                if wins and r["t"] - wins[-1][1] <= 2.5 * iv:
                    wins[-1][1] = r["t"]
                else:
                    wins.append([r["t"], r["t"]])
        return [(a, b) for a, b in wins]

    def _person_event_window(self, probe_q: str, desc: str):
        """First index-flagged window the VLM confirms for a person event, or None."""
        iv = self.index["interval_s"]
        for start, end in self._person_windows():
            mid = (start + end) / 2
            if self._probe(mid, probe_q, desc, confirm_yes=True) == "yes":
                return start - iv, mid, end + iv
        return None

    def first_event_timestamp(self, event_desc: str, t_start: float = 0.0) -> dict:
        """Scan for the first frame where the event's state holds, then bisect."""
        state = normalize_state(event_desc)
        strict = ("" if PERSON_EVENT.search(event_desc) else
                  " Be strict about object type, size, and location: answer no "
                  "if what you see is a similar-looking but different object, for "
                  "example a plate, bowl, bottle, or jug instead of the described object.")
        probe_q = (
            "Look at this CCTV frame. Is the following currently visible or true: "
            f"{state}?{strict} Answer strictly yes or no."
        )
        # person appearances: the YOLO index pinpoints candidate windows directly
        if PERSON_EVENT.search(event_desc):
            hit = self._person_event_window(probe_q, event_desc)
            if hit is not None:
                lo, mid, _ = hit
                t = self._bisect(max(lo, 0.0), mid, probe_q, event_desc)
                return {"answer": round(t, 1), "confidence": 0.7,
                        "evidence": self._ev(lo, t)}
        n_probes = min(24, max(8, self.budget.frames_remaining // 4))
        times = [t for t in self.active_times(n_probes) if t >= t_start]
        prev_no = t_start
        for t in times:
            ans = self._probe(t, probe_q, event_desc, confirm_yes=True)
            if ans == "yes":
                cand = self._bisect(prev_no, t, probe_q, event_desc)
                # stability: a real onset persists; a look-alike carried past
                # the camera doesn't. Require the state to hold shortly after.
                stable = all(self._probe(cand + dt, probe_q, event_desc) == "yes"
                             for dt in (6.0, 12.0))
                if stable and self._container_matches(cand + 3.0, event_desc):
                    return {"answer": round(cand, 1), "confidence": 0.65,
                            "evidence": self._ev(prev_no, cand)}
                prev_no = t  # false alarm; keep scanning past it
            elif ans == "no":
                prev_no = t
        return {"answer": "not_visible", "confidence": 0.4, "evidence": []}

    def _container_matches(self, t: float, desc: str) -> bool:
        """Verify a scan candidate's container type by open decomposition.

        Yes/no probes accept food-in-a-pan as 'a bowl of food'; asking 'what
        container is it?' does not. Only gates descs that name a container.
        """
        wanted = [w for w in CONTAINER_WORDS if re.search(rf"\b{w}\b", desc, re.I)]
        if not wanted:
            return True
        _, frames = self.frames_at([t])
        if not frames:
            return True
        raw = self.vlm_ask(
            self._with_zoom(frames, desc),
            f"Focus on this: {normalize_state(desc)}. What type of container is "
            f"actually involved: {', '.join(CONTAINER_WORDS)}, or none? "
            "Answer with one word.",
            max_new_tokens=8,
        )
        return any(re.search(rf"\b{w}", raw, re.I) for w in wanted)

    def state_duration(self, state_desc: str) -> dict:
        """Duration a state holds: first time it becomes true -> first time it stops."""
        state_desc = normalize_state(state_desc)
        strict = ("" if PERSON_EVENT.search(state_desc) else
                  " Be strict about object type, size, and location: answer no if "
                  "what you see is a similar-looking but different object.")
        probe_q = (f"Look at this CCTV frame. Is this currently true: {state_desc}?"
                   f"{strict} Answer strictly yes or no.")
        # person presence: bound the state by the index-flagged window edges
        if PERSON_EVENT.search(state_desc):
            hit = self._person_event_window(probe_q, state_desc)
            if hit is not None:
                lo, mid, hi = hit
                start = self._bisect(max(lo, 0.0), mid, probe_q, state_desc)
                end = self._bisect_down(mid, min(hi, self.duration), probe_q, state_desc)
                return {"answer": round(end - start, 1), "confidence": 0.6,
                        "evidence": self._ev(start, end)}
        n_probes = min(30, max(8, self.budget.frames_remaining // 5))
        times = self.active_times(n_probes)
        start = end = None
        prev_no, prev_yes = 0.0, None
        for t in times:
            ans = self._probe(t, probe_q, state_desc, confirm_yes=(start is None))
            if ans == "yes" and start is None:
                start = self._bisect(prev_no, t, probe_q, state_desc)
                prev_yes = t
            elif ans == "yes" and start is not None:
                prev_yes = t
            elif ans == "no" and start is not None:
                end = self._bisect_down(prev_yes, t, probe_q, state_desc)
                break
            elif ans == "no":
                prev_no = t
        if start is None:
            return {"answer": "not_visible", "confidence": 0.4, "evidence": []}
        if end is None:
            end = self.duration
        return {"answer": round(end - start, 1), "confidence": 0.5,
                "evidence": self._ev(start, end)}

    def _bisect_down(self, lo: float, hi: float, probe_q: str, desc: str = "") -> float:
        """Narrow the lo(yes)->hi(no) transition to <2 s. Returns refined boundary."""
        while hi - lo > 2.0 and self.budget.frames_remaining > 0:
            mid = (lo + hi) / 2
            ans = self._probe(mid, probe_q, desc)
            if ans is None:
                break
            if ans == "yes":
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def order_events(self, events: list[str], question: str) -> dict:
        """First-occurrence time per event from one shared spread of frames."""
        n = min(20, max(10, self.budget.frames_remaining // (2 + len(events))))
        times = self.active_times(n)
        first_seen: dict[str, float] = {}
        listing = "; ".join(f"({i + 1}) {e}" for i, e in enumerate(events))
        for t in times:
            missing = [e for e in events if e not in first_seen]
            if not missing:
                break
            _, frames = self.frames_at([t])
            if not frames:
                continue
            raw = self.vlm_ask(
                frames,
                "Look at this kitchen CCTV frame. For each event, answer whether it has "
                f"already happened or is happening now: {listing}. "
                "Reply with one line per event in the form '1: yes' or '1: no'.",
            )
            for i, e in enumerate(events):
                if e in first_seen:
                    continue
                m = re.search(rf"\b{i + 1}\s*[:\)]\s*(yes|no)", raw, re.I)
                if m and m.group(1).lower() == "yes":
                    first_seen[e] = t
        if not first_seen:
            return {"answer": "not_visible", "confidence": 0.3, "evidence": []}
        ordered = sorted(first_seen, key=first_seen.get)
        unseen = [e for e in events if e not in first_seen]
        answer = ordered + unseen  # events never observed sort last
        conf = 0.55 if not unseen else 0.4
        t0 = min(first_seen.values())
        t1 = max(first_seen.values())
        if re.search(r"\bfirst\b", question, re.I) and not re.search(r"order|sequence", question, re.I):
            return {"answer": ordered[0], "confidence": conf, "evidence": self._ev(t0, t0)}
        if re.search(r"\blast\b", question, re.I) and not re.search(r"order|sequence", question, re.I):
            return {"answer": answer[-1], "confidence": conf, "evidence": self._ev(t1, t1)}
        return {"answer": answer, "confidence": conf, "evidence": self._ev(t0, t1)}

    def temporal_relation(self, qtext: str) -> dict:
        """'Did/was X ... before/after Y?' -> locate both events, compare times.

        Falls back to a plain multi-frame VLM vote when either event can't be
        localized (e.g. transient micro-events).
        """
        m = re.search(r"^(?:was|were|did|is|are|has|had)?\s*(.+?)\s+(before|after)\s+(.+?)\??$",
                      qtext.strip(), re.I)
        if m:
            desc_a, rel, desc_b = m.group(1), m.group(2).lower(), m.group(3)
            ra = self.first_event_timestamp(desc_a)
            rb = self.first_event_timestamp(desc_b)
            ta, tb = ra.get("answer"), rb.get("answer")
            if isinstance(ta, (int, float)) and isinstance(tb, (int, float)):
                ans = "yes" if ((ta < tb) == (rel == "before")) else "no"
                return {"answer": ans, "confidence": 0.6,
                        "evidence": ra["evidence"] + rb["evidence"]}
        # fallback: show ordered frames and ask directly
        times = self.active_times(4)
        _, frames = self.frames_at(times)
        if not frames:
            return {"answer": "not_visible", "confidence": 0.0, "evidence": []}
        raw = self.vlm_ask(
            frames,
            "These CCTV frames are in chronological order from the same kitchen. "
            f"{qtext} Answer yes, no, or unclear, then one short sentence.",
        )
        ans, conf = yes_no_from_text(raw)
        if ans is None or "unclear" in raw.lower():
            return {"answer": "not_visible", "confidence": 0.3, "evidence": []}
        return {"answer": ans, "confidence": conf, "evidence": self._ev(times[0], times[-1])}

    def multiple_choice(self, options: list[str], question: str) -> dict:
        """Non-temporal MC: show a few spread frames, have the VLM pick an option."""
        times = self.active_times(4)
        _, frames = self.frames_at(times)
        if not frames:
            return {"answer": "not_visible", "confidence": 0.0, "evidence": []}
        listing = "; ".join(f"({i + 1}) {o}" for i, o in enumerate(options))
        raw = self.vlm_ask(
            frames,
            f"These CCTV frames are from different moments of the same kitchen. {question} "
            f"Options: {listing}. Reply with only the number of the best option.",
            max_new_tokens=8,
        )
        m = re.search(r"\d+", raw)
        if m and 1 <= int(m.group()) <= len(options):
            return {"answer": options[int(m.group()) - 1], "confidence": 0.7,
                    "evidence": self._ev(times[0], times[-1])}
        return {"answer": "not_visible", "confidence": 0.2, "evidence": []}

    def ocr_question(self, t: float | None, question: str) -> dict:
        from . import ocr

        times = [t] if t is not None else self.active_times(6)
        best: list[dict] = []
        t_used = None
        vlm_reads: dict[str, list[float]] = {}
        for tt in times:
            _, frames = self.frames_at([tt])
            if not frames:
                continue
            self.budget.model_calls += 1
            hits = ocr.read_text(frames[0], min_confidence=0.4)
            if hits and (not best or hits[0]["confidence"] > best[0]["confidence"]):
                best, t_used = hits, tt
            # second reader: the VLM with a 2x upscale - catches small or
            # rotated text that trips EasyOCR
            f = frames[0]
            up = cv2.resize(f, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            raw = self.vlm_ask(
                [f, up],
                f"The second image is a magnified copy. {question} "
                "Answer with just the text you can read, or unclear.",
                max_new_tokens=16,
            ).strip().strip(".\"'")
            non_answer = re.search(r"unclear|unanswer|unknown|not (readable|visible|legible)|cannot|can't|no text",
                                   raw, re.I)
            if raw and not non_answer and len(raw) < 60:
                key = raw.lower()
                vlm_reads.setdefault(key, []).append(tt)
        digits = [h for h in best if re.search(r"\d", h["text"])]
        wants_number = re.search(r"\b(number|no\.|#)\b", question, re.I)
        visibility = re.search(r"\b(visible|readable|legible|can you (see|read))\b", question, re.I)
        if self._qtype == "yes_no" or visibility:
            found = bool(digits) if wants_number else bool(best or vlm_reads)
            ans = "yes" if found else "no"
            conf = 0.7 if found else 0.5
            ev = self._ev(t_used, t_used) if t_used is not None else []
            return {"answer": ans, "confidence": conf, "evidence": ev}
        # value questions: prefer a VLM reading confirmed at >1 timestamp,
        # else an OCR hit, else a single VLM reading
        repeated = [k for k, ts in vlm_reads.items() if len(ts) > 1]
        if repeated:
            k = max(repeated, key=lambda k: len(vlm_reads[k]))
            tt = vlm_reads[k][0]
            return {"answer": k, "confidence": 0.75, "evidence": self._ev(tt, tt)}
        if wants_number:
            source = digits
        else:
            # a name/word answer needs substance; 1-2 char OCR hits are noise
            source = [h for h in best
                      if len(h["text"].strip()) >= 3 and re.search(r"[a-zA-Z]", h["text"])]
        if source:
            return {"answer": source[0]["text"], "confidence": round(source[0]["confidence"], 2),
                    "evidence": self._ev(t_used, t_used)}
        # an uncorroborated single VLM reading is a guess; the rubric scores
        # not_visible as its own category, so abstain instead
        return {"answer": "not_visible", "confidence": 0.5, "evidence": []}

    # -- helpers

    def _yolo_persons_near(self, t: float) -> int | None:
        recs = self.index["records"]
        if not recs:
            return None
        nearest = min(recs, key=lambda r: abs(r["t"] - t))
        if abs(nearest["t"] - t) > self.index["interval_s"]:
            return None
        return nearest.get("persons")

    def _ev(self, t0: float, t1: float) -> list[dict]:
        vid = Path(self.index["video"]["path"]).stem
        return [{
            "video_id": vid,
            "timestamp_start": round(max(t0, 0), 1),
            "timestamp_end": round(min(t1, self.duration), 1),
        }]

    # -- dispatch

    def answer(self, question: dict) -> dict:
        qtype = question.get("type", "")
        qtext = question.get("question", "")
        self._qtype = qtype
        t = parse_time_ref(qtext)
        options = parse_options(question)

        # deadline guard: past 85% of the scaled wall-clock cap, stop spending
        # VLM time and answer from the index alone
        if self.budget.time_exceeded:
            if qtype == "count" and t is not None:
                n = self._yolo_persons_near(t)
                if n is not None:
                    res = {"answer": n, "confidence": 0.5, "evidence": self._ev(t, t)}
                    res["id"] = question["id"]
                    return res
            res = {"answer": "not_visible", "confidence": 0.1, "evidence": []}
            res["id"] = question["id"]
            return res

        if any(k in qtext.lower() for k in OCR_KEYWORDS):
            res = self.ocr_question(t, qtext)
        elif qtype == "count" and t is not None:
            res = self.at_time_count(t, qtext)
        elif qtype == "yes_no" and t is not None:
            res = self.at_time_yes_no(t, qtext)
        elif qtype == "timestamp":
            desc = re.sub(r"^(what is the timestamp of|when (was|did|does)|at what time (was|did))\s*",
                          "", qtext, flags=re.I).rstrip("?")
            res = self.first_event_timestamp(desc)
        elif qtype == "duration":
            desc = re.sub(r"^(for )?how long (was|did|does|is)\s*", "", qtext, flags=re.I).rstrip("?")
            res = self.state_duration(desc)
        elif qtype in ("multiple_choice", "event_order", "order") or (options and len(options) >= 2):
            if options and re.search(r"\border|sequence|first|last|before|after\b", qtext, re.I):
                res = self.order_events(options, qtext)
            elif options:
                res = self.multiple_choice(options, qtext)
            else:
                res = {"answer": "not_visible", "confidence": 0.2, "evidence": []}
        elif qtype == "yes_no" and re.search(r"\s(before|after)\s", qtext, re.I):
            res = self.temporal_relation(qtext)
        elif qtype == "yes_no":
            times = self.active_times(3)
            _, frames = self.frames_at(times)
            if not frames:
                res = {"answer": "not_visible", "confidence": 0.0, "evidence": []}
            else:
                raw = self.vlm_ask(
                    frames,
                    f"These CCTV frames are from different moments of the same kitchen. {qtext} "
                    "Answer strictly yes, no, or unclear, then one short sentence.",
                )
                ans, conf = yes_no_from_text(raw)
                if ans is None:
                    res = {"answer": "not_visible", "confidence": 0.3, "evidence": []}
                else:
                    res = {"answer": ans, "confidence": conf,
                           "evidence": self._ev(times[0], times[-1])}
        elif qtype == "count":
            # count without an anchor time: modal YOLO person count; when YOLO
            # sees nobody (top-down/hands-only footage), ask the VLM instead
            counts = [r.get("persons", 0) for r in self.index["records"]]
            modal = max(set(counts), key=counts.count) if counts else None
            if modal == 0 or modal is None:
                times = self.active_times(4)
                _, frames = self.frames_at(times)
                if frames:
                    raw = self.vlm_ask(
                        frames,
                        "These frames are from different moments of the same video. "
                        f"{qtext} Answer with a single number.",
                        max_new_tokens=8,
                    )
                    n = count_from_text(raw)
                    if n is not None:
                        modal = n
            if modal is not None:
                res = {"answer": modal, "confidence": 0.5,
                       "evidence": self._ev(0, self.duration)}
            else:
                res = {"answer": "not_visible", "confidence": 0.0, "evidence": []}
        else:
            res = {"answer": "not_visible", "confidence": 0.0, "evidence": []}

        res["id"] = question["id"]
        return res
