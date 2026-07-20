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

# counts filtered by a person attribute need per-person grounding, not a
# whole-frame guess ("how many staff wear hair covers?")
ATTRIBUTE_COUNT = re.compile(
    r"\b(wear(?:s|ing)?|hair\s?cover|hairnet|hat|cap|glove|apron|mask|helmet|"
    r"headwear|uniform|shirt|jacket|beard\s?net)\b", re.I)

# contact judgments need a dense frame burst, not a single sample
TOUCH_VERBS = re.compile(
    r"\b(touch(?:es|ed|ing)?|contact|bare\s?hands?|handl(?:e|es|ed|ing)|"
    r"grab(?:s|bed|bing)?|pick(?:s|ed)?\s?up)\b", re.I)

# a "wearing a {color} {garment}" predicate needs the color bound to the
# right garment -- the 2B model's yes/no verdict happily says "yes" to
# "black shirt" when only the person's cap is black, even though its own
# free-text description correctly separates "red shirt" from "black cap"
COLOR_WORDS = ("red", "black", "white", "blue", "green", "yellow", "orange",
               "pink", "purple", "gray", "grey", "brown", "navy", "maroon")
GARMENT_WORDS = ("shirt", "t-shirt", "tshirt", "cap", "hat", "apron", "glove",
                 "jacket", "uniform", "hairnet", "mask", "helmet", "vest")


def color_garment_predicate(predicate: str) -> tuple[str, str] | None:
    """Extract (color, garment) from a predicate like 'wearing a red shirt'."""
    for color in COLOR_WORDS:
        if re.search(rf"\b{color}\b", predicate, re.I):
            for garment in GARMENT_WORDS:
                if re.search(rf"\b{garment}s?\b", predicate, re.I):
                    return color, garment
    return None


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


def verdict_from_text(answer: str) -> tuple[str | None, float]:
    """Parse a describe-then-verdict response.

    Small VLMs sometimes blurt a snap 'yes' as the very first token, then
    write an accurate description that actually contradicts it ("...so the
    statement is false"). An explicit trailing 'Answer: yes/no' marker (which
    callers should prompt for) is trusted first; failing that, the *last*
    yes/no token beats the first, since self-correction happens after the
    initial guess, not before it.
    """
    m = re.search(r"answer\s*:?\s*(yes|no)\b", answer, re.I)
    if m:
        return m.group(1).lower(), 0.85
    matches = list(re.finditer(r"\b(yes|no)\b", answer, re.I))
    if matches:
        return matches[-1].group(1).lower(), 0.6
    return yes_no_from_text(answer)


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

    def _record_near(self, t: float):
        recs = self.index["records"]
        if not recs:
            return None
        nearest = min(recs, key=lambda r: abs(r["t"] - t))
        if abs(nearest["t"] - t) > 2 * self.index["interval_s"]:
            return None
        return nearest

    def _person_frame_near(self, t: float, window: float = 12.0):
        """Best nearby index record with detected people, not just the nearest sample.

        YOLO can miss a person in the single frame closest to t (motion blur,
        partial occlusion) while a neighbor a couple seconds away catches them.
        """
        recs = self.index["records"]
        candidates = [r for r in recs if abs(r["t"] - t) <= window and r.get("person_boxes")]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (-len(r["person_boxes"]), abs(r["t"] - t)))

    def grounded_person_count(self, t: float | None, question: str) -> dict:
        """Attribute-filtered person count: crop each detected person and ask
        the VLM about each crop individually, then count the yeses."""
        predicate = re.sub(
            r"^how many\s+(people|persons?|staff(?:\s+members?)?|workers?|cooks?|employees?)?"
            r"\s*(are|were|is)?\s*", "", question.strip(), flags=re.I).rstrip("?")
        # a trailing time reference just confuses the per-crop predicate; the
        # frame we pick already anchors the moment
        predicate = re.sub(r"\s*(at\s+\d{1,2}:\d{2}(:\d{2})?|right now|currently)\s*$",
                           "", predicate, flags=re.I).strip()
        if t is not None:
            cand = [self._person_frame_near(t) or self._person_frame_near(t, window=25.0)]
        else:
            cand = [self._person_frame_near(tt) for tt in self.active_times(3)]
        cand = [r for r in cand if r and r.get("person_boxes")]
        if not cand:
            # genuinely nobody detected near the question's moment: fall back
            # to a blind whole-frame VLM count rather than guessing zero
            return self.at_time_count(t if t is not None else self.duration / 2, question)
        counts = []
        used_t = None
        for rec in cand:
            _, frames = self.frames_at([rec["t"]])
            if not frames:
                continue
            frame = frames[0]
            h, w = frame.shape[:2]
            boxes = rec["person_boxes"]
            crops = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                # generous margin: attribute cues (shirt color, headwear) sit
                # near the box edge and a tight crop clips them off -- but
                # stop at the midpoint to any neighboring person so a crop
                # never bleeds in another person's clothing
                mx, my = 0.3 * (x2 - x1), 0.3 * (y2 - y1)
                ex1, ey1, ex2, ey2 = x1 - mx, y1 - my, x2 + mx, y2 + my
                for j, (ox1, oy1, ox2, oy2) in enumerate(boxes):
                    if j == i:
                        continue
                    if ox2 <= x1:
                        ex1 = max(ex1, (x1 + ox2) / 2)
                    if ox1 >= x2:
                        ex2 = min(ex2, (x2 + ox1) / 2)
                    if oy2 <= y1:
                        ey1 = max(ey1, (y1 + oy2) / 2)
                    if oy1 >= y2:
                        ey2 = min(ey2, (y2 + oy1) / 2)
                cx1, cy1 = max(0, int(ex1)), max(0, int(ey1))
                cx2, cy2 = min(w, int(ex2)), min(h, int(ey2))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                if crop.shape[0] < 320:
                    s = 320 / crop.shape[0]
                    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                crops.append(crop)
            if not crops:
                continue
            # query each crop in its own call: the 2B model is unreliable at
            # discriminating between multiple images sharing one prompt (it
            # tends to repeat the same verdict for every line regardless of
            # what's actually in each image)
            cg = color_garment_predicate(predicate)
            n = 0
            for crop in crops:
                if cg:
                    # ask for the specific garment's color directly instead
                    # of a yes/no verdict: the model's own verdict happily
                    # says "yes, black shirt" when only the cap is black,
                    # even though it can correctly name each garment's color
                    # when asked one at a time
                    color, garment = cg
                    resp = self.vlm_ask(
                        crop,
                        f"This is one person cropped from a kitchen CCTV frame, at "
                        f"close range. What color is their {garment}? Answer with "
                        f"just the color word, or 'none' if they are not wearing a "
                        f"{garment}.",
                        max_new_tokens=8,
                    ).strip().lower()
                    synonyms = {"gray": "grey", "grey": "gray"}
                    ans = "yes" if (re.search(rf"\b{color}\b", resp)
                                    or re.search(rf"\b{synonyms.get(color, color)}\b", resp)) else "no"
                    if TRACE:
                        print(f"    crop {garment} color: {resp!r} -> {ans}")
                else:
                    # describe first, verdict last: a snap first-token 'yes'
                    # from this model often gets contradicted by its own
                    # description
                    single = self.vlm_ask(
                        crop,
                        "This is one person cropped from a kitchen CCTV frame, at "
                        "close range. First, in one sentence, describe their "
                        "visible clothing colors and headwear. Then on a new line "
                        f"write exactly 'Answer: yes' or 'Answer: no' for whether "
                        f"this is true of them: {predicate}.",
                        max_new_tokens=48,
                    )
                    ans, _ = verdict_from_text(single)
                    if TRACE:
                        print(f"    crop verdict: {single!r} -> {ans}")
                if ans == "yes":
                    n += 1
            counts.append(n)
            used_t = rec["t"]
        if not counts:
            return {"answer": "not_visible", "confidence": 0.2, "evidence": []}
        modal = max(set(counts), key=counts.count)
        return {"answer": modal, "confidence": 0.7, "evidence": self._ev(used_t, used_t)}

    def _locate_object(self, frame, obj: str):
        """VLM grounding with self-verification. Returns a pixel box or None."""
        raw = self.vlm_ask(
            frame,
            f"Locate {obj} in this image. Reply only with its bounding box as "
            "[x1, y1, x2, y2] in 0-1000 normalized coordinates, or none if not present.",
            max_new_tokens=32,
        )
        m = re.search(r"\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?", raw)
        if not m:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        mx, my = 0.4 * (x2 - x1), 0.4 * (y2 - y1)
        px1, py1 = max(0, int((x1 - mx) / 1000 * w)), max(0, int((y1 - my) / 1000 * h))
        px2, py2 = min(w, int((x2 + mx) / 1000 * w)), min(h, int((y2 + my) / 1000 * h))
        if px2 - px1 < 30 or py2 - py1 < 30:
            return None
        crop = frame[py1:py2, px1:px2]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # the 2B grounder sometimes latches onto a similar-looking object;
        # verify before trusting the crop
        check, _ = yes_no_from_text(self.vlm_ask(
            crop, f"Does this image crop clearly show {obj}? Answer yes or no.",
            max_new_tokens=8))
        return (px1, py1, px2, py2) if check == "yes" else None

    def touch_burst(self, t: float | None, question: str) -> dict:
        """Contact/touch judgment over a dense frame burst around the moment.

        If the question names a findable object, the burst is cropped to a
        verified grounding of it; otherwise full frames are used. The prompt
        requires the action to be actually seen — 'any frame' phrasing makes
        the model hallucinate contact with objects that aren't even in view.
        """
        obj = None
        m = re.search(TOUCH_VERBS.pattern + r"\s+(the|a|an)?\s*([^,?]+?)(?:\s+at\s+\d|\s+with\s+|\?|$)",
                      question, re.I)
        if m:
            det = m.group(len(m.groups()) - 1) or "the"
            obj = f"{det} {m.group(len(m.groups()))}".strip()
        anchors = [t] if t is not None else self.active_times(3)
        votes = []
        ev = []
        for a in anchors:
            times = [a + dt for dt in (-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0)]
            kept, frames = self.frames_at(times)
            if not frames:
                continue
            box = None
            if obj:
                box = self._locate_object(frames[len(frames) // 2], obj)
            if box:
                # contact needs person AND object in view: union the object box
                # with any detected person boxes before cropping
                x1, y1, x2, y2 = box
                rec = self._record_near(a)
                if rec and rec.get("person_boxes"):
                    h, w = frames[0].shape[:2]
                    pd = min(
                        rec["person_boxes"],
                        key=lambda p: abs((p[0] + p[2]) / 2 - (x1 + x2) / 2)
                                      + abs((p[1] + p[3]) / 2 - (y1 + y2) / 2),
                    )
                    x1, y1 = min(x1, int(pd[0]) - 10), min(y1, int(pd[1]) - 10)
                    x2, y2 = max(x2, int(pd[2]) + 10), max(y2, int(pd[3]) + 10)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                frames = [cv2.resize(f[y1:y2, x1:x2], None, fx=2.0, fy=2.0,
                                     interpolation=cv2.INTER_CUBIC) for f in frames]
            raw = self.vlm_ask(
                frames,
                "These are consecutive CCTV frames spanning about six seconds, in order. "
                f"{question} Answer yes only if you can actually see direct physical "
                "contact happen in these frames, not just proximity or reaching toward "
                "it. If the object or the contact itself is not clearly visible in any "
                "frame, answer no. Answer yes or no, then one short sentence.",
            )
            ans, _ = yes_no_from_text(raw)
            if ans == "yes":
                # a single lenient read is not enough evidence for a contact
                # claim: re-ask with an adversarial framing that explicitly
                # offers "near but not touching" as the honest answer
                confirm = self.vlm_ask(
                    frames,
                    "Look very carefully at these consecutive CCTV frames again. "
                    f"{question} Contact means the hand or object surface is "
                    "physically touching, with no visible gap. Being close, "
                    "reaching toward, or hovering over it does NOT count as "
                    "contact. If you cannot clearly see the point of contact "
                    "itself, answer no. Answer yes or no, then one short sentence.",
                )
                cans, _ = yes_no_from_text(confirm)
                if cans != "yes":
                    ans = "no"
            if ans is not None:
                votes.append(ans)
                if ans == "yes":
                    ev = self._ev(a - 3, a + 3)
        if not votes:
            return {"answer": "not_visible", "confidence": 0.3, "evidence": []}
        if "yes" in votes:
            return {"answer": "yes", "confidence": 0.75 if t is not None else 0.6,
                    "evidence": ev}
        return {"answer": "no", "confidence": 0.7 if t is not None else 0.5,
                "evidence": self._ev(anchors[0] - 3, anchors[-1] + 3)}

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

    def multiple_choice(self, options: list[str], question: str, t: float | None = None) -> dict:
        """Non-temporal MC: show a few frames, have the VLM pick an option.

        When the question names a moment, sample near it instead of a
        whole-video activity spread -- a question about "the gloves at
        2:10" is meaningless if none of the sampled frames are anywhere
        near 2:10.
        """
        times = [t - 1.5, t, t + 1.5] if t is not None else self.active_times(4)
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

    def _overlay_texts(self, near: float) -> set[str]:
        """Text burned into the video (timestamps, channel watermarks).

        Anything OCR also reads at a control frame far from the moment of
        interest is a static overlay, not scene content.
        """
        from . import ocr

        control = (near + self.duration / 2) % self.duration
        _, frames = self.frames_at([control])
        if not frames:
            return set()
        self.budget.model_calls += 1
        return {re.sub(r"\W+", "", h["text"]).lower()
                for h in ocr.read_text(frames[0], min_confidence=0.3)}

    def ocr_question(self, t: float | None, question: str) -> dict:
        from . import ocr

        # a single frame's VLM reading is an unreliable witness on its own
        # (a 2B model will confidently guess a plausible-looking brand name);
        # sample a short burst even for an anchored timestamp so a reading
        # needs to recur before it's trusted
        times = [t, t + 1.0, t + 2.0] if t is not None else self.active_times(6)
        overlays = self._overlay_texts(times[0])  # watermarks/timestamps burned into every frame
        best: list[dict] = []
        t_used = None
        vlm_reads: dict[str, list[float]] = {}
        digit_seen: dict[str, list[float]] = {}
        illegible_hint = False  # something is there but nobody could read it
        for tt in times:
            _, frames = self.frames_at([tt])
            if not frames:
                continue
            self.budget.model_calls += 1
            hits = ocr.read_text(frames[0], min_confidence=0.4)
            trusted = [h for h in hits if re.sub(r"\W+", "", h["text"]).lower() not in overlays]
            if trusted and (not best or trusted[0]["confidence"] > best[0]["confidence"]):
                best, t_used = trusted, tt
            for h in trusted:
                if len(re.sub(r"\D", "", h["text"])) >= 2:
                    digit_seen.setdefault(
                        re.sub(r"\W+", "", h["text"]).lower(), []).append(tt)
            # a weak pass below the trust threshold still tells us characters
            # are present even when we can't trust their content. The small
            # VLM cannot reliably self-report "illegible" vs "nothing here"
            # (it calls text it can't parse "none" just as often as it calls
            # it "illegible"), so this OCR signal -- not the model's wording --
            # is what decides the difference between not_visible and no.
            if not trusted:
                weak = ocr.read_text(frames[0], min_confidence=0.05)
                weak = [h for h in weak if re.sub(r"\W+", "", h["text"]).lower() not in overlays]
                if weak:
                    illegible_hint = True
            # second reader: the VLM with a 2x upscale - catches small or
            # rotated text that trips EasyOCR
            f = frames[0]
            up = cv2.resize(f, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            raw = self.vlm_ask(
                [f, up],
                f"The second image is a magnified copy. {question} Answer with just "
                "the text you can read, or 'unclear' if you cannot make it out.",
                max_new_tokens=16,
            ).strip().strip(".\"'")
            non_answer = re.search(
                r"unclear|unanswer|unknown|not (readable|visible|legible)|"
                r"cannot|can'?t|no text|^none$", raw, re.I)
            if raw and not non_answer and len(raw) < 60:
                key = raw.lower()
                vlm_reads.setdefault(key, []).append(tt)
                if len(re.sub(r"\D", "", raw)) >= 2:
                    digit_seen.setdefault(
                        re.sub(r"\W+", "", raw).lower(), []).append(tt)
        wants_number = re.search(r"\b(number|no\.|#)\b", question, re.I)
        visibility = re.search(r"\b(visible|readable|legible|can you (see|read))\b", question, re.I)
        if self._qtype == "yes_no" or visibility:
            # a lone VLM guess is not evidence: require either an OCR hit
            # (best) or the same reading recurring across sampled frames
            corroborated_read = any(len(set(ts)) > 1 for ts in vlm_reads.values())
            found = bool(digit_seen) if wants_number else bool(best or corroborated_read)
            ev = self._ev(t_used, t_used) if t_used is not None else []
            if found:
                return {"answer": "yes", "confidence": 0.7, "evidence": ev}
            if illegible_hint or vlm_reads:
                # something is there but nobody -- OCR or VLM -- could read
                # it confidently or consistently; that's not the same as
                # confidently no text (an uncorroborated single guess in
                # vlm_reads is itself evidence something was there to guess at)
                return {"answer": "not_visible", "confidence": 0.4, "evidence": ev}
            return {"answer": "no", "confidence": 0.5, "evidence": ev}
        if wants_number:
            # a number answer must recur across sampled moments and not be a
            # burned-in overlay; otherwise the text is not reliably readable
            repeated = [k for k, ts in digit_seen.items() if len(set(ts)) > 1]
            if repeated:
                k = max(repeated, key=lambda k: len(digit_seen[k]))
                tt = digit_seen[k][0]
                return {"answer": k, "confidence": 0.7, "evidence": self._ev(tt, tt)}
            return {"answer": "not_visible", "confidence": 0.5, "evidence": []}
        # word/name value questions: prefer a VLM reading confirmed at >1
        # timestamp, else a substantive OCR hit
        repeated = [k for k, ts in vlm_reads.items() if len(ts) > 1]
        if repeated:
            k = max(repeated, key=lambda k: len(vlm_reads[k]))
            tt = vlm_reads[k][0]
            return {"answer": k, "confidence": 0.75, "evidence": self._ev(tt, tt)}
        source = [h for h in best
                  if len(h["text"].strip()) >= 3 and re.search(r"[a-zA-Z]", h["text"])
                  and re.sub(r"\W+", "", h["text"]).lower() not in overlays]
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
        elif qtype == "count" and ATTRIBUTE_COUNT.search(qtext):
            res = self.grounded_person_count(t, qtext)
        elif qtype == "yes_no" and TOUCH_VERBS.search(qtext):
            res = self.touch_burst(t, qtext)
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
                res = self.multiple_choice(options, qtext, t)
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
