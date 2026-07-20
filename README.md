# builderr-kitchen

Agent for the builderr **Kitchen CCTV monitor** challenge (Round 1, $300, deadline Sep 9 2026).
Full rules and formats: [SPEC.md](SPEC.md).

## Run

```
pip install -r requirements.txt
python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

First run downloads three small local models (YOLO11n ~5 MB, Qwen3-VL-2B ~4.4 GB,
EasyOCR ~150 MB). **Everything runs locally — no API keys, `estimated_model_api_cost_usd`
is genuinely $0.00.** GPU (≥6 GB VRAM) recommended; CPU works with more wall-clock.

## Architecture (coarse-to-fine, all local)

1. **Ingest** — probe duration/fps; compute the scaled frame / cost / wall-clock caps.
2. **Coarse index** (`pipeline/coarse.py`) — ~1 frame/4 s: motion differencing +
   batched YOLO person detection. 60 min of video indexes in ~16 s.
3. **Question routing** (`pipeline/engine.py`) — per-type strategies:
   - *count / yes-no at a timestamp*: targeted frames + Qwen3-VL query; counts
     cross-checked against the YOLO index.
   - *timestamp*: activity-biased probe scan → **confirm-on-yes** (a false yes wrecks
     bisection, a false no only delays it) → bisection to inside the ±2 s tolerance.
   - *duration*: state-start and state-end located the same way.
   - *event order / multiple choice*: shared frame spread, one VLM call per frame
     covering all candidates; non-temporal MC answered by direct option vote.
   - *OCR* (order numbers, labels, screens): EasyOCR on routed frames only.
   - **Region zoom**: when a question names a location ("left end of the counter"),
     an upscaled crop rides along with the full frame — small-object accuracy at
     CCTV resolution depends on this.
4. **Honest abstention** — low-confidence results answer `not_visible` (a scored
   category) instead of guessing.
5. **Budget ledger** (`answer.py`) — frames/calls/cost tracked against the scaled
   caps; coarse pass capped at 55% of the frame budget; past 85% of the wall-clock
   cap the engine degrades to index-only answers rather than busting the cap.

## Measured (RTX 5050 laptop, 8 GB VRAM)

| Metric | 19.5-min, 10 static q | 19.5-min, 7 handoff q | 22-min portrait, 9 q | hygiene probes, 5 q | 60-min, 6 q |
|---|---|---|---|---|---|
| Frames / cap | 362 / 488 | ~360 / 488 | 391 / 555 | ~90 / 488 | 884 / 1500 |
| Wall-clock / cap | 55 s / 488 s | ~41 s / 488 s | 45 s / 555 s | ~35 s / 488 s | 39 s / 1500 s |
| API cost | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 |
| Accuracy | **10/10** | **5.5/7** | **8/9** | **5/5** | unlabeled (budget test) |

**Combined labeled accuracy: 28.5/31 (92%)** across three very different videos:
a wide-angle Indonesian wok kitchen (360p landscape), its handoff events (courier
arrival/duration, takeaway packing, event ordering), a top-down portrait
home-kitchen close-up (no visible people, small objects), and hygiene-domain
probes (attribute-filtered headwear counts, touch/no-touch contact, unreadable
text). All ground truth hand-labeled by frame scrubbing; timestamp answers land
0.0-1.6 s from truth.

VLM steady-state latency: ~0.21 s/query on GPU (Qwen3-VL-2B bf16); ~4 s/query
CPU-only (fp32) — a 60-min/6-question run extrapolates to ~5 min on pure CPU,
still well inside the 25-min cap.

## Layout

- `answer.py` — contract entry point + budget enforcement
- `pipeline/` — `ingest.py`, `coarse.py` (index), `engine.py` (strategies),
  `vlm.py` (Qwen3-VL wrapper), `ocr.py` (EasyOCR wrapper)
- `scripts/score.py` — local harness mirroring the challenge rubric
- `scripts/probe_gold_feedback.py` — targeted regression probes for each bug
  the organizer's gold-set runs surfaced (see below); reproduces the exact
  scenario against real footage instead of requiring a full pipeline run
- `questions.json` / `truth.json` — hand-labeled dev set (19-min public CCTV clip)
- `videos/` — local test footage (gitignored)

## Probe-scan hardening (learned on the dev sets)

- **State phrasing**: "did X first enter" is rewritten to "is X present" —
  a single frame can answer states, not narratives.
- **Person events route through the YOLO index**: a courier entering bumps the
  person count above the modal value; those windows are probed first.
- **Confirm-on-yes**: a false yes wrecks bisection; every scan hit is verified
  on a neighboring frame.
- **Stability check**: after bisection, the state must still hold at +6 s/+12 s —
  rejects look-alikes carried through the frame (a yellow plate is not a yellow barrel).
- **Strict-object suffix** (objects only, not people): tells the model to reject
  similar-looking objects of the wrong type/size — this alone took the barrel
  timestamp from 341 s error to 0.0 s.
- **Container decomposition**: yes/no probes accept food-in-a-pan as "a bowl of
  food"; asking "what container is it?" open-endedly does not. Scan candidates
  whose description names a container are verified this way.
- **Never quote answer literals**: instructing `Answer 'yes' or 'no'` makes the
  2B model parrot the first quoted token; unquoted instructions restore reasoning.
- **Temporal-relation questions** ("did X happen before Y?") localize both
  events and compare times instead of hoping a multi-frame prompt reasons it out.
- **Text reading is dual-source**: EasyOCR (with rotation) + VLM-on-upscale,
  answer accepted only when corroborated; otherwise `not_visible` — the rubric
  scores honest abstention, not lucky guesses.
- **Overlay rejection**: any string OCR also reads at a control frame far from
  the moment of interest is burned-in overlay (timestamps, channel watermarks),
  never scene content — critical for order-number questions on watermarked CCTV.
- **Attribute counts are per-person**: "how many staff wear hair covers" crops
  each YOLO person box and asks about each crop individually; whole-frame
  counting conflates "people" with "people matching the attribute".
- **Touch/contact runs on a dense burst** (9 frames across ±3 s) cropped to a
  *verified* VLM grounding of the named object unioned with the nearest person
  box — object-only crops cut out the person, person-only crops cut out the
  object, and "answer yes if it happens in ANY frame" phrasing makes the model
  hallucinate contact with objects that aren't in view.

## Second organizer gold-set pass: three bugs, three lessons

A re-run against the organizer's gold set held steady at 3/6 (with wall-clock
dropping 9.2 min → 3.0 min from the round above) and named three specific
failures. Each is now covered by `scripts/probe_gold_feedback.py`:

- **Multi-image batching is unreliable on a 2B model.** Attribute counting
  queried all detected people in one call ("reply one line per image") — the
  model answered "yes" to every line regardless of what was actually in each
  crop. Querying each crop in its own call fixed it immediately; the extra
  model calls are still ~free locally.
- **A snap first-token "yes" can contradict the model's own reasoning.**
  Asked "is this a black shirt?", the model opened with "Yes" and then wrote
  "...the person is wearing a red shirt, so the statement is false" — the
  description was right, the leading token was wrong. Two independent fixes
  were needed, not one: (a) prompt for describe-then-verdict with an explicit
  `Answer: yes/no` marker parsed instead of the first token, and (b) for
  color+garment predicates specifically, skip the yes/no verdict entirely and
  ask "what color is their {garment}?" directly — the model names each
  garment's color correctly even when it can't correctly bind color to
  garment inside a yes/no judgment.
- **A model's self-report of "illegible" vs "no text" isn't trustworthy
  evidence.** The same unreadable brand logo got called `'none'` as often as
  `'illegible'` — both mean "I couldn't read it," but only one matched the
  abstention regex. Root cause fixed by not asking the model to categorize
  its own uncertainty at all: a weak-confidence OCR pass (threshold 0.05,
  overlay-filtered) is the "something is here but unreadable" signal now,
  and a single uncorroborated VLM guess no longer counts as a confirmed
  reading — it must recur across a small frame burst or be corroborated by
  OCR before the engine answers "yes" to a readability question.
- **Overlay filtering had a gap**: `best` (the top OCR hit) was filtered
  against the channel watermark for *word-answer* questions but not for
  *yes/no visibility* questions — a watermark that clears the OCR confidence
  threshold could make "is any text visible" answer "yes" on every frame of
  a watermarked video. Filtering now happens once, before any downstream use
  of `best`.

## Known limits

- Short transient events (~20 s) involving small ambiguous objects can be missed
  by the sparse scan; the engine answers `not_visible` (honest abstention)
  rather than guessing.
- "Enter" timestamps read ~2 s late when the person is half-occluded in a doorway.
