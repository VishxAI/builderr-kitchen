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

| Metric | 19.5-min, 10 static q | 19.5-min, 7 handoff q | 22-min portrait, 9 q | 60-min, 6 q |
|---|---|---|---|---|
| Frames / cap | 361 / 488 | ~360 / 488 | 391 / 555 | 884 / 1500 |
| Wall-clock / cap | 52 s / 488 s | ~41 s / 488 s | 45 s / 555 s | 39 s / 1500 s |
| API cost | $0.00 | $0.00 | $0.00 | $0.00 |
| Accuracy | **10/10** | **5.5/7** | **8/9** | unlabeled (budget test) |

**Combined labeled accuracy: 23.5/26 (90%)** across three very different videos:
a wide-angle Indonesian wok kitchen (360p landscape), its handoff events (courier
arrival/duration, takeaway packing, event ordering), and a top-down portrait
home-kitchen close-up (no visible people, small objects). All ground truth
hand-labeled by frame scrubbing; timestamp answers land 0.0-1.6 s from truth.

VLM steady-state latency: ~0.21 s/query on GPU (Qwen3-VL-2B bf16); ~4 s/query
CPU-only (fp32) — a 60-min/6-question run extrapolates to ~5 min on pure CPU,
still well inside the 25-min cap.

## Layout

- `answer.py` — contract entry point + budget enforcement
- `pipeline/` — `ingest.py`, `coarse.py` (index), `engine.py` (strategies),
  `vlm.py` (Qwen3-VL wrapper), `ocr.py` (EasyOCR wrapper)
- `scripts/score.py` — local harness mirroring the challenge rubric
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

## Known limits

- Short transient events (~20 s) involving small ambiguous objects can be missed
  by the sparse scan; the engine answers `not_visible` (honest abstention)
  rather than guessing.
- "Enter" timestamps read ~2 s late when the person is half-occluded in a doorway.
