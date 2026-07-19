# Kitchen CCTV Monitor — Challenge Spec (builderr, Round 1)

Source: https://builderr.ai/kitchen-video (+ /kitchen-video-challenge-draft.md, /guidelines#video-rules, /start?challenge=video)
Captured: 2026-07-19

## Basics

- **Bounty:** $300 (Round 1)
- **Window:** Jul 12 – Sep 9, 2026
- **Submit:** email `submit@builderr.ai` with repo/endpoint URL, agent name, models/APIs used, expected cost per scored run, notes.

## Task

Build an agent that answers operational questions from fixed-camera kitchen CCTV footage:
headwear/hygiene checks, how long food sat at handoff points, container sealing, where
delays originated, event ordering, order-number visibility.

**Evaluation is fully automated.** Evaluators run:

```
python answer.py --videos ./videos --questions questions.json --out answers.json --log run_log.json
```

No manual video inspection. Hidden question set.

## I/O formats

`questions.json` (input):

```json
[
  {
    "id": "q001",
    "video_id": "sample_01",
    "type": "yes_no",
    "question": "Did the worker close the container before moving it away from the station?"
  }
]
```

`answers.json` (output):

```json
[
  {
    "id": "q001",
    "answer": "yes",
    "confidence": 0.72,
    "evidence": [
      { "video_id": "sample_01", "timestamp_start": 91.2, "timestamp_end": 94.8 }
    ]
  }
]
```

`run_log.json` (output):

```json
{
  "runtime_seconds": 812,
  "frames_processed": 642,
  "model_calls": 38,
  "estimated_model_api_cost_usd": 0.18,
  "normalized_model_api_cost_per_60min_usd": 0.18
}
```

## Question types

`yes_no`, multiple choice, `count`, `timestamp`, `duration`, short structured object,
and `not_visible` (answering not_visible when evidence is insufficient is a *scored*
behavior — do not guess).

## Scoring (100 pts)

| Component | Points | Notes |
|---|---|---|
| Answer accuracy | 80 | hidden questions: yes/no, MC, counts, states, events, timestamps, durations, not-visible |
| Cost efficiency | 15 | lower cost earns credit **only when accuracy holds**; must be inside all hard caps |
| Reproducibility | 5 | one-command run, logged sampling/model calls, deterministic-ish outputs |

**Timestamp tolerance:** full credit within **±2 s**, partial within **±5 s**, zero beyond
(unless the rubric marks an event as spanning longer).

## Hard caps (exceeding any = run does not rank)

Scaled proportionally to source video length:

- **$0.30** estimated model/API cost per 60 min of video
- **25 min** wall-clock for the full eval
- **~1,500 sampled frames** per 60 min (≈ 1 frame / 4 s — note: a naive 1 fps full pass
  is 3,600 frames/hr and busts this cap)

## Model policy

Open-source/local models recommended (cost → $0), cloud models allowed if every call is
logged and normalized cost stays under the cap.

## Organizer guidance

- Coarse-to-fine: cheap full-video pass, then targeted inspection of likely windows.
- Attach timestamp evidence to every answer.
- OCR selectively (labels, receipts, screens) — not every frame.
- This is not a captioning contest and not a spend race.

## Reference footage

Public samples, 19–60 min, fixed CCTV angles:

- https://www.youtube.com/watch?v=eAXIkGQYYWQ (60 min fixed kitchen CCTV)
- https://www.youtube.com/watch?v=ZrlbDd4Bs2Q (28 min)
- https://www.youtube.com/watch?v=2Bj50mmQeFM (22 min)
- https://www.youtube.com/watch?v=hwz0t3kckpo (19 min)
- https://www.youtube.com/watch?v=mWOoAf4rIhk (Chinese restaurant hygiene)
- https://www.youtube.com/watch?v=nQht56i2Xkg (IP bullet camera kitchen view)

Datasets: [Chinese Commercial Kitchen (HF)](https://huggingface.co/datasets/nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview),
[Kaggle restaurant kitchen video](https://www.kaggle.com/datasets/naoamscoltd/kitchen-video-in-restaurants),
[COM Kitchens](https://www.nii.ac.jp/dsc/idr/en/rdata/COM_Kitchens/),
[EPFL Smart Kitchen](https://github.com/amathislab/EPFL-Smart-Kitchen)
