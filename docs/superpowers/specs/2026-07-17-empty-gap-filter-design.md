# Empty-Gap Filter — suppress irrelevant empty shackles between a LOAD placard and the first bird

**Date:** 2026-07-17 · **Status:** approved by user (design + scope)

## Problem

Every delivery starts with a fluorescent LOAD placard hung on an empty shackle,
followed by a run of empty shackles before the first bird of that delivery
reaches the line. Those gap empties are irrelevant to plant reporting, but today
they inflate the cumulative `empty_shackles` count on every endpoint and the UI.

## Rule (user-specified)

From the moment a `load` marker crosses the counting line until the first
`slaughtered_chicken` crosses it, empty-shackle crossings are **not counted**.
Once that first chicken is counted, empty shackles count normally again.

- Applies after **every** load placard, not just the first.
- Before the first-ever load (stream start / after a counts reset), empties
  count normally — the rule is anchored on a load crossing.
- Two placards with no birds between them: the gap stays open across the second
  placard until a chicken finally crosses.

## Approach (approved: option A)

A small derived state machine, `EmptyGapFilter`, in `app/core/shipment.py`,
following the exact pattern of `ShipmentTracker`: it **reads** the cumulative
per-class IN counts the Ultralytics ObjectCounter already produces and never
mutates the counter. Alternatives rejected: folding into `ShipmentTracker.update`
(mixes responsibilities in proven code) and patching ObjectCounter internals
(invasive to code we just stabilized).

### State machine

Fed once per processed frame with `(load_cum, empty_cum, chicken_cum)`:

1. Attribute this frame's empty delta by the state **before** this frame's
   transitions: if a gap is open, add the delta to `discarded`.
2. If `chicken_cum` rose → close the gap.
3. If `load_cum` rose → open a gap.
4. Return the adjusted cumulative empty count: `max(0, empty_cum - discarded)`.

Prior-state attribution matches physical belt order (one-way flow): an empty
crossing in the same frame as a placard was downstream of it (tail of the
previous span) and counts; an empty crossing in the same frame as the first
chicken was still in the gap and is dropped. Chicken-before-load transition
order makes a same-frame chicken+placard open a fresh gap for the new delivery.
Cumulative values falling (counter rebuilt) resync silently, as in
`ShipmentTracker`; `reset()` zeroes everything and is wired next to
`_shipment.reset()` in `VideoProcessor.reset_counts()`.

### Integration — one choke point covers everything

`VideoProcessor._run()` computes `self._counts` from `classwise_count` each
frame (`app/core/video_processor.py:356`). The filter is fed right there and
`_counts["empty_shackles"]` is replaced with the adjusted value. Every consumer
reads that dict: legacy `/api/stream/*`, multi-stream `/api/streams/{id}/*`,
`/api/video/*`, and the UI panels that poll them. `ShipmentTracker` only reads
load/slaughtered counts — unaffected. The still-image endpoint counts
detections in one frame (nothing "passes" a line) and is out of scope.

**Nothing else changes:** no new API fields, no UI changes, chicken/load counts
and shipment records untouched.

## Testing

Pure-logic unit tests alongside `tests/test_shipment.py` (no torch/cv2): normal
counting before first load; suppression after load; release on first chicken;
same-frame ties on both boundaries; multi-load gaps; birdless shipments; reset;
counter-rebuild resync; adjusted count never negative.
