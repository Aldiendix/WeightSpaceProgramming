---
name: weight-edit
description: Improve a small LLM on benchmarks by directly editing its weights, following the Weight-Space Programming workflow - diagnose the failure, build contrast pairs, find the direction and the components writing it, edit, test selectivity and dose with controls, freeze, confirm on holdout, then deliver the edit. Use when asked to modify/repair model weights, run a weight-editing case or competition task, or continue a case under cases/.
---

# Weight editing workflow

You are the editor: you decide what to change from measurements of what parts of the network compute. Read
`playbook.md` first (what worked, what did not) and, in a competition task, the task's brief and the submission's
`METHOD.md`. Tools: `python -m wsp.case <command>` (`-h` for options), run wherever the GPU is. Keep decisions in
`cases/<case>/NOTES.md`; numbers come from `report` (`RESULTS.md`), never copied by hand.

## Ground rules
- Edits come from reasoning on measured activations and weights (means, differences, projections).
- The deliverable is an edit spec within `rules.json` -> `edit` (editable tensors, ops, the track's payload and
  magnitude limits; `edit` prints the spec's tensors and payload, the build manifest its relative changes). The
  model is always rebuilt from the pinned base, so only the spec counts.
- Choose on practice problems only: `dev*` sets for selection, `holdout*` only after `freeze`. Benchmarks are for the
  final confirmation when they are available; never tune on them.
- Generation uses the profile's model-card sampling; a gain that needs another decoding setting is not a gain.
- Report everything, including losses and failed attempts.

## Steps
1. **Screen**: `screen <profiles...>` (problem list in `cases/screening/sets/`), then `report`. A removable failure
   must dominate and `acc when finished` be far above `acc`; read failing answers yourself. Without that, editing
   is unlikely to help: say so instead of forcing an edit.
2. **Init**: `init <case> --profile P --include cases/screening/sets/screen64.json --subset dev128=dev:128
   --subset holdout256=holdout:256`, then `run <case> base dev`. When there is a current champion (or any edit to
   improve on), add `--parent <its edit.json>`: "base" is then that model, and your edits stack on it.
3. **Pairs**: `pairs <case> --sets dev`, and held-out pairs with `--exclude-pairs pairs --out pairs_heldout`. The
   contrast kind is `case.json` -> `contrast` (`wsp/directions.py` `CONTRASTS`; add a generator for a new failure).
4. **Extract**: `extract <case>` -> per-layer stability, separation, writer totals. Take layers where the direction
   is stable and MLPs write much of it; avoid the last layer.
5. **Edit**: `edit <case> --layers ... --topk 64 --alpha ...` and the same with `--random-control`; then
   `selectivity <case> <edit>` (failure log-prob down, normal log-prob ~unchanged) and `run <case> <edit> dev128` for
   the promising ones, with a dose series. Every `run` of an edit is paired with the base automatically.
6. **Freeze** the best edit (`freeze <case> <edit>`), then `run <case> base holdout256` and `run <case> <edit>
   holdout256` (and `bench:<name>` sets if available).
7. **Deliver**: `report`, update NOTES and `playbook.md`, then `python -m wsp.upload <case> <edit> [--derive DIR]
   [--task task.json]` before the deadline (several uploads are fine; the last complete one counts).
