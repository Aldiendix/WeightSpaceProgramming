---
name: weight-edit
description: Improve a small LLM on a benchmark by directly editing its weights, following the Weight-Space Programming protocol - diagnose the failure, build contrast pairs, find the direction and the components writing it, edit, test selectivity and dose with controls, freeze, confirm on holdout, then evaluate on benchmarks. Use when asked to modify/repair model weights, run a weight-editing case, or continue a case under cases/.
---

# Weight editing workflow

You are the editor: you decide what to change from measurements of what parts of the network compute. Tools and
layout: `README.md`. Read `playbook.md` first; add lessons to it at the end. Every command below is
`python -m wsp.case ...`, run on the GPU box through `./gpu queue <name> '<cmd>'` (or `./gpu run` for short ones);
`./gpu pull` brings results back. Keep decisions in `cases/<case>/NOTES.md`; numbers are generated
(`report` -> `RESULTS.md`, `REPORT.md` headline), never copied by hand.

## Rules
- Edits come from reasoning on measured activations (means, differences, projections).
- Edits stay within `rules.json` (editable tensors, whitelisted ops, payload and magnitude budgets of the case's
  track): `edit` self-checks, `python -m wsp.rules check <spec> --build` checks the built magnitudes.
- Benchmarks are never used to choose anything. Choose on practice problems: `dev*` sets for selection, `holdout*`
  only after freezing.
- Model-card sampling only (`profiles.py`); a gain that exists only under a bad decoding setting is not a result.
- `freeze` the chosen edit before any holdout or benchmark run; report everything, including losses and failures.

## Steps
1. **Screen**: `screen <profiles...>` on `cases/screening/sets/screen64.json`, then `report`. Continue only if a
   removable failure dominates and `acc when finished` is far above `acc`; read failing answers yourself.
2. **Init**: `init <case> --profile P --include cases/screening/sets/screen64.json --subset dev128=dev:128
   --subset holdout256=holdout:256`, then `run <case> base dev`.
3. **Pairs**: `pairs <case> --sets dev` (good traces of the base run) and a held-out file with
   `--exclude-pairs pairs --out pairs_heldout`. The contrast kind is `case.json` -> `contrast` (`directions.CONTRASTS`;
   add a generator there for a new failure type).
4. **Extract**: `extract <case>` -> per-layer stability, separation and writer totals. Pick layers where the
   direction is stable and MLPs write much of it; avoid the last layer.
5. **Edit**: `edit <case> --layers ... --topk 64 --alpha ...`, plus the same with `--random-control`.
   `selectivity <case> <edit>` first (failure log-prob down, normal log-prob ~unchanged), then
   `run <case> <edit> dev128` for the promising ones and a dose series.
6. **Freeze** the best edit (`freeze <case> <edit>`), then `run <case> base|<edit> holdout256` and
   `run <case> base|<edit> bench:<name>` (benchmarks: `data.BENCHMARKS`).
7. `report`, update NOTES (decisions) and `playbook.md` (lessons).

Paired comparisons only: base and edit runs share problems and per-problem seeds; `run` pairs them automatically.
