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
- No overfitting (`rules.json` -> `edit.no_overfitting`): never use the evaluation benchmarks' items or answers to
  derive or choose an edit; the gain must carry over to unseen problems. Edits come from
  reasoning on measured activations and weights (means, differences, projections).
- The deliverable is an edit spec within `rules.json` -> `edit` (editable tensors, ops, the payload and
  magnitude limits; `edit` prints the spec's tensors and payload, the build manifest its relative changes). The
  model is always rebuilt from the pinned base, so only the spec counts.
- Choose on practice problems only: `dev*` sets for selection, `holdout*` only after `freeze`. In a competition task
  everything about the evaluation is closed to you (which sets, their items, answers and results): never load, read
  or look it up, and aim at no particular set; outside one, `bench:<name>` sets only confirm a frozen edit.
- Generation uses the profile's model-card sampling; a gain that needs another decoding setting is not a gain.
- Report everything, including losses and failed attempts.

## Time budget (competition tasks have a hard deadline)
- Plan before starting: the deadline is in the brief. Generation dominates the cost: every `run` starts vLLM (a few
  minutes), and its length grows with the number of problems and how often the model runs to the token limit;
  probes (`extract`, `selectivity`) take minutes. Time one run, then count the runs you can afford and drop what does
  not fit.
- Keep every GPU busy: a `run` splits its problems over all visible GPUs; to run independent jobs side by side, give
  each its own `CUDA_VISIBLE_DEVICES`.
- Use the cheapest evidence first: `selectivity` screens edits in minutes; generate only for the most promising
  ones, on `dev128`.
- **Upload early, then improve**: as soon as one edit passes selectivity and gains on DEV, upload it
  (`python -m wsp.upload`). Re-upload when a better edit is confirmed; the last complete upload counts, so an early
  one is never wasted and a late one may never land.
- Keep the last stretch (at least 30 min) for the final upload; skip HOLDOUT if it does not fit and say so in the
  report.

## Steps
1. **Screen**: in a competition the base (and parent, if any) is given: run it on the screening problems
   (`init`, then `run <case> base dev`, and read `RESULTS.md`). Otherwise compare candidate models with
   `screen <profiles...>` and `report`. A removable failure must dominate and `acc when finished` be far above
   `acc`; read failing answers yourself. Without that, editing is unlikely to help: say so instead of forcing an
   edit.
2. **Init**: `init <case> --profile P --include cases/screening/sets/screen64.json --subset dev128=dev:128
   --subset holdout256=holdout:256` (the `--include` path may be relative to the repository), then
   `run <case> base dev`. The splits default to math practice: add `--practice` sources (`wsp/data.py`) so that every
   practice task type is covered, since an edit that helps one type can hurt another. When there is a
   current champion (or any edit to improve on), add `--parent <its edit.json>`: "base" is then that model, and your
   edits stack on it. First read the `REPORT.md` next to that edit.json: what the parent tried and what did not
   work.
3. **Pairs**: `pairs <case> --sets dev`, and held-out pairs with `--exclude-pairs pairs --out pairs_heldout`; build
   them from a base run with enough finished traces, since held-out pairs need traces of their own. The contrast
   kind is `case.json` -> `contrast` (`wsp/directions.py` `CONTRASTS`; add a generator for a new failure).
4. **Extract**: `extract <case>` -> per-layer stability, separation, writer totals. Take layers where the direction
   is stable and MLPs write much of it; avoid the last layer.
5. **Edit**: `edit <case> --layers ... --topk 64 --alpha ...` and the same with `--random-control`; then
   `selectivity <case> <edit>` (failure log-prob down, normal log-prob ~unchanged) and `run <case> <edit> dev128` for
   the promising ones, with a dose series. Every `run` of an edit is paired with the base automatically; choose
   between edits with `compare <case> <a> <b> <set>` on the same problems, and trust a gap only when it is about
   twice its SE; when it is not, add DEV problems or samples (`--n`) rather than choosing on HOLDOUT.
6. **Freeze** the best edit (`freeze <case> <edit>`), then `run <case> base holdout256` and `run <case> <edit>
   holdout256`.
   `edit` writes `dir_ablate` specs; any other allowed op is written by hand (or by `wsp.gate.ops` for one MLP
   unit) into `cases/<case>/edits/<name>.json`. Lists may instead be `"file.npz#key"` relative to the spec;
   `tensor` is a full name from the base's safetensors index (`wsp/edits.py` has the exact semantics):
   ```
   {"name": "E1", "base": "<profile>", "parent": "../parent/<parent>.json" (when stacking), "ops": [
    {"op": "dir_ablate", "tensor": "...layers.15.mlp.down_proj.weight", "side": "out", "u": "E1.npz#u15",
     "alpha": 6.0, "cols": [3, 17]},                                   # cols optional: all columns
    {"op": "scale_rows", "tensor": "...layers.4.mlp.up_proj.weight", "rows": [11], "factor": 0.5},
    {"op": "scale_cols", "tensor": "...layers.4.mlp.down_proj.weight", "cols": [11], "factor": 0.5},
    {"op": "set_rows", "tensor": "...layers.4.mlp.gate_proj.weight", "rows": [11], "values": "E1.npz#gate"},
    {"op": "set_cols", "tensor": "...layers.4.mlp.down_proj.weight", "cols": [11], "values": "E1.npz#down"}]}
   ```
   `values` is len(rows) x in for `set_rows`, out x len(cols) for `set_cols`. `low_rank` (`edit --low-rank`) is
   for research only: rules.json does not allow it.
7. **Deliver**: `report`, update NOTES (and `playbook.md` outside competition tasks), then the final
   `python -m wsp.upload <case> <edit> [--task task.json]` (see "Time budget").
