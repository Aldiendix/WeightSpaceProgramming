# Close on its own: remove the keep-verifying and loop states, so the stop units no longer close by force

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md).

## The failure that is left
The parent is the champion's model (loop removal and self-distilled gains) with a few units in the last layer that
raise the end-of-thinking token more and more the longer the thinking runs. They end almost every long trace: on
practice problems only about 2% of the parent's answers still run into the 32k-token limit. But they end it by force,
and they do not choose the moment. Measure this first (step 1). In our exploration on practice problems:
- the share of answers that finish **without an answer** (thinking closed, no final answer) rose from 0.2% on the
  model without the units to 2.7-3.9% on the parent;
- the answers that finish are right far less often: 61% of the parent's finished answers on the practice dev split,
  against 69% on the model without the units (holdout: 68% against 72%).

When the units fire, the model is often still re-checking a result it had already stated, or repeating itself. The
thinking is closed in the middle of that, and the answer written after it is then wrong or missing. Two states are
behind it, and both are still in the parent's model:
- **Keep verifying.** Once the model has stated a result, it often re-checks it instead of closing the thinking.
- **The loop.** The model starts repeating its own earlier text.

If they are removed, the model closes the thinking itself, after it has stated its result, before the units have to
force it; the units still end what is left.

## Why it can be edited, and where
Both directions come from the toolkit's contrasts (`wsp/directions.py`), measured on the parent's own text:
- `conclude`: take a correct, finished answer and cut it right after the thinking first states the right result. The
  good branch closes the thinking (`</think>` and the model's own final answer); the bad branch keeps verifying (the
  model's own next text).
- `loop`: take a correct answer and cut it at a sentence boundary. The good branch is the model's own continuation;
  the bad branch repeats the model's earlier text, as a loop does.

In both, the two branches share the prefix, so the per-layer mean difference of the residual stream isolates the
state. Each direction is projected out of the attention outputs (`self_attn.o_proj`), the writers that add to the
residual stream at every token: conclude at attention 15 and 19, loop at attention 11 and 23. These are different
tensors from the parent's own edits (its stop units are in the last layer's MLP), so nothing the parent does is
undone. Measured on the champion's model, the two directions are nearly orthogonal (cosine -0.19 to +0.13 at the
attention layers). How strongly to remove each is found by a paired search on practice problems, with a holdout and
a random control.

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`.
- `prun.py` is `wsp.case run` split over all GPUs: the same seeds and files, N times faster.
- `more_answers.py` samples the parent on further practice problems outside the case's dev and holdout splits.
- `pairs_more.py` builds pairs for one contrast from the parent's answers and saves that direction under its own name.
- `grid.py` writes one `dir_ablate` candidate per (attention layer set, strength) from a measured direction
  (`--prefix` names the family) and ranks results; `stack.py` stacks candidates that touch different tensors.
- `onesource.py` makes a practice split of one source only.
- `kinds.py` shows where the answers go (right, finished wrong, finished without an answer, cut in a loop, cut).

On 4x RTX 4090, one model on dev128 x 4 takes about 15-25 minutes, and on holdout256 x 4 about 25-40 minutes. Start
long runs detached (`nohup` or `tmux`) and keep PROGRESS.md current.

1. **Case and the parent's answers** (practice only):
   `python -m wsp.case init co --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`
   Check that the parent chain is complete (`parent/edit/` holds the stop units and, under its own parent, the
   champion's edits). Then `python derive/prun.py co base dev --n 2` and `python derive/kinds.py co dev`. Check what
   the method is about: few answers cut, but answers finished without an answer, and finished answers right less often
   than you would expect. Read a few of the empty and the wrong finished answers: where did the thinking close?
2. **More of the parent's own answers**, to measure the directions well: on each GPU i of 4,
   `python derive/more_answers.py co base ANS --per-source 256 --n 4 --max-tokens 12288 --shard i/4` (about 25
   minutes). These problems are outside dev and holdout.
3. **Measure the two directions.**
   `python derive/pairs_more.py co --kind conclude --sources gsm8k math --sets dev --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out concl`
   (the conclude contrast needs a gold answer it can find in the thinking, so it uses these two sources), then
   `python derive/pairs_more.py co --kind loop --sources gsm8k math mmlupro ifsynth --sets dev --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out loop --compare concl_dirs.npz`.
   Go on only if both are stable at the attention layers (split-half stability well above 0.9) and the two are far
   from parallel (|cosine| well below 0.5).
4. **Starting runs to pair with:** `python derive/prun.py co base dev128 --n 4` and `... base holdout256 --n 4`.
5. **Search the strengths, alone and stacked.** With `W=$WSP_WORK/cases/co`:
   `python derive/grid.py build co --dirs $W/concl_dirs.npz --sets 15,19 --alphas 0.75 1 --prefix C` and
   `python derive/grid.py build co --dirs $W/loop_dirs.npz --sets 11,23 --alphas 1 1.5 --prefix L`.
   Stack the conclude candidate at strength 1 with each loop candidate:
   `python derive/stack.py co S_C1_L1 C15-19_a1 L11-23_a1` (and likewise for L11-23_a1.5).
   Run C15-19_a1 and the two stacks on `dev128 --n 4` and rank with `python derive/grid.py table co dev128`.
   Read `kinds.py` for each: the answers finished without an answer should fall and the finished answers should be
   right more often, while the cut share stays low.
   - Upload the best candidate early (step 7), and keep confirming while time allows.
6. **Confirm and control.** Run the two best candidates on `holdout256 --n 4` and pick the one with the higher
   holdout gain; when the two are within noise, prefer the one with fewer answers finished without an answer, as long
   as no practice source is clearly worse. Build the control: a random direction on each of the same tensors at the
   same strengths (`python derive/grid.py control co C15-19_a1 --dirs $W/concl_dirs.npz`, the same for the loop
   candidate, then `stack.py` the two `RAND_` edits); run it on dev128. It should be within noise of 0. If at least
   1.5 hours remain, run the base and the chosen candidate on a `math_train` split
   (`python derive/onesource.py co math1 --source math_train --n 192 --exclude-files "ANS.*.jsonl"`, then `prun.py`
   on `math1 --n 4`): the hard problems are where the units close most often. If no candidate beats the parent on
   both dev128 and holdout256, upload nothing worse than the parent.
7. **Upload.** Write REPORT.md: the measured failure (empty answers, accuracy of finished answers, per source), both
   directions' stability and their cosine, the search table, the holdout and the control. Package without uploading:
   `python -m wsp.upload co <edit> --report REPORT.md --dry-run`. Check it: `axonry-check
   $WSP_WORK/deliver/co__<edit>/edit --task task.json`. Then run the same upload command with `--task task.json`
   instead of `--dry-run`. Re-upload when a better candidate is confirmed.

## What our exploration found (practice splits only)
These are the same edits, run by us on the parent's model with the toolkit's sampling and grading. We measured the
directions on the model without the stop units (they sit in the last layer's MLP, so the residual stream up to that
layer is the same for the same text). They are support for what to expect, not a fixed answer: the search decides.
- **The parent** (dev128 x 4): 60.5% right, 2.0% cut, 3.9% finished without an answer, finished answers right
  61.2%. Holdout256 x 4: 66.5% right, 1.6% cut, 2.7% without an answer, finished answers right 67.6%.
- **Paired gain over the parent, dev128 x 4:** conclude alone (attention 15, 19, strength 1) +3.9 +- 2.4; conclude
  with loop 1 +6.4 +- 2.2 (67.0% right: `gsm8k_train` +8.6, `math_train` +7.8, `mmlupro` +4.0, `ifsynth` +5.5;
  answers without an answer 20 -> 6, finished answers right 61.2% -> 69.3%); conclude with loop 1.5 +2.3 +- 2.0.
- **Holdout256 x 4:** conclude alone +1.0 +- 1.4 (finished answers right 67.6% -> 71.5%, without an answer 28 -> 10);
  conclude with loop 1.5 +0.4 +- 1.4.
- **Two further splits** (256 practice problems x 4 each): conclude with loop 1.5 +0.1 +- 1.3 and +4.5 +- 1.2.
- On the model without the stop units the same stacks gain +3.3 (loop 1) and +3.5 (loop 1.5) over six practice
  splits, with random directions on the same tensors at +1.0 (dev128) and -0.6 (holdout256): the gain is the
  directions.

## What does not work (measured on practice; do not spend the run on it)
- **More dose of the champion's self-distilled gains** or **self-distillation again**: -1.6 to +0.2.
- **Too much removal:** loop strength 2, conclude strength 1.25 and above.
- **Source-specific directions** (conclude from `gsm8k_train` only, loop from `math_train` or `mmlupro` only): no
  better than the mixed ones (cosine 0.99).
- **A doubt direction** (second-guessing) added to conclude, and a **right-versus-wrong** direction (contrast
  `correct`): not stable or not robust on further splits.
- **Short-finish self-distillation into rewritten MLP units** (the model's shortest right answers): instruction
  following falls by 8-14 points.
- **The stop units alone are not the problem to remove:** without them far more answers are cut (13% instead of 2%).
  Keep them; this method only removes what makes them fire at the wrong moment.

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: both directions, the writers and the strengths are measured and chosen in the run.
