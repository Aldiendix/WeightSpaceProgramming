# Steps that decide: close when done, stop repeating, and credit the thinking steps that make a problem solvable

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md). The parent already improved how its
answers are written; this method works on the thinking that comes before them.

## The failure that is left
Two things still cost the parent's line on practice problems:
- **Long math thinking runs to the token limit.** On `math_train` practice problems about 17% of the answers are cut
  at the 32k limit. Many of these traces re-check a result they have already stated, or repeat their own earlier
  text, and close too late or not at all.
- **On word problems, the answer is often decided by a few steps of the thinking.** On mixed `gsm8k_train` problems
  (some samples right, some wrong) the traces differ in a handful of steps: one sample sets up the quantities
  correctly and the rest follows, another makes the wrong choice early and never recovers.

The method has three parts. They touch different tensors (or the same ones in ways that commute) and stack.

## Part A: close when done (conclude direction, attention 15 and 19)
The toolkit's `conclude` contrast (`wsp/directions.py`) on the parent's own text: take a right, finished answer and cut
it right after the thinking first states the right result. The good branch closes the thinking (`</think>` and the
model's own final answer); the bad branch keeps verifying (the model's own next text). The two branches share the
prefix, so the per-layer mean difference of the residual stream isolates the keep-verifying state. Project it out of
the attention outputs (`self_attn.o_proj`) of layers 15 and 19 (`dir_ablate`); the strength is chosen by the search
(0.75 or 1).

## Part B: stop repeating (loop direction, attention 11 and 23)
The toolkit's `loop` contrast on the same answers: cut a right answer at a sentence boundary; the good branch is the
model's own continuation, the bad branch repeats the model's earlier text, as a loop does. Project that direction out
of the attention outputs of layers 11 and 23 at strength 1.

## Part C: credit the steps whose finishing value jumps
For each of the parent's own traces on a mixed practice problem, measure the **value of the thinking so far** at
checkpoints (every 512 tokens of the first 4096): let the model continue the thinking from that point 4 times, with
the normal sampling settings and a budget of 8192 tokens, and grade the answers. v(T) = share right. The budget makes
the value mean "solvable from here, in time".

Between two checkpoints the value either stays, rises or falls. The segments where it changes by at least 1/2 are the
decisive steps: a rise means that stretch of thinking made the problem solvable, a fall means it lost it. Each such
segment becomes a training row: the prompt is the question plus the thinking up to the segment (no loss there), the
target is the segment's own text, and the weight is the value change (positive or negative). Everything else gets no
credit. This is a policy-gradient step with a measured advantage on each step of the thinking, instead of one reward
for the whole answer.

The parameters are per-unit gains of the MLPs (`scale_rows` on `up_proj`) and per-head gains of the attention
outputs (`scale_cols` on `o_proj`), layers 4 to 21. The fit learns log-gains; the edit applies a dose of them (0.5:
half the learned log-gains). The values come from full continuations, not from forcing an answer: forced answers at a
cut of the thinking are a poor estimate of where the thinking will end up (see "What does not work").

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`.
- `prun.py` is `wsp.case run` split over all GPUs: the same seeds and files, N times faster.
- `sample.py` samples the parent on practice problems outside every split of the case.
- `pairs_more.py` builds pairs for one contrast from the parent's answers and saves that direction; `grid.py` writes
  `dir_ablate` candidates from it and ranks results.
- `pick_traces.py`, `step_values.py`, `step_rows.py`: Part C's traces, values and rows. `gains_fit.py` fits and exports
  the gains; `concat.py` joins the parts into one edit.
- `kinds.py` shows where the answers go (right, finished wrong, finished without an answer, cut).

Time plan on 4x RTX 4090 (6 hours): sampling 40 min, step values 70-90 min, the rest is screening. Start long runs
detached (`nohup` or `tmux`), keep PROGRESS.md current, and upload a candidate as soon as one beats the parent.

1. **Case and the parent's starting run** (practice only):
   `python -m wsp.case init co --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`.
   Check that the parent chain is complete (`parent/edit/` and its own parents). Then
   `python derive/prun.py co base dev128 --n 4` and `python derive/kinds.py co dev128`: note the cut share per source.
2. **The parent's own answers** on fresh practice problems, one run per GPU i of 4:
   `python derive/sample.py co base ANS --source gsm8k_train:768 --source math_train:384 --n 4 --max-tokens 16384 --shard i/4`
   (about 30-45 minutes). These problems are outside dev and holdout.
3. **Parts A and B.** With `W=$WSP_WORK/cases/co`:
   `python derive/pairs_more.py co --kind conclude --sources gsm8k math --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out concl`, then
   `python derive/pairs_more.py co --kind loop --sources gsm8k math --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out loop --compare concl_dirs.npz`.
   Go on only if both are stable at their layers (split-half stability well above 0.9) and far from parallel
   (|cosine| well below 0.5). Then
   `python derive/grid.py build co --dirs $W/concl_dirs.npz --sets 15,19 --alphas 0.75 1 --prefix C` and
   `python derive/grid.py build co --dirs $W/loop_dirs.npz --sets 11,23 --alphas 1 --prefix L`.
4. **Part C, values.** `python derive/pick_traces.py co TR.jsonl ANS.*.jsonl --sources gsm8k math --problems 300 --per 2`
   (one right and one wrong trace per mixed problem; expect 200-300 mixed problems). Then on each GPU i of 4:
   `python derive/step_values.py co base TR.jsonl --out VAL --step 512 --maxt 4096 --k 4 --budget 8192 --shard i/4`
   (about 70-90 minutes). Meanwhile write REPORT.md's first part.
5. **Part C, rows and fit.** `python derive/step_rows.py co ROWS.jsonl VAL.*.jsonl --traces TR.jsonl --jump 0.5`
   (expect 400-1000 rows, both signs). Fit on one GPU (a few minutes), with `MP` = the parent model directory
   (`python -c "from wsp import case; print(case.model_path('co', 'base'))"`) and S = 2 x rows / 8:
   `python derive/gains_fit.py train $MP ROWS.jsonl FIT --heldout ROWS.held.jsonl --rank 0 --unit-gains --gains --gain-layers 4 21 --lr 2e-3 --accum 8 --steps S --warmup 5 --max-tokens 512 --eval-every 25`,
   then `python derive/gains_fit.py export FIT co STEPS_05 --model $MP --scale 0.5`.
   Join: `python derive/concat.py co CLS_075 C15-19_a0.75 L11-23_a1 STEPS_05` (the main candidate: all three parts),
   `python derive/concat.py co CS_075 C15-19_a0.75 STEPS_05` (without Part B) and
   `python derive/concat.py co CLS_1 C15-19_a1 L11-23_a1 STEPS_05`.
6. **Screen.** `python derive/prun.py co <edit> dev128 --n 4` for `CLS_075` first, then `CS_075` and `CLS_1`; rank
   with `python derive/grid.py table co dev128`. Upload `CLS_075` (step 8) as soon as it beats the parent.
7. **Confirm.** Run `base` and the two best on `holdout256 --n 4`. Keep `CLS_075` unless another candidate is better
   on both dev128 and holdout256 by more than one standard error; never keep a candidate with a practice source
   clearly worse than the parent. If time allows, a single-source split is the most precise check of Part C:
   `python derive/onesource.py co gsm1 --source gsm8k_train --n 256 --exclude-files "ANS.*.jsonl"`, then `prun.py` on
   `gsm1 --n 4` for `base` and the chosen edit. Control, if at least 40 minutes remain: the same rows' weights on
   random segments (`step_rows.py ... --random-control`), fitted and exported the same way and joined with the same
   Parts A and B, on dev128; it should be clearly below the real one. If nothing beats the parent on both dev128 and
   holdout256, upload nothing worse than the parent.
8. **Upload.** Write REPORT.md: the cut shares and kinds per source, both directions' stability and their cosine, the
   number of mixed problems, traces, rows (up and down) and the fit's held-out loss, the screen table, the holdout and
   the control. Package without uploading: `python -m wsp.upload co <edit> --report REPORT.md --dry-run`; check it:
   `axonry-check $WSP_WORK/deliver/co__<edit>/edit --task task.json`; then the same upload command with
   `--task task.json` instead of `--dry-run`. Re-upload when a better candidate is confirmed.

## What our exploration found (practice splits only)
These are the same three parts, run by us on 0009547524-0008 (the parent's own parent; the parent's own change targets
how its answers are written) with the toolkit's sampling and grading, paired with that model on the same problems and seeds
(4 answers each). Support for what to expect, not a fixed answer: the search decides.
- **All three parts** (conclude 0.75, loop 1, Part C at dose 0.5 from 650 rows of 313 mixed problems): dev128
  +2.7 +- 1.6, holdout256 +1.5 +- 1.4, three further mixed splits +3.2 / +2.1 / +1.2; two 512-problem `gsm8k_train`
  splits +4.2 and +2.7 (+- 0.9-1.0); a 256-problem `ifsynth` split +2.5; a 256-problem `mmlupro` split +1.4; three
  384-problem `math_train` splits +1.6 / +1.4 / +2.0. Positive on every split we ran.
- **Parts A and B without Part C:** the same splits give about half the word-problem gain (+2.2 on the first
  `gsm8k_train` split, +1.3 on the second).
- **Part C alone:** dev128 +2.5, a further mixed split +3.5, the first `gsm8k_train` split +2.4; little on `math_train`
  and `ifsynth` alone. It works on the word problems, where the values are informative.
- **Control:** the same weights on random segments of the same traces, joined with Part A: dev128 +2.5 against +2.9,
  the second `gsm8k_train` split +1.5 against +2.8, holdout256 +0.2 against +1.4 for the real rows.
- More rows help: with 156 rows (73 mixed problems) the word-problem gain of the joined edit was +2.0 and +0.8 on the
  two `gsm8k_train` splits; with 650 rows +3.8 and +2.8.

## What does not work (measured on practice; do not spend the run on it)
- **A higher dose of Part C:** dose 1 or 2 costs instruction following (`ifsynth` -5 to -8) and math.
- **Values from forced answers** (close the thinking at the checkpoint and sample short answers) instead of full
  continuations: the rows cost instruction following at dose 1 and above, and at dose 0.5 they add less than rows
  valued by continuations.
- **Training the model to close where its forced answer is already right:** it learns to close after any line, and
  accuracy collapses (-20 points and more).
- **Part A at four attention layers (11, 15, 19, 23)** or at strength 1.25 and above: no better on average; worse on
  `math_train`.
- **Part B at strength 0.5 or 1.5:** no better than 1.

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: the directions, the values, the rows and the gains are all measured and fitted in the run.
