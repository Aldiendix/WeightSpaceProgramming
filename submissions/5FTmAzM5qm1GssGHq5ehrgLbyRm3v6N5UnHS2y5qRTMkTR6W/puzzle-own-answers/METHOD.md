# Credit the parent's own right answers to reasoning puzzles, measured against its own wrong ones

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md). The parent already closes its thinking in
time, repairs failed answers and prefers its own right math answers; it has never been shaped on puzzles that need a
procedure carried out step by step (ordering items, tracking swaps, checking schedules, following a route). This method
adds that: on the toolkit's generated reasoning puzzles (`practice:bbhsynth`, exact answers), the parent's own right
answers are made more likely and its own wrong answers less likely, per puzzle, with nothing else changed.

## Time rules for this method (read first)
- **No command may run longer than 40 minutes.** One background job is cut at 60 minutes; every step below is sized
  to stay under that (the sampling is split into two rounds).
- **Write PROGRESS.md after every numbered step** (what finished, the numbers, the next command). If you are asked to
  continue, read PROGRESS.md first and go on from there.
- **Upload as soon as a candidate beats the parent** on the puzzle hold-out (step 5), then replace it only if the other
  dose is better.
- Start each long command detached (`nohup ... > log 2>&1 &`) and check its log every few minutes; never wait on a
  command for more than 15 minutes without looking at its progress lines.

## The failure that is left
On generated puzzles of fixed form the parent is right on about two thirds of its samples. On a third of the puzzles
its four samples disagree: the same question is solved in one sample and lost in another, by a slip in carrying out the
procedure (a swap tracked to the wrong person, a slot taken for free because it was mentioned, a word misplaced in an
ordering) or by a wrong reading of what is asked. The model can do these puzzles; it does not yet prefer the way of
working that gets them right.

## The edit: group-relative credit on whole answers
For every training puzzle the parent answers 4 times (normal sampling settings, budget 16384 tokens). On a puzzle whose
samples disagree, each finished sample becomes a training row whose target is the sample itself (its thinking and its
answer; the first 3072 tokens of it are fitted) and whose weight is its advantage: its grade (1 right, 0 not) minus the
mean grade of the puzzle's other samples. Right samples on a mostly-wrong puzzle get the most credit, wrong samples on a
mostly-right puzzle the most blame; puzzles the parent always or never solves give no rows. This is a policy-gradient
step with a group baseline, measured on the model's own answers.

The parameters are per-unit gains of the MLPs (`scale_rows` on `up_proj`) and per-head gains of the attention and
linear-attention outputs (`scale_cols`), layers 4 to 21: different ops from the parent's own, they stack with it. The
fit learns log-gains and **is taken at its best held-out step** (a few puzzles are held out of the fit; their right
samples measure it). The edit applies a **dose** of them (0.25 or 0.5 of the learned log-gains): a full dose costs
word problems and puzzles alike (see below).

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`. `prun.py` is `wsp.case run` split over all GPUs (the
same seeds and files, N times faster); `kinds.py CASE SET` shows where a screen's answers go (right, finished wrong,
cut), `answer_kinds.py` the same for sample.py's files.

| step | what | about (4x RTX 4090) |
|---|---|---|
| 1 | case, puzzle hold-out | 10 min |
| 2 | the parent's own answers (two rounds) | 2 x 30 min |
| 3 | rows, fit, export | 20 min |
| 4 | screen the parent and dose 0.5 on the puzzle hold-out, upload | 45 min |
| 5 | dose 0.25 if needed, mixed hold-out check, final upload | 60 min |

1. **Case and puzzle hold-out** (practice only):
   `python -m wsp.case init co --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`.
   Check that the parent chain is complete (`parent/edit/` and its own parents) and that
   `python -c "from wsp import case; print(case.model_path('co', 'base'))"` builds the parent model. Then
   `python derive/puzzle_split.py co --name puzzle_hold --per-family 20` (380 puzzles, 20 of each family, kept out of
   every fit).
2. **The parent's own answers** on the other puzzles, in two rounds; each round is one command that uses all four GPUs
   (the toolkit's generation spreads one call over every visible GPU): round A
   `python derive/sample.py co base PZ --source bbhsynth:2048 --n 4 --max-tokens 16384 --shard 0/2`, then round B the
   same command with `--shard 1/2`. Then `python derive/answer_kinds.py PZ.*.jsonl`
   (expect about two thirds right, a few percent cut, a third of the puzzles mixed).
3. **Rows, fit, export.** `python derive/group_rows.py co ROWS.jsonl PZ.*.jsonl --span full` (expect 1500-2500 rows,
   both signs). Fit on one GPU, with `MP` = the parent model directory and S = 2 x rows / 8:
   `python derive/gains_fit.py train $MP ROWS.jsonl FIT --heldout ROWS.held.jsonl --rank 0 --unit-gains --gains --gain-layers 4 21 --lr 2e-3 --accum 8 --steps S --warmup 5 --max-tokens 3072 --eval-every 25 --log-every 10 > FIT.log 2>&1`.
   Then `python derive/best_step.py FIT.log` prints the best held-out step B (it must lower the held-out loss below
   step 0; if it does not, stop and report). Export
   `python derive/gains_fit.py export FIT co PUZ_05 --model $MP --scale 0.5 --step B` and the same with
   `PUZ_025 --scale 0.25 --step B`.
4. **Screen and upload.** `python derive/prun.py co base puzzle_hold --n 4`, then
   `python derive/prun.py co PUZ_05 puzzle_hold --n 4`; `python derive/grid.py table co puzzle_hold`. If PUZ_05 beats
   the parent there, upload it right away (step 5 commands).
5. **Other dose, check, final upload.** If PUZ_05 did not beat the parent on puzzle_hold, screen `PUZ_025` the same way.
   Take the dose with the larger puzzle_hold gain (if neither gains, PUZ_025, the smaller change). Check it on the
   mixed hold-out: `python derive/prun.py co base holdout256 --n 4` and the same for the chosen edit; if it loses more
   than 2 points there and the other dose does not, take the other dose. Write REPORT.md: the number of puzzles,
   samples (right / wrong / cut), mixed puzzles, rows (up and down), the fit's held-out loss at step 0 and at step B,
   and the screen tables. Package without uploading: `python -m wsp.upload co <edit> --report REPORT.md --dry-run`;
   check it: `axonry-check $WSP_WORK/deliver/co__<edit>/edit --task task.json`; then the same upload command with
   `--task task.json` instead of `--dry-run`.

## What our exploration found (practice splits only)
Paired with the model the gains were fitted on, on the same problems and seeds (4 answers each), on a model of the
same family as the parent (a lab build with the same kinds of rounds); support for what to expect, not a fixed answer:
the screen decides.
- Its own answers to 1664 training puzzles: 67.8% right, 18.0% finished wrong, 14.2% cut at 16384 tokens; 573 puzzles
  mixed, 1799 rows (1252 up, 547 down). The fit's held-out loss 0.2340 at step 0, best 0.2142 at step 200 of 426.
- Dose 0.5: a 256-puzzle split +0.7 +- 1.1 (rescued 76, lost 69); mixed hold-out of 256 practice problems +1.6 +- 1.1
  (word problems +3.4, math +2.9, multiple choice +1.5, instruction requests -1.5).
- Dose 1: mixed hold-out -1.5 +- 1.1 (word problems -3.9) and puzzles -1.9 +- 1.3: too strong; use 0.5 or 0.25.

## What does not work (measured on practice; do not spend the run on it)
- **Rows whose prompt is the whole thinking** (fitting only the final answer after it): the thinking of these puzzles
  runs to 10-16 thousand tokens, and such rows do not fit in one GPU's memory. Fit the whole sample from its start.
- **A full dose** (above).
- **Long single commands:** sampling all training puzzles at once takes more than an hour; step 2 is split in two.

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: the answers, rows and gains are all measured and fitted in the run.
