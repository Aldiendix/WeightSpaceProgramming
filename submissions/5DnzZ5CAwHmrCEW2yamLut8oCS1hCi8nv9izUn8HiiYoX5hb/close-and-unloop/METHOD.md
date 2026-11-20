# Close and unloop: two states the self-distilled gains left behind, removed with a searched loop strength

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md).

## The failure that is left
The parent is a loop-removal edit followed by self-distilled gains on MLP units and attention heads. Measure what
still goes wrong on its model first (step 1). In our exploration on practice problems, about 13% of the parent's
answers still ran into the 32k-token limit and scored zero: a little over half ended in an exact loop, the rest in
long non-repeating re-checking. The finished answers were right about 73% of the time. The cut answers are not spread
evenly: on `math_train` problems about 30% of the parent's answers were cut, on `gsm8k_train` problems under 5%.

Two different states produce them, and both survive the gains:
- **Keep verifying.** Once the model has stated a result, it often re-checks it instead of closing the thinking.
- **The loop.** The model starts repeating its own earlier text and never leaves it.

Measured on the parent's own answers, the two directions are nearly orthogonal (cosine -0.19 to +0.13 at the
attention layers), so they can be removed separately, from different attention writers, and the removals add up.
How strongly to remove the loop direction matters most, and it is searched: a stronger removal ends more loops on
the long problems, too strong a removal costs answers elsewhere.

## Why it can be edited, and where
Both directions come from the toolkit's contrasts (`wsp/directions.py`), measured on the parent's own text:
- `conclude`: take a correct, finished answer and cut it right after the thinking first states the right result. The
  good branch closes the thinking (`</think>` and the model's own final answer); the bad branch keeps verifying (the
  model's own next text).
- `loop`: take a correct answer and cut it at a sentence boundary. The good branch is the model's own continuation;
  the bad branch repeats the model's earlier text, as a loop does.

In both, the two branches share the prefix, so the per-layer mean difference of the residual stream isolates the
state. Each direction is projected out of the attention outputs (`self_attn.o_proj`), the writers that add to the
residual stream at every token: conclude at attention 15 and 19; loop at attention 11 and 23 (the parent's ancestor
already removed a loop direction at 15 and 19, and on this parent the loop state that is left is best reached at the
other attention layers). The two removals touch different tensors, so they stack without interacting.

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`.
- `prun.py` is `wsp.case run` split over all GPUs: the same seeds and files, N times faster.
- `more_answers.py` samples the parent on further practice problems outside the case's dev and holdout splits.
- `pairs_more.py` builds pairs for one contrast from the parent's answers and saves that direction under its own name.
- `grid.py` writes one `dir_ablate` candidate per (attention layer set, strength) from a measured direction
  (`--prefix` names the family) and ranks results; `stack.py` stacks candidates that touch different tensors.
- `onesource.py` makes a practice split of one source only (to measure where the cut answers are).
- `kinds.py` shows where the answers go.

On 4x RTX 4090, one model on dev128 x 4 takes about 15-25 minutes, and on holdout256 x 4 about 25-40 minutes. Start
long runs detached (`nohup` or `tmux`) and keep PROGRESS.md current.

1. **Case and the parent's answers** (practice only):
   `python -m wsp.case init cu --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`
   Check that the parent chain is complete (`parent/edit/` holds the gains and, under its own `parent/`, the
   loop-removal edit). Then `python derive/prun.py cu base dev --n 2` and `python derive/kinds.py cu dev`: check that
   a part of the answers is still cut, in loops and in long re-checking, and on which source.
2. **More of the parent's own answers**, to measure the directions well: on each GPU i of 4,
   `python derive/more_answers.py cu base ANS --per-source 256 --n 4 --max-tokens 12288 --shard i/4` (about 25
   minutes). These problems are outside dev and holdout.
3. **Measure the two directions.**
   `python derive/pairs_more.py cu --kind conclude --sources gsm8k math --sets dev --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out concl`
   (the conclude contrast needs a gold answer it can find in the thinking, so it uses these two sources), then
   `python derive/pairs_more.py cu --kind loop --sources gsm8k math mmlupro ifsynth --sets dev --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out loop --compare concl_dirs.npz`.
   Go on only if both are stable at the attention layers (split-half stability well above 0.9) and the two are far
   from parallel (|cosine| well below 0.5).
4. **Starting runs to pair with:** `python derive/prun.py cu base dev128 --n 4` and `... base holdout256 --n 4`.
5. **Search the strengths, alone and stacked.** With `W=$WSP_WORK/cases/cu`:
   `python derive/grid.py build cu --dirs $W/concl_dirs.npz --sets 15,19 --alphas 0.75 1 --prefix C` and
   `python derive/grid.py build cu --dirs $W/loop_dirs.npz --sets 11,23 --alphas 1 1.5 2 --prefix L`.
   Stack the conclude candidate at strength 1 with each loop candidate:
   `python derive/stack.py cu S_C1_L1.5 C15-19_a1 L11-23_a1.5` (and likewise for L11-23_a1 and L11-23_a2).
   Run each on `dev128 --n 4` and rank with `python derive/grid.py table cu dev128`.
   - If time is short, run the stacks with loop strength 1.5 and 1 first, then C15-19_a1 alone.
   - Upload the best stack early (step 7), and keep confirming while time allows.
6. **Confirm and control.** Run the two best stacks on `holdout256 --n 4`. Build the control: a random direction on
   each of the same tensors at the same strengths (`python derive/grid.py control cu C15-19_a1 --dirs
   $W/concl_dirs.npz` and the same for the loop candidate, then `stack.py` the two `RAND_` edits); run it on dev128.
   It should be within noise of 0. If at least 1.5 hours remain, measure where the gain sits: `python
   derive/onesource.py cu math1 --source math_train --n 192 --exclude-files "ANS.*.jsonl"`, then `prun.py` the base
   and the two stacks on `math1 --n 4`. Pick the stack with the higher holdout gain; when the two are within noise,
   prefer the one with the larger gain on the `math_train` split (that is where most cut answers are) as long as no
   practice source is clearly worse. Read some rescued and some lost answers: loops should end and re-checking
   should close once the result is stated, with the final answer in the task's format. If no stack beats the parent
   on both dev128 and holdout256, upload nothing worse than the parent.
7. **Upload.** Write REPORT.md: the measured failure (per source), both directions' stability and their cosine, the
   search table, the holdout, the control and the single-source split if it was run. Package without uploading:
   `python -m wsp.upload cu <edit> --report REPORT.md --dry-run`. Check it: `axonry-check
   $WSP_WORK/deliver/cu__<edit>/edit --task task.json`. Then run the same upload command with `--task task.json`
   instead of `--dry-run`. Re-upload when a better candidate is confirmed.

## What our exploration found (practice splits only)
These are the same steps, run by us on the parent's model, with the toolkit's sampling and grading. They are support
for what to expect, not a fixed answer: the search decides.
- **The parent** (dev128 x 4): 60.0% right, 13.5% cut.
- **Directions:** conclude 512 pairs from 288 traces, split-half stability 0.996 at attention 11-23; loop 512 pairs
  from 413 traces, stability 0.97-0.998; cosine between them -0.19 to +0.13.
- **Search on dev128 x 4, paired gain over the parent:** conclude alone at attention 15, 19: strength 0.75 +3.5,
  1 +3.5. Stacked with conclude 1, loop strength 0.5 +4.3, 1 +3.9, 1.25 +4.5, 1.5 +6.4, 2 +2.3.
- **Holdout256 x 4:** conclude alone +2.7; stacked with loop 1 +5.5, loop 1.25 +4.0, loop 1.5 +6.1.
- **Four further splits** (256 practice problems x 4 each, used for nothing else): loop 1 +2.1, +3.6, +2.5, +2.5;
  loop 1.5 +1.8, +2.9, +1.2, +4.1. Over all six splits (5632 answers) the stack with loop 1.5 gains +3.5 +- 0.6
  points (`gsm8k_train` +3.7, `math_train` +2.6, `mmlupro` +4.6, `ifsynth` +3.1) and with loop 1 +3.3; conclude
  alone about +2.1.
- **Single-source splits:** on three `math_train` splits (384 problems x 4 each) the parent cut about 29% of its
  answers; the stack with loop 1.5 gains +4.4, +2.6, +3.5 (cut down to about 22%), with loop 1 +4.2, +2.3, +3.0, with
  loop 1.25 and 2 about the same as with 1.5 (+3.4 to +3.5 over the three). On a
  `gsm8k_train` split (512 problems x 4) the parent cut only 4.7% and the stacks gain +1.3 to +2.0: there is little
  left to rescue there.
- **Random control** (random directions on the same four tensors and strengths as the loop-1.5 stack): +1.0 +- 1.8
  on dev128 and -0.6 +- 1.3 on holdout256, against +6.4 and +6.1 for the measured stack.

## What does not work (measured on practice; do not spend the run on it)
On this parent:
- **Self-distillation again** (per-unit and per-head gains on its own correct answers, three fits averaged), and a
  **further dose of the parent's own gains** (dose 1.0 / 1.5 / 1.75 instead of the parent's 1.25): -1.6 to +0.2.
- **Too much removal:** loop strength 2, conclude strength 1.25 and above.
- **Source-specific directions** (conclude measured on `gsm8k_train` only, loop on `math_train` or `mmlupro` only):
  cosine 0.99 with the mixed ones and no better; conclude also removed at attention 11/23: lower on the further splits.
- **A doubt direction** (wrong answers that had stated the right result, closed there vs their own doubting text),
  added to conclude: large on dev128, not on the further splits. **A right-versus-wrong direction** (toolkit contrast
  `correct`): not stable (split-half 0.1-0.45).
- **Short-finish self-distillation into rewritten MLP units** (the model's own shortest right answers, at most 4000
  tokens, ~600 units by gradient), alone or on top of the stack: instruction following falls by 8-14 points; with
  those answers kept fixed it recovers, but stays 2 points below the stack alone on holdout256.

On a close relative (the loop-removal model with the conclude direction removed from attention 11, 15, 19, 23):
- **Learned low-rank filters** fitted to the model's own shortest correct answers, and **readers instead of
  writers** (MLP `gate_proj` / `up_proj` stop reading the direction): centred on zero or below.

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: both directions, the writers and the strengths are measured and chosen in the run.
