# Two states the gains left behind: "keep verifying" and the loop

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md).

## The failure that is left
The parent is a loop-removal edit followed by self-distilled gains on MLP units and attention heads. Measure what
still goes wrong on its model first (step 1). In our exploration on the practice dev split (512 problems x 2
answers), 14.7% of the parent's answers still ran into the 32k-token limit and scored zero: 8.8% ended in an exact
loop, the other 5.8% in long non-repeating re-checking. The answers that finished were right 73% of the time.

These are two different states, and both survive the gains:
- **Keep verifying.** Once the model has stated a result, it often re-checks it instead of closing the thinking.
- **The loop.** The model starts repeating its own earlier text and never leaves it.

Measured on the parent's own answers, the two directions are nearly orthogonal (cosine -0.19 to +0.13 at the
attention layers). They can be removed separately, from different attention writers, and the two removals add up.

## Why it can be edited, and where
Both directions come from the toolkit's contrasts (`wsp/directions.py`), measured on the parent's own text:
- `conclude`: take a correct, finished answer and cut it right after the thinking first states the right result. The
  good branch closes the thinking (`</think>` and the model's own final answer); the bad branch keeps verifying (the
  model's own next text).
- `loop`: take a correct answer and cut it at a sentence boundary. The good branch is the model's own continuation;
  the bad branch repeats the model's earlier text, as a loop does.

In both, the two branches share the prefix, so the per-layer mean difference of the residual stream isolates the
state. Each direction is projected out of the attention outputs (`self_attn.o_proj`), the writers that add to the
residual stream at every token. The conclude direction is removed at attention 15 and 19. The loop direction is
removed at attention 11 and 23: the parent's ancestor already removed a loop direction at 15 and 19, and on this
parent the loop state that is left is best reached at the other attention layers. Because the two removals touch
different tensors, they stack without interacting. How strongly to remove each is found by a paired search on
practice problems, with a holdout and a random control.

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`.
- `prun.py` is `wsp.case run` split over all GPUs: the same seeds and files, N times faster.
- `more_answers.py` samples the parent on further practice problems outside the case's dev and holdout splits.
- `pairs_more.py` builds pairs for one contrast from the parent's answers and saves that direction under its own name.
- `grid.py` writes one `dir_ablate` candidate per (attention layer set, strength) from a measured direction
  (`--prefix` names the family) and ranks results; `stack.py` stacks candidates that touch different tensors.
- `kinds.py` shows where the answers go.

On 4x RTX 4090, one model on dev128 x 4 takes about 15-25 minutes, and on holdout256 x 4 about 25-40 minutes. Start
long runs detached (`nohup` or `tmux`) and keep PROGRESS.md current.

1. **Case and the parent's answers** (practice only):
   `python -m wsp.case init concl --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`
   Check that the parent chain is complete (`parent/edit/` holds the gains and, under its own `parent/`, the
   loop-removal edit). Then `python derive/prun.py concl base dev --n 2` and `python derive/kinds.py concl dev`.
   Check that a part of the answers is still cut, in loops and in long re-checking.
2. **More of the parent's own answers**, to measure the directions well: on each GPU i of 4,
   `python derive/more_answers.py concl base ANS --per-source 256 --n 4 --max-tokens 12288 --shard i/4` (about 25
   minutes). These problems are outside dev and holdout.
3. **Measure the two directions.**
   `python derive/pairs_more.py concl --kind conclude --sources gsm8k math --sets dev --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out concl`
   (the conclude contrast needs a gold answer it can find in the thinking, so it uses these two sources), then
   `python derive/pairs_more.py concl --kind loop --sources gsm8k math mmlupro ifsynth --sets dev --files ANS.*.jsonl --exclude dev128 holdout256 --n 512 --out loop --compare concl_dirs.npz`.
   Go on only if both are stable at the attention layers (split-half stability well above 0.9) and the two are far
   from parallel (|cosine| well below 0.5).
4. **Starting runs to pair with:** `python derive/prun.py concl base dev128 --n 4` and `... base holdout256 --n 4`.
5. **Search the strengths, alone and stacked.** With `W=$WSP_WORK/cases/concl`:
   `python derive/grid.py build concl --dirs $W/concl_dirs.npz --sets 15,19 --alphas 0.75 1 1.25 --prefix C` and
   `python derive/grid.py build concl --dirs $W/loop_dirs.npz --sets 11,23 --alphas 0.5 1 1.5 --prefix L`.
   Stack the conclude candidate at strength 1 with each loop candidate:
   `python derive/stack.py concl S_C1_L1 C15-19_a1 L11-23_a1` (and likewise for L11-23_a0.5 and L11-23_a1.5).
   Run each on `dev128 --n 4` and rank with `python derive/grid.py table concl dev128`.
   - If time is short, run C15-19_a1, then the stacks with loop strength 1 and 1.5.
   - Upload the best stack early (step 7), and keep confirming while time allows.
6. **Confirm and control.** Run the two best stacks on `holdout256 --n 4` and pick the one with the higher holdout
   gain; when two are within noise, prefer the one with no practice source clearly worse. Build the control: a random
   direction on each of the same tensors at the same strengths (`python derive/grid.py control concl C15-19_a1 --dirs
   $W/concl_dirs.npz` and the same for the loop candidate, then `stack.py` the two `RAND_` edits). Run it on dev128;
   it should be within noise of 0. Read some rescued and some lost answers: loops should end and re-checking should
   close once the result is stated, with the final answer in the task's format. If no stack beats the parent on both
   dev128 and holdout256, upload nothing worse than the parent.
7. **Upload.** Write REPORT.md: the measured failure, both directions' stability and their cosine, the search table,
   the holdout and the control. Package without uploading: `python -m wsp.upload concl <edit> --report REPORT.md
   --dry-run`. Check it: `axonry-check $WSP_WORK/deliver/concl__<edit>/edit --task task.json`. Then run the same
   upload command with `--task task.json` instead of `--dry-run`. Re-upload when a better candidate is confirmed.

## What our exploration found (practice splits only)
These are the same steps, run by us on the parent's model, with the toolkit's sampling and grading. They are support
for what to expect, not a fixed answer: the search decides.
- **The parent** (dev128 x 4): 60.0% right, 13.5% cut.
- **Directions:** conclude 512 pairs from 288 traces, split-half stability 0.996 at attention 11-23; loop 512 pairs
  from 413 traces, stability 0.97-0.998; cosine between them -0.19 to +0.13.
- **Search on dev128 x 4, paired gain over the parent:** conclude alone at attention 15, 19: strength 0.75 +3.5,
  1 +3.5, 1.25 +2.3. Loop alone at attention 11, 23, strength 1: +1.2. Stacked (conclude 1 + loop): loop 0.5 +4.3,
  loop 1 +3.9, loop 1.5 +6.4, loop 2 +2.3. Loop spread over attention 3, 7, 11, 23 instead: +3.5.
- **Holdout256 x 4** (1024 answers, never used for a choice): conclude alone +2.7; stacked with loop 0.5 +4.2;
  stacked with loop 1 +5.5 (63.0% -> 68.5%: `gsm8k_train` +3.5, `ifsynth` +3.1, `math_train` +5.9, `mmlupro` +9.4;
  cut answers 12.8% -> 7.3%). With the conclude direction measured from fewer pairs (128) the stack still gains
  +4.4 on holdout256. Loop strength 1.5 reaches +6.1 on holdout256 but is less even on the further splits.
- **Three further splits** (256 practice problems x 4 each, used for nothing else): stacked with loop 1 +2.1, +3.6,
  +2.5; conclude alone 0.0, +3.1, +2.0. Over all 4608 answers of the five splits the stack gains about +3.5 points and
  conclude alone about +2.1: the loop removal adds about +1.4, and it lifts `math_train`, which conclude alone does not.
- **Random control** (random directions on the same four tensors and strengths): +0.8 +- 1.4 on dev128, against +3.9 for the measured stack and +1.4 +- 1.3 on holdout256, against +5.5.

## What does not work (measured on practice; do not spend the run on it)
On this parent:
- **Self-distillation again**: three fits of per-unit and per-head gains on the parent's own correct answers to 520
  further practice problems, averaged. The held-out loss falls by 10%, but practice accuracy does not rise (-1.4 and
  -1.6 at two strengths). The parent already holds what the model can learn from its own answers.
- **Too much removal**: loop strength 2 (+2.3 against +3.9..+6.4 at 1..1.5), conclude strength 1.25 and above.
- **A doubt direction** (wrong answers that had stated the right result, closed there vs their own doubting text),
  added to conclude: large on dev128, not on the further splits.

On a close relative (the loop-removal model with the conclude direction removed from attention 11, 15, 19, 23):
- **Learned low-rank filters** fitted to the model's own shortest correct answers: they shorten the answers but turn
  cut `math_train` answers into wrong ones (nine variants, centred on zero).
- **Readers instead of writers** (MLP `gate_proj` / `up_proj` stop reading the direction): -0.6 to -5.3.

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: both directions, the writers and the strengths are measured and chosen in the run.
