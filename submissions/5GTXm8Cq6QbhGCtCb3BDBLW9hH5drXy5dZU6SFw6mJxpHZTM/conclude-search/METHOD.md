# Conclude-search: find where the model's "keep verifying" state is written, and remove it there

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md).

## The failure that is left
Measure it first on the parent's model (step 1). On the loop-removal parent, a quarter of the practice answers still
run into the 32k-token limit and score zero, and most of them are no longer exact loops. The model has found an
answer, often the right one, and keeps re-checking it, re-deriving it or weighing the options again until the limit.
In our exploration on the practice dev split (512 problems x 2 answers), 25% of the parent's answers were cut, and only
5.5% of them in exact loops. Half of the `math_train` answers were cut, yet the `math_train` answers that finished were
right 97% of the time. Among the cut `gsm8k_train` and `math_train` answers, 38% and 19% had already stated the correct
result earlier in the thinking. That is a missing behaviour: **stop once the answer is found**. Removing loops further
does not supply it.

## Why we think it can be edited, and where
Right after the model first states a result, it chooses between closing the thinking and verifying again. If that
choice is carried by one direction in the residual stream, the toolkit can measure it on the model's own text
(contrast `conclude`, `wsp/directions.py`). Take a correct, finished answer and cut it right after the thinking first
states the right result. The good branch closes the thinking (`</think>` and the model's own final answer). The bad
branch keeps verifying (the model's own next text). Both share the prefix, so the per-layer mean difference of the
residual stream isolates "keep verifying".

The open question is **where** this state is written and **how strongly** to remove it. A direction in the residual
stream has many writers. Removing it from the wrong ones, or too strongly, cuts answers short or damages reasoning.
So the writers and the strength are found by a paired search on practice problems, with a holdout and a random
control, not assumed. The attention outputs (`self_attn.o_proj` of the full-attention layers 3, 7, 11, 15, 19, 23)
write into the residual stream at every token and are the natural first family. The method also shows how to test
other writers (MLP `down_proj`, linear-attention `out_proj`) with the same script.

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`.
- `prun.py` is `wsp.case run` split over all GPUs: the same seeds and files, N times faster.
- `pairs_src.py` builds conclude pairs from chosen practice sources.
- `writers.py` writes `dir_ablate` specs from the measured directions for any writer family; `grid.py` builds a
  candidate grid and ranks the candidates.
- `kinds.py` shows where the answers go.

On 4x RTX 4090, one model on dev128 x 4 takes about 15-25 minutes, and on holdout256 x 4 about 25-40 minutes. Start
long runs detached (`nohup` or `tmux`) and keep PROGRESS.md current.

1. **Case and the parent's answers** (practice only, with the conclude contrast):
   `python -m wsp.case init concl --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`
   `python derive/prun.py concl base dev --n 2`, then `python derive/kinds.py concl dev`. Check that cut answers are
   mostly not exact loops and that finished answers are mostly right. If neither holds, this method does not apply.
2. **Measure the direction.**
   `python derive/pairs_src.py concl --sources gsm8k math --sets dev --exclude dev128 holdout256 --n 128` (the
   conclude contrast needs a gold answer it can find in the thinking, so only these two practice sources are used),
   then a held-out set: the same command with `--exclude-pairs pairs --n 48 --out pairs_heldout`. Then
   `python -m wsp.case extract concl`. Go on only if the direction is stable (split-half stability well above 0.9)
   and separates the pairs at the attention layers. Check this with
   `python derive/writers.py show $WSP_WORK/cases/concl/directions.npz`.
3. **Starting runs to pair with:** `python derive/prun.py concl base dev128 --n 4` and `... base holdout256 --n 4`.
4. **Search the writers and the strength.**
   `python derive/grid.py build concl --dirs $WSP_WORK/cases/concl/directions.npz` builds one edit per (writer set,
   strength). The sets are the parent's own writers (attention 15, 19), the attention layers from the middle up
   (11, 15, 19, 23) and all six. The strengths are 0.75, 1 and 1.25: 1 projects the component out, and more starts to
   flip it. Run each on `dev128 --n 4` (`python derive/prun.py concl <edit> dev128 --n 4`). Then
   `python derive/grid.py table concl dev128` ranks them by paired gain over the parent. Upload the best one early
   (step 6), and keep searching if time allows.
   - If time is short, run strength 1 for the three sets first, then the other strengths for the best set.
   - With time left, test one other writer family on top of the best attention candidate. For example,
     `writers.py build ... --d 15 19 --topk 128` makes an MLP variant. Keep it only if it beats the attention-only
     candidate on both dev128 and holdout256.
5. **Confirm and control.** Run the two best on `holdout256 --n 4` and pick the one with the higher holdout gain.
   Build its random control with `python derive/grid.py control concl <edit> --dirs ...`: a random direction on the
   same tensors and strength. Run it on dev128; it should be within noise of 0. Read some rescued and some lost
   answers: the thinking should close once the result is stated, and the final answer should keep the task's format.
   If no candidate beats the parent on both dev128 and holdout256, upload nothing worse than the parent.
6. **Upload.** Write REPORT.md: the measured failure, the direction's stability, the search table, the holdout and the
   control. Package without uploading: `python -m wsp.upload concl <edit> --report REPORT.md --dry-run`. Check it:
   `axonry-check $WSP_WORK/deliver/concl__<edit>/edit --task task.json`. Then run the same upload command with
   `--task task.json` instead of `--dry-run`. Re-upload when a better candidate is confirmed.

## What our exploration found (practice splits only)
These are the same steps, run by us on the parent's model, with the toolkit's sampling and grading. They are support
for what to expect, not a fixed answer: the search decides.
- **The direction is stable.** Split-half stability is 0.98-0.99 at the attention layers, and the pair agreement is
  100%. Two measurements from independent samples of the parent's answers agree with cosine 0.98-0.99. It is nearly
  orthogonal to the parent's own loop direction (cosine 0.01 at layer 15, -0.21 at layer 19).
- **Search on dev128 x 4, paired gain over the parent:**
  - attention 11, 15, 19, 23 at strength 1: +7.0 +- 2.2; at 0.75: +7.0 +- 2.1; at 1.25: +5.3 +- 2.2
  - all six attention layers at strength 1: +5.7 +- 2.2
  - attention 15, 19 at strength 1: +4.1 +- 2.2
- **Holdout256 x 4** (1024 answers, never used for a choice): the best candidate gains +6.2 +- 1.5 (60.1% -> 66.3%).
  Cut answers fall from 22.2% to 11.3%, and answers get a third shorter.
- **Random control** (same tensors, strength 1): -1.8 +- 1.9 on dev128.
- **Stacking more on the best candidate lowers practice accuracy:** the same direction also on MLP `down_proj`
  (-2.0), on linear-attention `out_proj` (-2.1), or a second round re-measured on the edited model (-2.3). Past the best
  strength, more concluding costs more right answers than it rescues.

## What does not work on this parent (measured on practice; do not spend the run on it)
- **More loop-direction removal**, re-measured on the parent's model. MLP `down_proj` writers (-3.5 / -9.4),
  Gated-DeltaNet `out_proj` (-1.2 to -7.2, with more answers cut) and other attention layers (+0.2 / +0.0 / -0.2) are
  neutral or harmful. More strength at the parent's own layers gives -14.8.
- **Closing the thinking deep in a loop** (an exit unit): forced answers out of a loop are rarely right on
  `math_train` (about 10%).

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. It ships
no arrays: the direction, the writers and the strength are all measured and chosen in the run.
