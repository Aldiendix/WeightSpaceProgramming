# Conclude once the answer is found

This entry stacks on its parent (`entry.json`, sid `0009531781-0011`): the published loop-removal edit, which
removes the loop direction from the attention output (`self_attn.o_proj`) of layers 15 and 19. Start from that
model: `python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md).

## The failure that is left
On the parent's model, exact loops are mostly gone, but a quarter of the answers still run into the 32k-token
limit and score zero. Most of them are not loops any more. The model has found an answer, and often the right one,
but it keeps re-checking it, re-deriving it, or weighing the options again until the limit. Measured on the parent's
model (practice dev, 512 problems x 2 answers, four sources, model-card sampling):

| practice source | accuracy | cut at 32k | of which exact loops | right when finished |
|---|---|---|---|---|
| GSM8K | 67.2% | 16.4% | 7.8% | 80.4% |
| MATH | 47.3% | 51.2% | 5.5% | 96.8% |
| MMLU-Pro | 40.6% | 25.4% | 3.1% | 54.4% |
| IF | 74.6% | 6.6% | 5.5% | 79.9% |

On MATH, half of the answers are cut, yet the answers that finish are right 97% of the time. Among cut GSM8K and MATH
answers, 38% and 19% had already stated the correct result earlier in the thinking. More loop removal does not help
here (see "What does not work"). The missing behaviour is to **stop once the answer is found**.

## The edit
The toolkit's `conclude` contrast (`wsp/directions.py`) measures this behaviour on the model's own text. It takes a
correct, finished answer and cuts it right after the thinking first states the right result. At that point the good
branch closes the thinking (`</think>` and the model's own final answer) and the bad branch keeps verifying (the
model's own next text). Both branches share the same prefix. Per layer, `u = mean(residual | keep verifying) -
mean(residual | close)` over the continuation tokens, on the **parent's** model.

The edit removes that direction (`dir_ablate`, side `out`, alpha 1) from the attention output (`self_attn.o_proj`) of
four attention layers: 11, 15, 19 and 23. The parent edited 15 and 19 for loops; 11 and 23 are the other attention
layers from the middle of the network up. Alpha 1 projects the component out of what the attention writes at those
layers, so the attention no longer pushes "keep verifying". Alpha 2 (a sign flip) cuts more answers short and is
worse, and so are layers 15 and 19 alone or all six attention layers (below).
The result is 4 tensors and 16 KB of payload. Measured against the parent, as the limits are, the change is
2.5-3.5% per tensor and 0.4% globally. The conclude direction is nearly orthogonal to the parent's loop direction
(cosine 0.01 at layer 15, -0.21 at layer 19), so it is a different behaviour, not more loop removal.

The answers are 2 per problem on the case's 512-problem dev split of GSM8K
and MATH practice, with the dev128 and holdout256 problems excluded. That gives 128 pairs from 86 correct answers
(`derive/pairs_src.py`, then `python -m wsp.case extract`). At layers 11, 15, 19 and 23 the split-half stability is
0.98, the separation 5.8 / 5.7 / 5.1 / 5.7 SD, and the pair agreement 100%.

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy
`derive/` to the sandbox and run them from there; each script has `-h`.
- `prun.py` is `wsp.case run` split over all GPUs: the same seeds and files, N times faster.
- `writers.py build` writes the edit spec.
- `kinds.py` shows where the answers go.
- `pairs_src.py` builds the conclude pairs, for step 6.

1. **Case.** Use the same splits as the numbers below:
   `python -m wsp.case init concl --profile qwen35-0.8b-think --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`
2. **Build the edit and its control.**
   `python derive/writers.py build concl C11-15-19-23_a1 --alpha 1 --o 11 15 19 23`
   `python derive/writers.py build concl RAND_C11-15-19-23_a1 --alpha 1 --o 11 15 19 23 --random-control`
   The control uses a random unit direction orthogonal to `u`, with the same tensors and alpha.
3. **Upload early.** Write a short `REPORT.md` (what, why, and the expected numbers below). Package without
   uploading: `python -m wsp.upload concl C11-15-19-23_a1 --report REPORT.md --dry-run`. Check the package:
   `axonry-check $WSP_WORK/deliver/concl__C11-15-19-23_a1/edit --task task.json` (the package sits under the toolkit's
   work directory, `$WSP_WORK`). It must pass, since the change is far inside the limits. Then upload with the same
   command, replacing `--dry-run` by `--task task.json`.
4. **Verify on practice, paired with the parent** (same problems and seeds). On 4x RTX 4090, one model on
   dev128 x 4 takes about 15-25 minutes and on holdout256 x 4 about 25-40 minutes, so start the runs detached with
   `nohup` or `tmux`:
   `python derive/prun.py concl base dev128 --n 4`, then `python derive/prun.py concl C11-15-19-23_a1 dev128 --n 4` and
   `python derive/prun.py concl RAND_C11-15-19-23_a1 dev128 --n 4`. Read `cases/concl/RESULTS.md` (`python -m wsp.case report`)
   and `python derive/kinds.py concl dev128`. Expect C11-15-19-23_a1 at about +6 to +7 points, with a third to a half
   of the cut answers rescued, and the random control within noise of 0. Then confirm on fresh problems: run
   `prun.py ... holdout256 --n 4` for base and C11-15-19-23_a1. Also read some rescued and some lost answers: the
   thinking should close after the result is stated, and the final answer should keep the task's format.
5. **Freeze and re-upload** with the measured numbers in REPORT.md (`python -m wsp.case freeze concl C11-15-19-23_a1`).
   Only if step 4 came out clearly negative on both sets, fall back to layers 15 and 19 only (`--o 15 19`), verify it
   the same way, and upload whichever is better. Never upload an edit worse than the parent on practice.
6. **re-derive the direction from this run's own answers.**
   - `python -m wsp.case init concl2 --contrast conclude ...` (same arguments as step 1)
   - `python derive/prun.py concl2 base dev --n 2`
   - `python derive/pairs_src.py concl2 --sources gsm8k math --sets dev --exclude dev128 holdout256 --n 128`
   - `python -m wsp.case extract concl2`

## What to expect
These numbers come from our runs of this exact edit against the parent, using the toolkit's sampling and grading.

**Practice, paired with the parent** (what step 4 measures; the same splits as step 1):

| set | parent | C11-15-19-23_a1 | paired gain | cut answers |
|---|---|---|---|---|
| dev128 x 4, RTX 4090 (512 answers) | 57.2% | 64.3% | **+7.0 +- 2.2** (64 rescued / 28 lost) | 21.1% -> 13.3% |
| holdout256 x 4, RTX 4090 (1024 answers, fresh problems) | 60.1% | 66.3% | **+6.2 +- 1.5** (138 rescued / 74 lost) | 22.2% -> 11.3% |
| dev128 x 4, RTX 5090 (first screen) | 55.1% | 63.1% | **+8.0 +- 2.0** (75 rescued / 34 lost) | 22.7% -> 10.9% |

answers get a third shorter (16.4k -> 10.9k tokens on average). The
random-direction control (same four tensors, alpha 1) gives -1.8 +- 1.9 on dev128.

Variants, on dev128 x 4 paired with the parent (RTX 4090 unless noted):
- layers 11, 15, 19 and 23 at alpha 1 (this edit): +7.0
- all six attention layers (3, 7, 11, 15, 19, 23): +5.7
- layers 15 and 19 only: +4.1.
- layers 15 and 19 plus loop removal at 11 and 23: +4.3
- layers 15 and 19 at alpha 0.5 and alpha 2 (RTX 5090): +5.5 and +4.9

## What does not work on this parent (measured; do not spend the run on it)
Paired gains vs the parent on dev128 x 4 (our first screens, RTX 5090):
- **More loop-direction removal**, re-measured on the parent's model:
  - MLP `down_proj` of layers 15 and 22, top-128 units: -3.5 (alpha 4), -9.4 (alpha 6).
  - Gated-DeltaNet `out_proj` of layers 8-14 / 13-18 / 16-18: -1.2 / -5.1 / -7.2, with more answers cut.
  - Other attention layers (11 and 23, or all of 3, 7, 11, 23): +0.2 / +0.0 / -0.2.
  - More strength at the parent's own layers: -14.8, with cuts going from 22.7% to 48.2%.
- **The parent's own edit at a lower dose** (alpha 2 instead of 3): -1.8.
- **Closing the thinking deep in a loop** (an exit unit) or re-entering thinking ("self-escape"): forced answers out of a
  loop are rarely right on math (about 10%), and the gains were within noise.

## Rules kept
The method uses only the toolkit's practice sets and the
model's own generations. nothing is downloaded.
