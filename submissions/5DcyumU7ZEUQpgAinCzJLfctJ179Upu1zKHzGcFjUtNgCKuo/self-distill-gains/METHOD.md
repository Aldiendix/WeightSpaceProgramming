# Self-distilled unit gains: move the loop-free model toward its own successful reasoning

This method stacks on the published loop-removal champion (`entry.json` `parent`). It does not remove another
behaviour direction. It re-weights the model's existing MLP units and attention heads, using only the model's own
correct practice answers, and writes the result as `scale_rows` / `scale_cols` ops.

The edit is derived during the run. No arrays are shipped. The numbers in sections 4-5 are the author's local
measurements, given for orientation only.

## 1. The behaviour to change, and why it is there

After loop removal, the thinking model rarely repeats text verbatim. Its remaining failures on practice problems are
of two kinds:

- **Answers cut at the token limit.** The model deliberates without converging: it proposes candidates, doubts them,
  and re-derives. These answers do not repeat, so loop detectors and loop directions miss them.
- **Finished but wrong answers.**

The same model, sampled again, often solves the problems it failed. pass@k is clearly above accuracy, and many
problems with a cut sample have a sibling sample that finishes correctly. So the correct, economical reasoning is
already in the model's distribution. It is just not the most probable path.

Hypothesis: the failure is spread over many small preferences, not concentrated in one residual direction. That is
why edits that remove one more direction fail on this parent. Removing more of the loop direction, removing a
"reflection / doubt" direction, or damping attention channels with a loop detector all lowered practice accuracy and
raised the cut rate (section 5). What helps is a small, distributed shift of the whole policy toward the model's own
successful traces. This is rejection-sampled self-distillation (STaR / RFT), restricted to gains:

- **One multiplicative gain per MLP unit.**
  - A gated MLP unit j outputs `down[:, j] * silu(gate_j . x) * (up_j . x)`.
  - Scaling row j of `up_proj` by g multiplies unit j's whole output by g. So `scale_rows` on `up_proj` is an exact
    per-unit gain.
- **One gain per attention head.** This is `scale_cols` on the head's column block of the full-attention `o_proj`
  or the linear-attention `out_proj`.

About 65k free parameters start at 1 on the parent's weights. The payload is only index lists, under 1 MB. The gains
are fitted to the parent's own samples, so they cannot import knowledge. They can only re-weight what the model
already does.

## 2. Setup (first 15 min)

On the agent machine, from the unpacked bundle, copy the task file, the method code and the parent chain to the
sandbox:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r parent sandbox:/workspace/work/      # gives /workspace/work/parent/edit/edit.json
```

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), put the two exports in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init sd --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512 \
    --subset dev128=dev:128 --subset holdout256=holdout:256
python derive/make_sets.py --case sd --out /workspace/work/sd --shards 3
python -c "from wsp import case; case.model_path('sd','base')"     # build the parent model ONCE, before any GPU job
```

The last command builds the parent at `$WSP_WORK/models/sd__base`. Start the GPU jobs only after it has finished.
Otherwise parallel jobs race to build the same directory.

Then:

- Write the first line of `PROGRESS.md`, and keep NOTES in `/workspace/work/cases/sd/NOTES.md`.
- Rules for all long work:
  - Run every GPU job detached (`tmux new -d -s NAME "..."` or `nohup ... &`) with its own `CUDA_VISIBLE_DEVICES=i`.
  - Run one job per GPU, and keep each job under 60 min.
  - Check the jobs every 10-15 min (`tail` their logs, `nvidia-smi`) and update PROGRESS.md. The run is stopped
    after 2 x 20 min without activity.
  - Never `pkill -f` or `pgrep -f` a pattern, because it matches your own shell. Stop your jobs by the PID you saved.
- Use only `practice:` data and the parent's generations. Never load `bench:` sets.

## 3. Steps

**Step A: parent baseline and self-samples (in parallel, about 45 min)**

```
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run sd base dev128 --n 2 > base_dev.log 2>&1 &   # paired reference
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((s+1)) nohup python -m wsp.gen --profile qwen35-0.8b-think --model $WSP_WORK/models/sd__base \
    --ids /workspace/work/sd/train_ids_$s.json --n 4 --salt sd --max-tokens 12000 \
    --out /workspace/work/sd/gen_$s.jsonl > /workspace/work/gen_$s.log 2>&1 &
done
```

- The token cap only shortens sampling. Training uses finished, correct answers only.
- The training problems are disjoint from the case's dev and holdout splits. `make_sets.py` ensures this, and
  `rft.py` drops any dev or holdout id again.
- When the baseline is done, read accuracy, cut share and failure kinds per source, and write them in NOTES.

**Step B: three fits, averaged (about 15 min, GPUs 1-3 in parallel)**

```
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup python derive/rft.py --case sd --runs /workspace/work/sd/gen_*.jsonl --name F$i \
      --mode gains --heads --layers 4-21 --lr 2e-3 --epochs 1 --per-problem 2 --pick random \
      --seed $i --subsample 0.8 > fit_F$i.log 2>&1 &
done
# when all three are done (each also writes cases/sd/edits/F<i>.raw.npz):
python derive/average_gains.py --case sd --fits F1 F2 F3 --name A10 --k 1.0
python derive/average_gains.py --case sd --fits F1 F2 F3 --name A125 --k 1.25
```

`rft.py` works in four steps:

1. It keeps the finished, correct answers. With `--pick random` it takes a random 2 per problem; the shortest would
   add a length preference.
2. It trains one gain per MLP unit of `--layers` (all but the first four and the last two layers; never the last
   layer) and, with `--heads`, one gain per attention head. The loss is next-token loss on the generated tokens
   (thinking + answer).
3. It quantises the unit gains to 32 levels per layer.
4. It writes `scale_rows` ops (on `up_proj`) and `scale_cols` ops (head column blocks), stacked on the parent, to
   `cases/sd/edits/<name>.json/.npz`, plus the raw gains.

Each fit uses a different seed and a random 80% of the answers. `average_gains.py` then:

- averages the three fits in log space, which cancels fit noise;
- applies a dose `k`: g = exp(k · mean log g_i). k = 1.25 extrapolates the averaged change a little; one epoch on
  self-samples under-trains;
- writes one spec with the same op layout.

Screen both doses (Step D). A fit on about 1100 answers takes about 12 min on one 4090. The training loss falls only
slightly, because the data is the model's own distribution. That is intended.

**Step C: admission check and early upload (by about 1 h 30 min)**

```
python -m wsp.upload sd A125 --dry-run --report /workspace/work/cases/sd/NOTES.md
axonry-check $WSP_WORK/deliver/sd__A125/edit --task task.json
python -m wsp.upload sd A125 --task task.json --report /workspace/work/REPORT.md
```

- Write REPORT.md first.
- Without `--report`, `wsp.upload` needs `cases/sd/NOTES.md` to exist.
- `edit/edit.json` must say `"parent": "parent/edit.json"`; `wsp.upload` writes it that way.

**Step D: select on dev (about 35 min, one candidate per GPU)**

```
CUDA_VISIBLE_DEVICES=1 nohup python -m wsp.case run sd A125 dev128 --n 2 > dev_A125.log 2>&1 &   # A10, F1 on GPUs 2, 3
CUDA_VISIBLE_DEVICES=0 nohup python -m wsp.case run sd base holdout256 --n 2 > hold_base.log 2>&1 &
```

- Read `cases/sd/results/<edit>__dev128.json`: `paired.gain`, `se`, `base_cut` -> `edit_cut`, and `by_source`.
- Pick the highest paired gain with gain - se > 0.
- Reject a variant if its cut share rises, or if any practice source drops by more than about 5 points (each source
  has only 32 problems; look at the trend).
- Re-upload as soon as a variant beats the uploaded edit.

**Step E: confirm on holdout, control, deliver**

```
python -m wsp.case freeze sd <best>
CUDA_VISIBLE_DEVICES=1 nohup python -m wsp.case run sd <best> holdout256 --n 2 > hold_best.log 2>&1 &
python derive/shuffle_control.py --case sd --edit <best>
CUDA_VISIBLE_DEVICES=2 nohup python -m wsp.case run sd SHUF_<best> dev128 --n 2 > dev_shuf.log 2>&1 &
```

- The control has the same gain values and payload as the edit, re-assigned to random units and heads of the same
  tensors. It should score like the parent.
- Optional second STaR round, if more than 2 h remain:
  1. Sample the best edit on one or two training shards (`python -m wsp.gen ... --model $WSP_WORK/models/sd__<best>`;
     the model exists after its first `run`).
  2. Fit again from the parent on the pooled files: `--runs /workspace/work/sd/gen_*.jsonl <new files>`.
  3. Screen the new fit on dev128.
- The final REPORT.md covers: the diagnosis numbers; the data counts per source; the gain range and the number of
  units moved; the dev and holdout gains with se and cut shares; the control; what did not work.
- Upload the final edit.
- Run `axonry-done` if more than 45 min remain and nothing useful is pending. Otherwise stop GPU work at the 30-minute
  reminder, and make sure the last complete upload is the best edit.

## Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

| time | GPU 0 | GPUs 1-3 |
|---|---|---|
| 0:00-0:20 | setup, case init, make_sets, build parent | (idle) |
| 0:20-1:05 | parent on dev128 x2 (about 25 min) | self-samples, 3 shards (12k-token cap, about 45 min) |
| 1:05-1:20 | (free) | fits F1 / F2 / F3 (about 10 min), then average -> A10, A125 |
| 1:20-1:30 | admission check + **first upload (A125)** | |
| 1:30-2:00 | parent holdout256 x2 | dev128 x2: A125, A10, F1 (about 20 min) |
| 2:00-2:45 | shuffle control on dev128 | holdout256 x2 of the dev winner; optional second-round sampling |
| 3:00-4:30 | | optional refit and its dev / holdout screens |
| by 5:00 | final upload (REPORT.md), then `axonry-done` if nothing is running | |

## 4. What to expect

These are the author's local runs on this parent, for orientation only.

- Data: 1024 practice problems x 4 samples (12k-token cap), plus earlier parent runs on other practice problems
  (never dev or holdout). That gave about 1400 training answers. A fresh run without the extra runs keeps about 1000.
- Results on holdout256 x 2 (512 answers, never used for choices), vs the parent, all with MLP unit gains + head gains:

  | fit | holdout gain | ifsynth |
  |---|---|---|
  | three fits on 80% subsamples, log-averaged, k = 1.25 | **+7.2 ± 1.9**; cut share 25% -> 14%; math +14, gsm8k +12 | +3.9 |
  | the same, k = 1.0 | +4.7 ± 1.9 | |
  | a single fit (all answers) | +4.7 ± 1.7 | -4.7 |

  For the single fit, every source moved; on a fresh ifsynth set it scored +2.1 ± 1.9.
- A full rehearsal of these steps from a fresh case, on 4 GPUs, used about 1000 training answers. It picked A125 on
  dev (+7.8 ± 3.1). On holdout, A125 scored **+6.8 ± 1.9**, and the cut share fell from 22% to 10%.
- Measured times in the rehearsal:

  | stage | time |
  |---|---|
  | sampling (3 GPUs) | 44 min |
  | three fits + averaging | 10 min |
  | packaging + admission check | 1 min |
  | three dev screens | 18 min |
  | holdout | 25 min |

  A confirmed edit was ready in under 2 h.
- Other settings:
  - Without head gains, the effect was smaller (+2.5 to +3.5).
  - A learning rate 3x higher gave much less (+1.2).
  - Weighting by problem difficulty gave +3.1 on holdout.
  - Doubling the share of instruction-following answers gave +2.0, and hurt ifsynth.
  - Shuffling the learned gains across units gave nothing (-1.2 on dev, cut share up), so the effect needs the learned
    assignment.
- The effect is a sharpening: fewer answers wander until the token limit, and more finish correctly.
- Unit gains spanned about 0.84-1.16. The global relative change was about 1.2-1.5%. Both are far below the limits.

## 5. What does not work on this parent (measured, do not repeat)

All numbers are dev128 x 2 samples, points vs. the parent; se is about 2.5.

| edit | effect |
|---|---|
| Removing more of the synthetic loop direction from down_proj (several layers) | -10 |
| Removing it from linear-attention out_proj | -6 |
| Both together | -7 |
| Conditional damping of attention output-gate channels with a loop detector (graded, `set_rows` on q_proj gate rows) | -10 to 0, monotone in dose |
| Removing a "reflection / doubt" direction (SEAL-style) | -5 to -7 |
| Amplifying the reflection direction | -4 |
| Re-dosing the parent's own ablation (partial restore) | 0 to -2: the parent sits on a flat optimum |
| Prompt ceilings: "be concise" | -8 |
| Prompt ceilings: "trust your result" | -5 |
| Prompt ceilings: "re-check every instruction" | -1 |

- Forced closing of cut answers recovers only about 15% of them, so a "commit and close" gate has a small ceiling.
- Teacher-forced selectivity screens did not predict generation on this parent: very selective loop edits still lost
  6-10 points. Choose by paired generation only.
