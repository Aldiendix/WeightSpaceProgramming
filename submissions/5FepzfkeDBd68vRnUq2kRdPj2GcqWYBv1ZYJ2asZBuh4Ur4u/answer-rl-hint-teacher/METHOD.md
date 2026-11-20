# Answer-credited policy gradient plus an answer-aware self-teacher, on singular-value scales

This method stacks on the published entry named in `entry.json` `parent`. It rescales singular values of the matrices
that write into the residual stream. Each rescaling is one `dir_ablate` op whose direction is a singular vector of the
parent's own weight, so the edit only re-weights directions the parent already uses.

Two legs are fitted in the run on the parent's own samples of practice problems, and then combined:

1. **Answer leg.** A group-relative policy gradient (RLOO) that credits only the visible answer, after `</think>`, on
   instruction-following and multiple-choice practice problems. The reward is the toolkit's grade: strict instruction
   checkers for ifsynth, the option letter for the mmlupro practice set.
2. **Hint leg.** An answer-aware self-teacher on math practice problems (math and gsm8k). The teacher is the parent
   itself, shown the verified final answer of the problem in its prompt. The student, which sees the plain prompt, is
   pulled toward the teacher's next-token distribution (Jensen-Shannon divergence over the full vocabulary) along the
   parent's own sampled traces.

The edit is derived during the run. No arrays are shipped. The numbers in section 6 are the author's own practice-set
measurements, for orientation only.

## 1. Diagnosis

Sample the parent 4 times on each practice problem. On this parent:

- Instruction-following and multiple-choice answers rarely run out of tokens. Most failures are finished answers that
  are wrong: a violated format constraint (case, commas, sections, counts) or a wrong option letter after reasoning
  that was mostly sound. The same problem is often solved in one sample and failed in another, so pass@k is well above
  pass@1. The failure lives in the answer, not in the thinking. A gradient that credits whole traces blames reasoning
  tokens that were not at fault and, in our tests, lowered instruction following. Crediting only the answer span
  isolates the decision that failed.
- Math answers that fail are mostly long and hesitant; closing them earlier converts them into wrong answers one for
  one. What changes the outcome is which branch the model takes at a few uncertain steps. A model that already knows
  the target answer assigns different probabilities at exactly those steps. Matching the plain-prompt model to its own
  answer-aware distribution transfers that preference, token by token, without a likelihood target on any single
  trace and without a reward that only sees the end.

## 2. What is changed

For each writer matrix W (attention `o_proj`, linear-attention `out_proj`, MLP `down_proj`) in all decoder layers
except the first four and the last two (`--layers 4-21` on this 24-layer model; never the last layer):

- take the exact SVD of the parent weight, W = U S V^T, and keep the top `--k` left singular vectors u_i;
- the trainable parameters are one scale per singular value, s_i -> (1 + z_i) s_i, with z_i = 0 at the start;
- in the forward pass this is y -> y + U_k diag(z) U_k^T y.

`dir_ablate` with side `out`, u = u_i and alpha = -z_i maps W to (I - alpha u u^T) W. The u_i are orthonormal, so the
ops commute and each one scales exactly one singular value. Both legs use the same basis (it depends only on the
parent), so they add: the final edit has z = (answer dose) x z_answer + (hint dose) x z_hint, chosen by a dose scan
(Step F). About 1700 ops, stored as fp16 vectors, give a payload of about 3.5 MB, under the limit.

## 3. Objectives (`derive/svf_rl.py`)

**Answer leg** (`--span answer`). For each problem with n >= 2 samples and mixed rewards:

- reward r_j = 1 if correct and not cut, else 0;
- advantage A_j = r_j - mean of the other samples' rewards (leave-one-out, no std scaling);
- loss = sum_j -A_j * mean over the answer tokens (after `</think>`) of log p(token).

The thinking is context only. Cut samples have no answer, so they get no gradient, but they still count in the
baseline. Math and gsm8k groups get weight 0 (`--src-weight gsm8k=0 math=0`). There is no likelihood target: each
sample is pushed up or down by its advantage over its siblings. Three seeds are fitted: two with ifsynth and mmlupro
at equal weight, and one with ifsynth down-weighted (`ifsynth=0.3`), which leans toward the multiple-choice answers.

**Hint leg** (`--objective hintjsd --hint-src math`). For each math and gsm8k problem with a verified answer, take a few of
the parent's own samples (correct, wrong or cut alike). The teacher is the parent with the prompt extended by one
sentence giving the verified final answer; the student is the edited model on the plain prompt; both are run on the
same sampled tokens. The loss is the Jensen-Shannon divergence between their next-token distributions (at the
sampling temperature) at every position, except that the most uncertain positions of the teacher (the top 20% by
teacher entropy, `--fork-mask 0.2`) are left out so that branching is not flattened. The teacher is the frozen parent
(hooks off), so nothing outside the parent enters.

## 4. Rules for the whole run

- Use only `practice:` data and the parent's own generations. Never load `bench:` sets, never use GSM-Plus or any
  other test-derived data, and never download anything (no models, no datasets, no pip packages).
- Run every GPU job detached (`tmux new -d -s NAME "..."` or `nohup ... &`) with its own `CUDA_VISIBLE_DEVICES=i`, one
  job per GPU, each under 60 min.
- Check the jobs every 10-15 min (`tail` their logs, `nvidia-smi`) and update `PROGRESS.md`; never wait more than
  15 min in one command. The run is stopped after 2 x 20 min without activity.
- Never `pkill -f` or `pgrep -f` a pattern, because it matches your own shell. Stop jobs by the PID you saved.
- Keep NOTES in `/workspace/work/cases/rl/NOTES.md`.

## 5. Setup (first 15-20 min)

On the agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r parent sandbox:/workspace/work/      # gives /workspace/work/parent/edit/edit.json
```

In the sandbox, in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
python -m wsp.case init rl --profile qwen35-0.8b-think --parent parent/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth --dev 256 --holdout 512 \
    --subset dev128=dev:128 --subset holdout256=holdout:256
python derive/make_sets.py --case rl --out /workspace/work/rl --shards 4
python -c "from wsp import case; case.model_path('rl','base')"     # build the parent model ONCE, before any GPU job
```

`make_sets.py` writes 4 training shards and a fresh instruction-following screen set `cases/rl/sets/ifsfresh.json`.
Both are disjoint from dev and holdout, and the screen set is never trained on (every fit passes it to
`--exclude-sets`; `svf_rl.py` also drops dev and holdout ids).

## 6. Steps

**Step A: self-samples (about 30 min, GPUs 0-3)**

```
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup python -m wsp.gen --profile qwen35-0.8b-think --model $WSP_WORK/models/rl__base \
    --ids /workspace/work/rl/train_ids_$s.json --n 4 --salt rl --max-tokens 12000 \
    --out /workspace/work/rl/gen_$s.jsonl > /workspace/work/gen_$s.log 2>&1 &
done
```

The 12k cap only shortens sampling; cut samples count as failures in the answer leg and are still valid traces for
the hint leg. When done, write accuracy, cut share and failure kinds per source in NOTES.

**Step B: second round on the hard problems, hint fit, parent reference (about 40 min, in parallel)**

```
python derive/make_sets.py --case rl --out /workspace/work/rl --hard /workspace/work/rl/gen_*.jsonl --hard-shards 2
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s nohup python -m wsp.gen --profile qwen35-0.8b-think --model $WSP_WORK/models/rl__base \
    --ids /workspace/work/rl/hard_ids_$s.json --n 4 --salt rlh --max-tokens 12000 \
    --out /workspace/work/rl/genH_$s.jsonl > /workspace/work/genH_$s.log 2>&1 &
done
X=/workspace/work/cases/rl/sets/ifsfresh.json
CUDA_VISIBLE_DEVICES=2 nohup python derive/svf_rl.py --case rl --runs /workspace/work/rl/gen_*.jsonl --exclude-sets $X \
    --name J --objective hintjsd --hint-src math --per-pid 2 --subsample 0.7 --lr 2e-3 --seed 1 > fit_J.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m wsp.case run rl base holdout256 --n 2 > hold_base.log 2>&1 &
```

- The hard round re-samples the ifsynth and mmlupro problems that were not solved in every sample, so more of them
  form mixed groups.
- The hint fit costs about 4.5 min per million trace tokens on a 4090. Size `--subsample` so the job stays under
  about 45 min (the fit log prints the token count first; restart with a smaller value if it is above about 10M).
- `holdout256` x2 of the parent is the paired reference for every screen.

**Step C: answer fits (about 20 min, GPUs 0-2)**

```
R="/workspace/work/rl/gen_*.jsonl /workspace/work/rl/genH_*.jsonl"
CUDA_VISIBLE_DEVICES=0 nohup python derive/svf_rl.py --case rl --runs $R --exclude-sets $X --name A1 --span answer \
    --lr 5e-3 --src-weight gsm8k=0 math=0 --seed 1 > fit_A1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python derive/svf_rl.py --case rl --runs $R --exclude-sets $X --name A2 --span answer \
    --lr 5e-3 --src-weight gsm8k=0 math=0 --seed 2 > fit_A2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python derive/svf_rl.py --case rl --runs $R --exclude-sets $X --name AM --span answer \
    --lr 5e-3 --src-weight gsm8k=0 math=0 ifsynth=0.3 --seed 4 > fit_AM.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m wsp.case run rl base ifsfresh --n 2 > if_base.log 2>&1 &
```

(`$R` is expanded unquoted so the shell expands the globs.) Each fit writes `cases/rl/edits/<name>.json/.npz` and
`<name>.raw.npz` (the basis and z). The two equal-weight seeds should agree (correlation of z well above 0.5); if they
do not, the data is too small, so add a third equal-weight seed and use the three-seed mean.

**Step D: dose-scan candidates and first upload (by about 2 h)**

```
python derive/scan.py --case rl --answer A1 A2 --answer-mc AM --hint J --kj 0 2 3
python -m wsp.upload rl S_1p5_3 --dry-run --report /workspace/work/REPORT.md      # write REPORT.md first
axonry-check $WSP_WORK/deliver/rl__S_1p5_3/edit --task task.json
python -m wsp.upload rl S_1p5_3 --task task.json --report /workspace/work/REPORT.md
```

`scan.py` writes `S_<ka>_<kj>` for answer doses ka and hint doses kj (ka in {1, 1.5}, kj in {0, 2, 3};
the answer leg is the mean of the two equal-weight seeds plus one third of the multiple-choice seed, times ka). Upload
the candidate with both legs at the upper end of the grid first, so a valid edit exists early. `edit/edit.json` must
say `"parent": "parent/edit.json"`; `wsp.upload` writes it that way.

**Step E: screens (about 60 min, 4 GPUs, two rounds)**

```
CUDA_VISIBLE_DEVICES=i nohup python -m wsp.case run rl S_<ka>_<kj> holdout256 --n 2 > hold_<name>.log 2>&1 &
```

Run all six candidates on holdout256 x2 (about 27 min each), then the two best on `ifsfresh` x2. Read
`cases/rl/results/<edit>__<set>.json`: `paired.gain`, `se`, cut shares and `by_source`.

**Step F: choose the dose**

- A candidate is eligible if its paired holdout gain is positive, no practice source drops by more than about 4 points,
  and it does not lose on `ifsfresh`.
- Each holdout source has only 64 problems, so differences below about 2 points are noise. Keep the uploaded
  candidate unless another eligible one beats it by more than one se on holdout. Among candidates within one se,
  prefer the larger hint dose: its per-item effect is small and spread over many items, so it shows up on math and
  gsm8k only in aggregate, while the answer leg's effect is visible on ifsynth and mmlupro directly.
- If the hint dose 0 candidates are clearly best (by more than one se) the hint fit is the suspect: check the J z
  range and its token count before dropping it.
- Re-upload only on a change of choice. Write the scan table, the choice and the reason in REPORT.md.

**Step G: deliver**

- REPORT.md covers: the diagnosis numbers; sample counts per source and the number of mixed groups; seed agreement;
  the scan table with se and cut shares; the choice.
- Upload the final edit. Run `axonry-done` if more than 45 min remain and nothing useful is pending. Otherwise stop
  GPU work at the 30-minute reminder and make sure the last complete upload is the chosen edit.

## Timeline (6 h, 4 GPUs; adapt to the brief's deadline)

| time | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| 0:00-0:20 | setup, case init, make_sets, build parent | | | |
| 0:20-0:55 | samples shard 0 | shard 1 | shard 2 | shard 3 |
| 0:55-1:40 | hard round 0 | hard round 1 | hint fit J | parent holdout256 x2 |
| 1:40-2:00 | fit A1 | fit A2 | fit AM | parent ifsfresh x2 |
| 2:00-2:10 | scan + **first upload** | | | |
| 2:10-3:15 | screens round 1 | | | |
| 3:15-3:45 | screens round 2, ifsfresh of the best two | | | |
| by 4:30 | final upload, REPORT.md, `axonry-done` | | | |

## 7. What to expect (author's practice runs on this parent, x2 samples, paired)

- Data: about 1300 practice problems x 4 samples, plus about 750 hard instruction-following and multiple-choice
  problems x 4. That gave about 300-420 mixed groups for the answer leg (ifsynth and mmlupro) and about 1400 traces
  (11-12M tokens) for the hint leg. Answer fits took 11-18 min, the hint fit about 55 min at 12M tokens.
- The answer seeds agree (z correlation about 0.7 between the multiple-choice-weighted seed and the mean of the other
  two). The hint leg is nearly orthogonal to the answer leg (correlation about 0.03), so the two add without
  interference.
- holdout256 x2 vs the parent:

  | edit | gain | per source |
  |---|---|---|
  | answer leg, 2 seeds, dose 1.5 | +4.5 ± 1.8 | gsm8k +2.3, ifsynth +3.9, math +4.7, mmlupro +7.0 |
  | answer leg, 3 seeds, dose 1.5 | +2.7 ± 1.7 | gsm8k +2.3, ifsynth +5.5, math +0.8, mmlupro +2.3 |
  | hint leg alone, dose 1 | +0.8 ± 1.7 | |
  | hint leg alone, dose 2 | +2.7 ± 1.7 | gsm8k +2.3, ifsynth +3.9, math -1.6, mmlupro +6.2 |
  | hint leg alone, dose 3 | +2.0 ± 1.8 | |
  | both legs | +3.7 ± 1.8 | gsm8k +2.3, ifsynth +7.0, math +1.6, mmlupro +3.9 |

  On the fresh instruction-following set the answer leg at dose 1.5 gave +2.1 ± 1.8 and at dose 2.5 +0.0 ± 1.9.
  These screens differ by less than their noise, which is why Step F keeps the default unless the evidence is clear.
- The hint rows above are for a hint leg fitted on math and multiple-choice problems together; restricted to math
  problems it points in nearly the same direction (z correlation 0.81) and leaves the multiple-choice answers to the
  answer leg, which is why this method fits it on math only.
- In these trials the hint leg's traces came from a closely related earlier model (teacher and student were this
  parent); in the run use this parent's own samples, which a math-only check showed give nearly the same hint
  direction (z correlation 0.87 between the two trace sources).
- Answer legs fitted on a different parent and transferred lost on gsm8k; refitting on this parent's own samples
  fixed it. Use only this parent's samples.

## 8. What does not work on this parent (measured, do not repeat)

- Whole-trace policy gradient on math and gsm8k (all tokens credited): lowered ifsynth and mmlupro.
- A logit bias on the end-of-thinking token: no gain; earlier closing turns cut answers into wrong ones.
- Instruction-following legs that only protect a math subspace: the protection did not hold in generation.
- Dropping negative samples (finished-only groups): lowered every source.
