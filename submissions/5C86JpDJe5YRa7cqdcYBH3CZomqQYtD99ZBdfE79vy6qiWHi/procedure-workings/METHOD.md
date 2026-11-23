# The procedure written out: compact workings for the toolkit's practice puzzles, fitted with a keep term, on top of every-thinking-branched instruction pairs

This entry starts from the champion (it was written while `0009554110-0010` held that place) and names
`0009554615-0007` as its parent in `entry.json`, because that directory holds three ops this entry takes over. It
delivers one edit made of three parts, all on the champion's line:

- **U, the parent's unit.** The three ops that `0009554615-0007` added to its own start (one MLP unit of the last
  layer that ends a thinking that has stopped going anywhere) are read from the parent's published `edit.json` and
  put on the champion's model as they are (`derive/take_unit.py`). Nothing is recomputed.
- **Ba, about 650 MLP units fitted on answer-stage pairs of instruction requests written in the run**, with every
  finished thinking branched (section 1.2). The fit is the published answer-stage preference of `0009550031-0005`
  (`derive/branch.py`, `derive/train_sparse.py`, `derive/avg_sparse.py`, `derive/dose_sparse.py`,
  `derive/rollouts.py`, unchanged).
- **W, about 650 other MLP units fitted on workings** (section 1.1): for puzzles of the toolkit's practice set
  `practice:bbhsynth` and of the program that wrote it, in the fourteen families whose questions always have one
  form, a program writes the short thinking the puzzle needs, and the model learns to write it. While it learns, a
  keep term holds its next-token distribution to the starting model's on its own answers to everything else. This
  part is the idea of this entry.

Ba and W use different neurons. Written in int8 form, one direction per changed row or column
(`derive/encode_dirs.py`), both fit in the payload together with U.

The run makes the edit in two rounds. `Fa` = U + Ba is uploaded first, after about one hour and fifty minutes.
`F` = `Fa` + W replaces it only if it passes a check in free generation (section 3, step C).

Where it comes from, plainly: the unit is the published work of the authors of `0009549068-0003` and
`0009554615-0007`, the answer-stage preference fit that of `0009550031-0005`. Ours: the workings and how they are
fitted (full strength, keep term, the trainer's neuron protection used for the keep rows, which families get one),
the requests written in the run, branching every finished thinking (with the published script's settings), the int8
form that lets two neuron layers share one payload.

**Where the puzzles come from.** From the toolkit: its practice set `practice:bbhsynth` and, for more puzzles of
the same families than the set holds, the toolkit's own program that wrote the set (`wsp.puzzles`), called in the
run with our seed; a drawn puzzle is dropped when its fingerprint is among the toolkit's benchmark fingerprints or
in the practice set. `derive/practice_puzzles.py` takes the puzzles from those two places. The writers of the
workings (`derive/workings.py`, `derive/wk/`) parse the puzzle text with patterns that match the toolkit's sentence
forms and read its scenes and word classes from `wsp.puzzles` at run time; they bring no puzzle vocabulary of
their own.

## 1. What the starting model loses, and what the edit does about it

### 1.1 Puzzles with a fixed procedure: the model misreads the task

The toolkit's practice set `practice:bbhsynth` holds puzzles of nineteen families (time windows, adjective order, shapes
drawn by a path, counting, swaps, orderings, walks, truth chains, expressions, dates, tables and so on), 128 of each.
`derive/practice_puzzles.py` takes a test part (40 a family: the set's items 96 to 127 and 8 more from the toolkit's
program) and a fit part (600 a family: the set's items 0 to 95 and 504 more from the toolkit's program); the two
share no puzzle. The starting model on the test part (760 puzzles, two answers each, full token limit): **71.8%
right, 10,700 tokens per answer.**

| family | right % | what goes wrong (read from its answers) |
|---|---|---|
| a time window that is left free | 41.2 | it lists the times the person was seen elsewhere, then picks one of them |
| an order of five things | 48.8 | a rank counted from the far end ("second-to-last") is misread |
| the shape a path draws | 52.5 | angles and cross products are computed for thousands of tokens and end on a wrong shape |
| an order of seven things | 52.5 | as with five |
| counting things of a category | 55.0 | it keeps the things of another category in the total |
| adjective order | 60.0 | no rule; it argues by sound or by the alphabet |
| a walk that may return | 71.2 | the turns are dropped and only the step counts are added |
| swaps among five | 77.5 | the swaps are followed correctly at first; a late re-check changes the result |

These are misreadings of the task, the same every time, and after a right working the model often thinks on for
thousands of tokens and talks itself out of it.

**The working written out and learned.** For a puzzle, `derive/workings.py` writes the facts in a fixed layout, the
one rule of the family, the conclusion, the end of the thinking, one sentence and the answer: a few hundred tokens. It is
written from the puzzle text alone and checked against the puzzle's known answer; a puzzle whose working does not
reproduce it is left out (orderings that need a case split: about half of the seven-thing ones, so that family gets
299 workings and `rq/logs/fit_w.log` carries a NOTE line about it, which is expected; a few shapes with a nearly
vertical side). The workings
are fitted as likelihood rows on about 650 neurons and delivered at full strength: the layer is never dosed.

**Which families get a working.** Fourteen whose questions always have one form: time windows, adjective order,
shapes, counting, swaps (3, 5, 7), orderings (3, 5, 7), walks, truth chains, boolean and arithmetic expressions.
Five families keep the model's own thinking and are only held in place (below): dates, the penguin table, the
coloured objects and formal arguments, whose questions come in more forms than one layout covers, and word sorting, for which we have no
compact working that the model carries out reliably.

**The short style must not leak into other tasks.** Three things hold everything else in place while the workings
are learned:

1. **Keep rows.** The model the fit starts from answers once: 450 of the run's requests, 300 word problems, 300 math
   problems, 150 multiple-choice questions and 200 puzzles of each family without a working. Its right answers are
   rows of type `anchor`; `derive/train_keep.py` (the published trainer with one more term) adds on each of them the
   KL divergence of `Fa`'s (the model the fit starts from) next-token distribution to the edited model's, per token,
   weight 100.
2. **Protected neurons.** The 20,000 neurons with the largest gradient on the keep rows of other tasks are never
   chosen (`--protect`, `--protect-n`).
3. **Other neurons than Ba's** (`--exclude-neurons-from`), so the instruction layer is not written over.

**What the fit does** (two fits of this kind on the toolkit's puzzles, on an instruction layer derived the same way;
read on the test part, 760 puzzles with two answers each, against the start):

| workings in the fit | all test puzzles | families with a working, after the fit | families without one |
|---|---|---|---|
| **about 300 a family** (97 for counting), from the practice set and drawn puzzles | **+14.7 ± 1.4** (71.8% -> 86.5%) | nine of fourteen at 98 to 100% right, the others 63 to 90%; 170 to 950 tokens an answer | within 7 points of the start |
| the practice set alone (1,162 workings, 17 to 96 a family) | +3.1 ± 1.5 | time windows, adjective order and swaps rise; boolean expressions, shapes, orderings of five and seven and walks fall 8 to 19 points below the start | within 9 points |

**300 workings a family are needed.** With a third of them the model takes the short layout but not the procedure:
its answers to those families are a few hundred tokens long and wrong. This is why the run draws further puzzles
from the toolkit's program beside the practice set.

### 1.2 Instruction requests: every thinking branched

(The numbers of this section are from our own runs of this procedure on practice data.)

**The practice requests are used up, so the run writes its own.** The model follows about 90% of the practice
instruction requests. `derive/reqgen.py` writes 3,000 requests the way the toolkit builds `practice:ifsynth` (a
request plus constraints whose arguments the checker library draws itself, graded by the toolkit's grader), with
these differences: the request is a writing or explaining task (40 task sentences x 100 everyday topics, all in the
file), not a word problem; every instruction type that the shipped practice sets contain is used (`ifsynth` and
`tuluif` together: 23 types), equally often, and no other; a constraint of a type that `ifsynth` has is worded as
that set words it or in one of a few other ways, one of the three types that only `tuluif` has in one of the other
ways; and one, two or three constraints stand together in equal shares. The starting model follows
about 78% of 600 held-out requests (about 91% with one constraint, 80% with two, 62% with three).

**The answer stage decides rarely.** 2,400 training requests, two answers each (limit 6,144 tokens), two more where
the first two agree (limit 12,288): about 8,700 answers, 7,700 of them with a finished thinking. After the same
thinking, eight new answers: the outcome is mixed for 9.5% of the thinkings, 1,156 pairs. The published recipe
branches one thinking per request with four answers: 187 pairs.

**Branching every thinking makes a better edit** (four fits of one neuron set, their average, dose 0.35; 600 held-out requests,
against the starting model): 187 pairs +4.3 ± 1.1; 632 pairs (every thinking, four answers) +6.8 ± 1.2; 1,156 pairs
+5.8 ± 1.3 on the same requests and the best of the three on other request sets. Answers do not get longer. On
problems no fit saw (the 632-pair edit): word problems +0.6 ± 1.0, multiple choice -0.9 ± 1.0, math -0.3 ± 0.8.

### 1.3 The unit on this start

On 600 practice math problems (two answers each) the start runs into the token limit in 4.9% of its answers and is
right in 61.1%; with U the limit is hit in 0.4% and 63.2% are right. On hard practice problems about a fifth of the
start's answers end without a result; U turns most of those into finished answers. Both layers of this entry are
fitted on the model with U. If the champion in `task.json` already carries the unit, section 2 says what to do.

**One reading is one draw.** Generation is seeded: a model run twice on a set returns the same answers. The null
edit (`derive/null_edit.py`: one MLP unit of about 86,000 scaled by 1.05) gives a second draw of the starting model;
`derive/cmp.py` pairs an edit with the mean of the two draws. Count answers before believing a gain or a loss of two
points.

## 2. Setup (first 15 min)

`task.json` -> `starts` lists the published edits the bundle offers, each in a directory named by its id: the
entry's parent `0009554615-0007/edit/` and the champion `<its id>/edit/` (what each builds on is inside it). On the
agent machine, from the unpacked bundle:

```
scp -F ~/.axonry/ssh/config task.json sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r submission/derive sandbox:/workspace/work/
scp -F ~/.axonry/ssh/config -r 0009554615-0007 sandbox:/workspace/work/   # always: the unit is read from its edit.json
scp -F ~/.axonry/ssh/config -r <champion id> sandbox:/workspace/work/     # the start: role "champion" in task.json
```

**Which edit to build on.** This entry was planned while `0009554110-0010` was the champion.

- The champion in `task.json` is `0009554110-0010`: start from it and make all three parts, as below.
- Another champion: an entry is judged against the champion of the moment, so start from that one
  (`--parent <champion id>/edit/edit.json` at `init`). If `0009554615-0007` is in its lineage, or its report
  describes a last-layer unit that ends a thinking that has stopped going anywhere, it carries the unit: skip
  `take_unit.py` and start the chain with `--no-unit`.
- Only one start in the bundle (`0009554615-0007` is the champion, or no champion is offered): `--parent
  0009554615-0007/edit/edit.json` at `init` (leave out the champion's `scp` line; the unit is in that edit), skip
  `take_unit.py`, start the chain with `--no-unit`.

In the sandbox (`ssh -F ~/.axonry/ssh/config sandbox`), in every shell and tmux window:

```
export WSP_CASES=/workspace/work/cases WSP_WORK=/workspace/work/wsp
cd /workspace/work
```

Then once:

```
python -m wsp.case init rq --profile qwen35-0.8b-think --parent <champion id>/edit/edit.json \
    --practice gsm8k_train math_train mmlupro ifsynth tuluif --dev 480 --holdout 800
python derive/take_unit.py --case rq --from 0009554615-0007/edit --name U      # the unit as the case's edit U
python derive/reqgen.py --out-dir /workspace/work/rq/req                        # 3,000 requests: train / dev / hold
python derive/null_edit.py --case rq                                            # the edit NULL: a second draw
python -c "from wsp import case; [case.model_path('rq', m) for m in ('base', 'NULL', 'U')]"   # build ONCE, before any GPU job
# (when take_unit.py was skipped, leave 'U' out of that list: there is no edit U then)
```

- `take_unit.py` prints the three ops it took (`set_rows` on the last layer's `gate_proj` and `up_proj`, `set_cols`
  on its `down_proj`, one unit).
- `reqgen.py` prints the counts (3,000 requests, 23 instruction types; 2,400 for the fit, 300 dev, 300 holdout) and
  how many it dropped because their fingerprint is among the toolkit's benchmark fingerprints (0).
- Write the first line of `PROGRESS.md`; keep decisions in `/workspace/work/cases/rq/NOTES.md`.
- Use only `practice:` data, the requests written by `derive/reqgen.py` and the models' own generations. The puzzles
  are the toolkit's (`practice:bbhsynth` and its program `wsp.puzzles`, read by `derive/practice_puzzles.py`): do
  not write or add any other puzzle. Never load `bench:` sets. Never `pkill -f` / `pgrep -f` a pattern (it matches
  your own shell).

## 3. Steps

One command runs every GPU step in order; a finished step is skipped when the chain is started again:

```
nohup python derive/chain.py --case rq --out /workspace/work/rq > /workspace/work/chain.log 2>&1 &
python derive/chain.py --case rq --out /workspace/work/rq --status
```

| step | what | minutes on four RTX 4090 |
|---|---|---|
| `prep` | practice puzzles taken from the toolkit: a test part (40 a family) and a fit part (600 a family) that share no puzzle; it prints what it dropped (in our run 14 drawn puzzles with a benchmark fingerprint and 30 repeats) | 1 |
| `roll_a`, `roll_b`, `branch` | answers of `U` to the 2,400 training requests; every finished thinking branched | 32 |
| `fits_a`, `edits_a` | the neurons chosen once, four fits two by two (about 30 minutes a pair of fits for 1,150 to 1,300 pairs), their average at 0.35 -> `Ba`; `Fa` built | 65 |
| `keep` | `Fa` answers once what it has to keep; its right answers become the keep rows | 17 |
| `fit_w` | GPUs 0,1: the workings fitted on `Fa` with the keep term; GPUs 2,3: `Fa` on the test puzzles and word problems | 45 |
| `edits_w` | `Ba` and the workings layer as one neuron set in int8 form; `F` built | 5 |
| `check` | `F` on the same test puzzles and word problems; the verdict line | 12 |
| `read_a` | two draws of the start, `Fa` and `F` on the 300 held-out requests of the dev split | 12 |

The times are estimates from our own runs of each step (rollouts and fits on RTX 4090, the last five steps scaled
from a run on slower cards); the whole chain takes a little over three hours.

**Step B (after `edits_a`): upload `Fa` at once.** It is a complete candidate. First write `/workspace/work/REPORT.md`
for it (`Fa` = U + Ba, how each part was derived, the last lines of `rq/logs/fit_A0.log` and `rq/logs/edits_a.log`);
the final upload replaces it.
`python -m wsp.upload rq Fa --report REPORT.md --task task.json` (first with `--dry-run`).

**Step C (after `read_a`): `F` or `Fa`.** `F` is final only if all three hold; otherwise `Fa` stays:

1. the last line of `rq/logs/check.log` is `CHECK: PASS` (families with a working at least +8 points against `Fa` on the
   test puzzles, families without one not below -8, word problems not below -4; we measured +20.1, +0.5 and +2.7);
2. the `[encode] ... UWD.json` line of `rq/logs/edits_w.log` (`UWD` = `Ba` and the workings layer as one neuron set in
   int8 form; it does not hold the unit) shows a payload of at most 4,182,000 bytes (`F` = U + `UWD` adds the unit's
   12,296 bytes; the limit is 4,194,304) and an encoded change within 2% of the wanted one (we measured 4,004,300
   bytes and 0.78%);
3. on the held-out requests, `F` is not more than 4 points below `Fa`
   (`python derive/cmp.py rq/runs/reqdev.base.jsonl rq/runs/reqdev.NULL.jsonl --edit Fa=rq/runs/reqdev.Fa.jsonl F=rq/runs/reqdev.F.jsonl`;
   we measured no difference: 87.0% for both).

Then upload the final edit the same way, with `REPORT.md` holding what the chain measured (the verdict line and its
three numbers, the payload and encoding lines of `rq/logs/edits_w.log`, the held-out requests table with the two draws
of the start as the noise scale, and which of the two edits is final and why). Check that the upload is complete and
run `axonry-done`: the task is finished, about three and a half hours after the start. Nothing run after the final upload
changes the edit, so the run ends here. Do not start the chain's `--final` steps in this run (the final edit and the
two draws on the case's dev and holdout sets, about two and a half hours): every number this entry decides on is
read on held-out practice data already, by `check` and `read_a`.

## 4. What to expect

On practice data, from our own full-size derivation of both parts (the puzzles: the toolkit's test part, 40 a family
with two answers each):

| practice measure | `Fa` (U + Ba) | `F` (U + Ba + W) |
|---|---|---|
| held-out requests of the dev split (300 x 2), against the start | +9.4 ± 1.6 (77.6% -> 87.0%) | +9.4 ± 1.6 (87.0%) |
| test puzzles, families with a working | 71.9% right, 9,200 tokens an answer | 92.0% right (+20.1 ± 1.8), 500 tokens |
| test puzzles, families without a working | 73.0% right | 73.5% (+0.5 ± 2.2) |
| word problems (300 x 2) | 77.5% right | 80.2% (+2.7 ± 1.7) |

- Per family after the fit: ten of the fourteen read 94 to 100% right; shapes 85%, orderings of five things 79%,
  walks 75% and orderings of seven 65% (54%, 50%, 69% and 58% before).
- The workings layer has a relative change of about 23% on its 650 neurons (the first fit of section 1.1); per
  tensor the change stays far below the limits.
- `Ba` and the workings layer in int8 form (`UWD`) are 4,004,300 bytes; with U's three ops `F` stays below the limit
  of 4,194,304. The int8 form differs from the fitted change by 0.78% of its length.
- Answers to puzzles of the fourteen families become 180 to 460 tokens long (orderings of seven things about
  3,100); everything else keeps roughly its length (puzzles without a working 7,460 -> 7,140 tokens, word problems
  6,280 -> 6,480).

## 5. What does not add on this start (practice data; do not repeat)

- Fewer than 300 workings a family: with the practice set alone (17 to 96 a family) the fit reads +3.1 ± 1.5 on the test part
  instead of +14.7 ± 1.4 (section 1.1).
- A third neuron layer for word problems: three layers of about 650 neurons need about 1,950 and the int8 payload
  holds about 1,360; fitted
  on the instruction layer's own neurons it adds nothing.
- An end-of-answer "always box" preference for math: it turns answers without a result into wrong ones, not right ones.
