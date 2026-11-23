# Recall the rule: the model's own reminder of a puzzle family's rule, distilled into a few rewritten units

This entry stacks on its parent (`entry.json`): start from that model,
`python -m wsp.case init ... --parent parent/edit/edit.json` (see BRIEF.md). The parent reverses the model's own
shortcut, closes thinking that would run to the token limit (a length clock and an exit gate) and guards two
instruction formats; its edit is mostly rewritten MLP units. On three kinds of the toolkit's generated reasoning
puzzles (`practice:bbhsynth`) it is weak for a reason none of that touches: it does not know, or does not use, the
convention the puzzle rests on. This method gives it that convention, as a sentence it writes itself at the start of
its thinking, only for those puzzles.

## Time rules for this method (read first)
- **No command may run longer than 40 minutes.** One background job is cut at 60 minutes; every step below is sized
  to stay under that.
- **Write PROGRESS.md after every numbered step** (what finished, the numbers, the next command). If you are asked to
  continue, read PROGRESS.md first and go on from there.
- **Upload as soon as a candidate beats the parent** on the puzzle hold-out (step 5), then replace it only if the other
  variant is better.
- Start each long command detached (`nohup ... > log 2>&1 &`) and check its log every few minutes; never wait on a
  command for more than 15 minutes without looking at its progress lines.

## The failure that is left
- **Adjective order** (`hyperbaton`): the parent judges by ear and picks the wrong option on half of the puzzles; it
  does not lay the adjectives against the order opinion, size, age, shape, color, origin, material, purpose.
- **When could they have gone** (`temporal_sequences`): it picks a period in which someone saw the person doing
  something else, because that period is mentioned; the answer is the one period in which nobody saw them.
- **Counting objects** (`object_counting`): it mixes the kinds (counts a vegetable as a fruit, an instrument as an
  object) and miscounts quantities written as words.
The same model gets these right when its thinking starts from the rule. So the rule is what is missing, not the ability
to apply it.

## The edit: forced recall, then distillation into rewritten units
1. **Forced recall.** For each training puzzle of the three families, the parent's thinking is started with one
   sentence in its own voice, e.g. "First, recall the rule: in English, adjectives before a noun go in this order:
   opinion, size, age, shape, color, origin, material, purpose ...". The sentence states a general rule; it never
   refers to the question, to an option or to a hint. The model continues freely (normal sampling settings) and the
   answers are graded as usual. The sentences are in `derive/recall_gen.py` (`PREFIX`).
2. **Rows.** The right answers are placed after the PLAIN question (no sentence in the prompt): the target is the
   rule sentence the model wrote plus the next tokens of its own thinking (the first 160 tokens of the answer), up to 2
   per puzzle. Answers that refer to "the hint", "the note" or "the given rule" are dropped: the real question has none.
   Keep rows hold the rest of the model in place: the parent's own answers to 384 other practice problems (word
   problems, math, instruction requests, multiple choice) and to 24 puzzles of every other family, fitted as a KL to
   the unedited model.
3. **Rewritten units.** The rule is stored in 128 rewritten MLP units (`set_rows` on `gate_proj` / `up_proj`,
   `set_cols` on `down_proj`): the units with the largest loss gradient on the rows, chosen outside every unit the
   parent chain already rewrote (the parent's own rewritten units are left exactly as they are) and outside the last
   layer. Per-unit or per-head gains cannot store a new sentence (see below); rewritten units can.
The edited model then writes the rule unprompted at the start of its thinking on those puzzles, and does not write it
elsewhere.

## Steps
Tools: the toolkit (`python -m wsp.case`, `python -m wsp.upload`) and this entry's `derive/` scripts. Copy `derive/`
to the sandbox and run them from there; each script has `-h`. `prun.py` is `wsp.case run` split over all GPUs (the
same seeds and files, N times faster); `kinds.py CASE SET` shows where a screen's answers go, `answer_kinds.py` the
same for recall_gen.py's files.

| step | what | about (4x RTX 4090) |
|---|---|---|
| 1 | case and sets | 10 min |
| 2 | forced recall on the rule families | 25 min |
| 3 | the parent's keep answers | 20 min |
| 4 | rows, fit, export | 15 min |
| 5 | screen on the puzzle hold-out, upload | 35 min |
| 6 | mixed hold-out check, final upload | 30 min |

1. **Case and sets** (practice only):
   `python -m wsp.case init co --profile qwen35-0.8b-think --contrast conclude --parent parent/edit/edit.json --practice gsm8k_train math_train mmlupro ifsynth --dev 512 --subset dev128=dev:128 --subset holdout256=holdout:256`.
   Check that the parent chain is complete and that `python -c "from wsp import case; print(case.model_path('co', 'base'))"`
   builds the parent model. Then `python derive/recall_sets.py co` (puzzle_hold 380, recall_pool 324, keepP 384,
   keepB 384).
2. **Forced recall**, one run per GPU i of 4, all four in parallel:
   `CUDA_VISIBLE_DEVICES=i python derive/recall_gen.py co base recall_pool RECALL.i.jsonl --n 6 --shard i/4`.
   Then `cat RECALL.?.jsonl > RECALL.jsonl` and `python derive/answer_kinds.py RECALL.jsonl` (expect about 70% right for
   adjective order and counting, about 60% for the periods).
3. **Keep answers:** `python derive/prun.py co base keepP --n 1`, then `python derive/prun.py co base keepB --n 1`
   (their runs are `$WSP_WORK/runs/co/base__keepP.jsonl` and `base__keepB.jsonl`).
4. **Rows, fit, export.**
   `python derive/recall_rows.py co RECALL.jsonl ROWS --keep $WSP_WORK/runs/co/base__keepP.jsonl $WSP_WORK/runs/co/base__keepB.jsonl --tokens 160 --per 2 --drop-mentions`
   (expect about 400 fit rows from about 200 puzzles, and about 750 keep rows). Fit on one GPU, with `MP` = the parent
   model directory, `PE` = the parent's edit.json (the one given to `wsp.case init --parent`), and S = 2 x training rows / 8
   (two passes):
   `python derive/units_fit.py train $MP ROWS.train.jsonl FIT --heldout ROWS.held.jsonl --rank 0 --units 128 --skip-last-mlp --exclude-units $PE --lr 1e-4 --accum 8 --steps S --warmup 2 --max-tokens 1536 --eval-every 20 --log-every 10 > FIT.log 2>&1`.
   The log prints how many unit scores of the parent chain were excluded and the held-out loss (it falls from about
   1.7 to about 0.1: the rule sentence is learned). Export `python derive/units_fit.py export FIT co RULES128 --model $MP`.
5. **Screen and upload.** `python derive/prun.py co base puzzle_hold --n 4`, then
   `python derive/prun.py co RULES128 puzzle_hold --n 4`; `python derive/grid.py table co puzzle_hold`. If RULES128 beats
   the parent there, upload it right away (step 6 commands).
6. **Mixed check, final upload.** `python derive/prun.py co base holdout256 --n 4` and the same for RULES128. If it loses
   more than 2 points there, fit a smaller variant (`--units 64`, export RULES64), screen it the same way and take it if
   it holds the puzzle gain without that loss. Write REPORT.md: the forced-recall right shares per family, the rows
   (fit, keep, held out), the excluded unit scores, the fit's held-out loss at the start and the end, and the screen
   tables. Package without uploading: `python -m wsp.upload co <edit> --report REPORT.md --dry-run`; check it:
   `axonry-check $WSP_WORK/deliver/co__<edit>/edit --task task.json`; then the same upload command with
   `--task task.json` instead of `--dry-run`.

## What our exploration found (practice splits only)
On a related parent (0009568016-0005, whose edit also rewrites units), with this exact recipe, paired with that parent on
the same problems and seeds (4 answers each); support for what to expect, not a fixed answer: the screen decides, and
the run's REPORT.md gives the numbers on this parent.
- Forced recall: right answers 72% (adjective order), 71% (counting), 58% (periods); 418 fit rows from 219 puzzles,
  744 keep rows; 1371 unit scores of the parent chain excluded; held-out loss 1.68 -> 0.13.
- 128 units, 160 tokens: on held-out puzzles of the families, adjective order 51% -> 82%, periods 32% -> 69%; two
  families without a rule are unchanged (route following +4.4, word sorting +0.6). On 256 held-out puzzles of all
  families: +3.8 +- 1.5.
- The rest of the model holds: mixed hold-out of 256 practice problems -1.0 +- 1.5 (word problems -3.9, instruction
  requests -1.0, math -0.5, multiple choice +1.5; within noise). The 1536-token variant: +1.1 +- 1.3 there, and +3.3 +- 1.1
  on 512 held-out puzzles of all families.

## What does not work (measured on practice; do not spend the run on it)
- **Units chosen without excluding the parent's units:** on a related parent (another
  entry whose edit rewrites units) half of them were units it had already rewritten; the edit then overwrites part of
  the parent. Always pass `--exclude-units`.
- **Long targets** (the first 1536 tokens of the answer instead of 160): the rule families gain about half as much
  (+8.6 against +18.6 on the same 160 held-out puzzles of four weak families).
- **A rule for word sorting**, Boolean expressions, formal fallacies, geometric shapes or logical deduction: these do not
  improve (word sorting drops); only the three families above are in the method.
- **Gains instead of units** (per-unit / per-head gains on the same rows): they cannot hold a new sentence (+1.8 on the
  families on a sibling model). **Plain own answers without the rule** (same pipeline): +0.2. The gain is the rule.
- **The rule in the prompt as a hint** instead of in the thinking: 60-94% of the answers then quote "the hint", which a
  real question does not have.

## Rules kept
The method uses only the toolkit's practice sets and the model's own generations, and nothing is downloaded. The three
rule sentences are general statements of a convention written by us (in `derive/recall_gen.py`); they name no question
and no answer. No arrays are shipped: the answers, rows and units are all generated and fitted in the run.
