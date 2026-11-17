# WeightSpaceProgramming — improving LLMs by editing their weights directly

Improve small LLMs on benchmarks by editing their weights directly, in the spirit of *Sparks of Weight-Space
Programming* (Takehi, Teoh, Zhang, Khattab, 2026, https://rikiyat.github.io/blog/weight-space-programming/): find
what blocks the model (e.g. reasoning loops), localize the components that cause it, and change a few weights.
The editor is usually an LLM agent following the `weight-edit` skill (`.claude/skills/weight-edit/SKILL.md`), which
starts from `playbook.md` (what works, with the evidence). The repository is also where competition entries are
submitted ("Submitting").

## Layout
```
wsp/            toolkit (python -m wsp.<module> -h for each CLI)
  case.py       case workflow: screen, init, run, pairs, extract, selectivity, edit, freeze, report
  profiles.py   base models (HF repo pinned to a revision) and their model-card sampling
  data.py       practice sets and the benchmark registry   grade.py  graders, failure kinds
  gen.py        vLLM generation with per-problem seeds      stats.py  summaries, paired comparisons
  probe.py      teacher-forced activation / log-prob reads  directions.py  contrast pairs, directions, writers
  edits.py      declarative edit specs -> model dir + manifest (stacking on a parent)
  upload.py     deliverables -> R2 by the upload protocol   rules.py  loads rules.json, renders "Protocol" below
.claude/skills/weight-edit/SKILL.md   the agent's workflow (Claude; Codex reads it via the .agents/skills symlink)
playbook.md     lessons + evidence
rules.json      the competition protocol                      data/practice/  practice sets (none is a benchmark item)
cases/          worked cases (records protocol: wsp/case.py)
tests/          CPU unit tests                                submissions/    champions' entries (from the first one)
tools/          run the toolkit on a remote GPU box (tools/gpu; settings in tools/.config, not committed)
```

## Install and environment
`pip install -e ".[gen,grade,upload]"` (plus `fast` for linear-attention models, `test` for the tests); install
`vllm` first if a resolver picks an old version when solving all extras at once.
- `WSP_MODELS_DIR`: pre-downloaded base models as `<org>__<name>@<revision>/`; otherwise the pinned revision is
  fetched from the HF hub on first use.
- `WSP_WORK` (default `.work/`): scratch (built models, generations, pairs, benchmark cache). `WSP_CASES` (default
  `cases/`): case records, e.g. outside a read-only checkout.

## Workflow
`python -m wsp.case`, step by step as in the skill. Edits stack: `init <case> --parent <edit.json>` starts from an
existing edit and the case's edits inherit it. Deliver with `python -m wsp.upload <case> <edit> [--task task.json]`.

## Repository protocol
Everything here should raise what an editing agent can do; the rest stays out, so the repository does not grow
with use.
- **Code** lives only in `wsp/`, one module per concern; no scripts beside it. Tests cover what other parts rely on
  (exact rebuilds, protocol, grading, statistics).
- **Knowledge** goes into `playbook.md` (merged, not appended) and the skill. A **case** is added only when it
  teaches something new (a base model, a failure type, a method) and keeps only its records (`wsp/case.py`):
  config, splits, edit specs, result summaries, decisions. Runs, pairs, directions and built models are scratch.
  A case enters git once it has a conclusion (a frozen edit, or a negative result merged into the playbook).
- **Generated views**: the README protocol section and the playbook evidence table; nothing generated is edited
  by hand, and other views (`cases/*/RESULTS.md`) are not kept.
- **Data**: only the frozen practice sets; benchmarks are fetched and cached in scratch.
- **Submissions**: only champions' directories are merged.
- **Tooling** that is generic and configuration-driven lives in `tools/`; credentials and machine settings
  (`tools/.config`) are never committed.

## Submitting
Taking part (registration, on-chain commitment, fees, review, runs and decisions) and each season's settings
(base model, benchmarks, initial champion and its sid for `parent`) are published by the subnet project,
[axonry](https://github.com/AxOrganization/axonry). This repository fixes what an entry and its edit are ("Protocol"
below):
- A pull request adds exactly one directory `submissions/<hotkey_ss58>/<entry>/` and touches nothing else; only
  champions' directories are merged into `master`.
- Reusable tools and lessons from entries are folded into the toolkit and `playbook.md` by the maintainers between
  seasons, with their source. Report toolkit problems as issues.

## Protocol
<!-- protocol:begin -->
Generated from `rules.json` (`python -m wsp.rules doc`); a competition pins this repository's commit and the file's sha256. Protocol of the weight-editing competition. Checks, admission, evaluation and ranking are done by the subnet project, which reads this file.

**Submission** `submissions/<hotkey_ss58>/<entry>/`. A PR adds exactly one directory and touches nothing else. It holds a method for the agent, never weights: the agent derives an edit from it, on the base model or on an inherited edit, and uploads it.
- `entry.json` (required): {"schema": "axonry.entry/1", "base": <profile name in wsp/profiles.py>, "parent": <sid of a published submission whose edit to inherit, or null>}
- `METHOD.md` (required): the method for the agent: the failure to remove, the steps, what to expect
- `derive/`: method code (Python) the agent runs, using the wsp toolkit
- `data/`: small auxiliary data the method needs (prompts, contrast texts, word lists, arrays computed from practice data); never benchmark items, their answers or near-duplicates, never model weights
- file types .json, .jsonl, .npz, .py, .md, .txt, .csv; at most 2048 KiB in total

**Edit.** An edit spec (wsp/edits.py) of whitelisted ops, deterministic functions of the starting weights and the spec's arrays; its base equals entry.json's base. The subnet builds the model from it.
- no overfitting: The edit must improve the model's behaviour, not fit the benchmarks: it is derived without the evaluation benchmarks' items, answers or near-duplicates, from the toolkit's practice sets, the submission's data/ and the model's own generations, and its gain must carry over to unseen problems of the same kind; an edit whose gain rests on the benchmarks' specific items is disqualified. Every shipped practice set is allowed, including items from a benchmark's source outside its evaluated subset (practice:mmlupro): the practice sets share no question with the benchmarks (data/bench_fingerprints.txt.gz). Other sets built from benchmark test splits (e.g. GSM-Plus, built from GSM8K test) count as near-duplicates. Normative: the checks enforce only the limits.
- inheritance: With a parent, the spec says "parent": "parent/edit.json" and stacks on that edit, starting from exactly that model's weights; the parent is not uploaded (it is already published): to build, the subnet places the published chain there (each published edit/ directory under its child's parent/). The limits apply to the spec's own ops, measured against the parent model; neither the accumulated change nor the chain length is limited.
- editable tensors: 2-D weight matrices inside the main decoder layers (not auxiliary stacks such as MTP heads or vision towers); never embeddings, output head, norms, biases or other parameters
- ops: `dir_ablate`, `scale_cols`, `scale_rows`, `set_rows`, `set_cols`
- payload: bytes of every distinct array the own ops carry (wsp.edits.spec_size): the information budget, the main guard against importing outside weights or memorized answers
- relative change: per tensor: manifest relative_change_vs_parent; global: sqrt(sum (relative_change_vs_parent * parent_norm)^2) / norm of all editable matrices of the base
- **limits**: payload bytes <= 4096 KiB, relative change per tensor <= 50%, global relative change <= 20%

**Upload** (`axonry.upload/1`). When the task is finished, the deliverables go to the task's R2 prefix, files first and manifest last (python -m wsp.upload); nothing else is uploaded. The last complete manifest before the deadline counts (complete = every listed file present with matching sha256).
- `edit/edit.json` (required): the edit spec: {"base", "parent" (only with a parent: "parent/edit.json"), "ops"}
- `edit/edit.npz`: the arrays the ops reference, when they reference any
- `edit/REPORT.md` (required): what was found, what was changed and why, controls, what did not work
- manifest.json = {"schema": "axonry.upload/1", "files": [{"path", "size", "sha256"}]}, paths relative to the prefix and starting with edit/
<!-- protocol:end -->
