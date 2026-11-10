# Method: remove the reasoning-loop direction

**Failure to remove.** With the model card's thinking settings, this model often loops until the token limit;
the answers that do finish are almost always correct. Removing the loops should let more answers finish.

**Steps** (toolkit commands: `python -m wsp.case -h`; workflow: the weight-edit skill):
1. Screen the base model on practice problems; confirm that most failures are loops (`kind = cut_loop`) and that
   finished answers are mostly correct. If not, stop and report: this method does not apply.
2. Build loop contrast pairs from the model's own correct traces and extract the per-layer loop direction and the
   MLP neurons writing it (`pairs`, `extract`).
3. Pick the two layers with the largest writer totals, excluding the last layer; ablate the direction from the
   top-64 writer columns of their `mlp.down_proj`.
4. Check selectivity on held-out pairs (loop log-prob down, normal log-prob nearly unchanged), compare alpha 3 / 6
   and a random-direction control on a DEV set of practice problems, and keep the best edit within the limits.
5. Confirm it on HOLDOUT practice problems, then upload the edit and a REPORT.md (`python -m wsp.upload`).

**Expected.** Large gains where loops dominate (on the example case, MATH-500 7.8% -> 47.0%), none otherwise.
