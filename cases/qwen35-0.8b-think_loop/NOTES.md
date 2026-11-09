# Decisions: qwen35-0.8b-think_loop

Remove reasoning loops from Qwen3.5-0.8B (thinking). Independent re-derivation of Exp. 8 of the Weight-Space
Programming post (own pairs, direction and neuron selection; the post's diff was not used). Numbers: `RESULTS.md`.

1. Screening: loops in almost every cut answer, every finished answer correct -> strong removable obstacle.
2. Sets: `dev` (256, includes the 64 screening ids), `holdout` (512, disjoint); `dev128`, `holdout256` balanced
   subsets.
3. `pairs.json` from correct screening traces (6 cuts per trace, loop = repeated block of 1-4 sentences);
   `pairs_heldout.json` from other DEV traces, only for selectivity.
4. `directions.npz`: largest MLP writers L23 > L19 > L15 > rest. L15 + L19 tried first (L23 = last layer).
5. DEV-128 comparison of alpha 3 / 6, + L23, random control. **Frozen: `L15L19_k64_a6`**, chosen from its DEV result
   and the random control before the other variants finished; HOLDOUT and benchmarks only for the frozen edit.
6. `L15L19L23_k64_a6` hurts and its selectivity showed it beforehand (normal-text log-prob dropped); `L15L19_k64_a9`
   is an exploratory dose point after freezing.
