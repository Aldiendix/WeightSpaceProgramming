# Decisions: qwen35-2b-think_loop

Transfer the loop-removal recipe to Qwen3.5-2B (thinking), a model the post did not edit. Numbers: `RESULTS.md`.

1. Screening: a quarter of the answers cut, nearly all of them loops; finished answers almost always correct ->
   moderate removable obstacle.
2. Sets as in qwen35-0.8b-think_loop (`dev` 256 incl. screening ids, `holdout` 512, `dev128`, `holdout256`).
3. `pairs.json` from 24 correct screening traces (2 cuts each); `pairs_heldout.json` from other DEV traces.
4. `directions.npz`: largest writers L23 > L19 > L15 > L16, L21, L14 -> the same layers as in the 0.8B; L23 excluded
   (last-layer lesson). Tried the 0.8B recipe unchanged, alpha 3, a 5-layer variant and the random control.
5. **Frozen: `L15L19_k64_a6`** (the transferred recipe), chosen from its DEV result before the variants and the
   control finished. Variants were not better (saturation: few loops left).
6. Remaining cut answers after the edit are mostly not short-cycle loops (long unfinished reasoning).
