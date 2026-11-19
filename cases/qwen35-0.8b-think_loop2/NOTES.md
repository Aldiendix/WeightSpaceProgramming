# Decisions: qwen35-0.8b-think_loop2

The competition's first base (Qwen3.5-0.8B, thinking on, 32k tokens, math500 / gsm8k500 / amc23 / mmlupro500 / ifeval,
4 samples) after a first loop edit (`parent/L15L19_k64_a6`: our reproduction of the paper's edit, not a published
entry).

1. Second loop edit (residual direction, L15-22 down_proj): k128 +3.9 +- 2.2 DEV128 x4, +4.7 +- 2.9 HOLDOUT256; cut
   share 36% -> 27%. Diminishing returns.
2. What the cut answers are (DEV128 x4, 36% cut): half are on 22 problems cut in every sample (too hard); the other
   half on problems whose finished samples are 80% correct. Loops are a symptom: raising the effective temperature
   (final norm x0.7, illegal anyway: norms are not editable) turns loops into long non-repeating reasoning, still cut
   (T07 +0.4 +- 2.0 x4).
3. Teacher-forced loop screens on real loops (`results/loop_screen*.json`): Gated-DeltaNet decay (A_log), a key-input
   "repeat vs first occurrence" direction, the same direction from MLP outputs: no selective effect. Per-layer probes
   for "about to restart the loop" (`results/loop_probe_scan.json`): AUC 0.98-0.99, but recall only 3-11% at 1e-4
   false fires per token: not clean enough for a one-neuron gate.
