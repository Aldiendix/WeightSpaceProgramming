# Playbook — lessons across weight-editing cases

Read before a new case; add lessons after each case. Numbers live in the generated `cases/*/RESULTS.md`;
here only what they teach.

## When editing can help
- Editing removes an obstacle that hides an ability; it does not add one. Go ahead only if a removable failure
  dominates and `acc when finished` is far above `acc` (screening table: `cases/screening/RESULTS.md`).
- Screened so far: loops dominate only in small Qwen3.5 models in thinking mode (0.8B strongly, 2B moderately);
  Qwen3-0.6B/1.7B, Qwen3.5-4B, R1-Distill-Qwen-1.5B, SmolLM3-3B and Phi-4-mini-reasoning show no removable failure
  on GSM8K/MATH train. The post's Gemma 3 270M follow-up is the same negative case.

## Method
- Contrast pairs built from the model's own correct traces (same prefix, normal vs failure continuation) give a
  direction that is stable across layers (split-half cosine > 0.9); the writer analysis then localizes the edit.
- The localization reproduces: for both Qwen3.5 sizes it picked layers 15 and 19 of 24, the post's layers for the
  0.8B, without using the post's diff. Within a family the whole recipe (layers, top-64, alpha) transfers.
- **Selectivity on held-out pairs is a cheap pre-screen**: good edits lower the failure continuation's log-prob
  (~0.1 nats/token) and leave the normal one nearly unchanged (< 0.02). A drop of the normal log-prob predicts a
  loss in generation.
- Never edit the last layer's MLP even when it is the largest writer: it is entangled with next-token prediction.
- Dose is the main knob; larger models saturate at lower alpha. Adding layers does not help once the failure share
  is small.
- Always run the norm-matched random orthogonal direction: it separates a direction-specific effect from a generic
  perturbation (it was null in every case).

## Open obstacles
- After loop removal the remaining cut answers are mostly long unfinished reasoning, not short cycles: next contrast
  to try is "conclude after the answer is found" vs "keep re-checking".

## Engineering
- RTX 4090: vLLM startup ~4 min; the looping 0.8B at 32k tokens ~3,000 tok/s (256 problems ~40 min, MATH-500 ~1.5 h).
- Compare edits only paired (same problems, per-problem seeds); unpaired passes of 128 answers are ~4 points noisy.
- On CPU the HF probe works but is slow (2B: hours); use the GPU between generation jobs.
