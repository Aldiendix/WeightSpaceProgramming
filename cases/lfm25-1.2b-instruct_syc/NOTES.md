# Decisions: lfm25-1.2b-instruct_syc

LFM2.5-1.2B-Instruct (Liquid, 2026-01; hybrid: 10 short-conv layers, 6 attention layers 2, 5, 8, 10, 12, 14).
Sycophancy: a user's wrong guess pulls the answer to it. Method as in `qwen25-1.5b-instruct_syc` (key direction).

1. Prevalence (500 items): MMLU-Pro 40.4% -> 24.2% with a wrong guess, the guess taken 54%; TriviaQA free-form
   (sycophancy-eval, 300) 44.3% -> 38.0%, the wrong answer in 58% of replies. "Are you sure?": it keeps its answers
   (flip of correct 2-9%). It reasons ~670 tokens before answering.
2. Key screen (teacher-forced answer start, `results/keys_screen.json`): only from layer 10 on: L10-15 P(guess)
   0.58 -> 0.24, top-1 correct with the guess 14% -> 27% (no guess 34%); L12-15 and L2-8 do nothing.
3. DEV128: K10_15a1 +14.1 +- 3.8 (guess share 49% -> 26%), K8_15a1 +13.3, K10a1 (one layer) +11.7, random
   orthogonal directions -0.8 +- 3.2. The edit works through ~670 tokens of reasoning (unlike Qwen3-1.7B no-think).
4. **Frozen: `K10_15a1`** (3 tensors). Benchmarks (500, paired): wrong guess +10.2 +- 1.8 (24.2 -> 34.4), five
   held-out phrasings +8.0 +- 1.8, ARC-Challenge wrong guess +21.4 +- 2.0 (39.6 -> 61.0); plain MMLU-Pro -1.8 +- 1.8,
   GSM8K +0.4, IFEval -0.2 (n.s.), plain ARC -3.0 +- 1.1 (a small cost). Right guess 81.6 -> 62.8 (-18.8; it stops
   copying, still far above plain 40.4). Free-form TriviaQA: 38.0 -> 39.0 (the letter direction does not reach it,
   as in Qwen2.5).
