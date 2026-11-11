# Decisions: qwen3-4b-thinking-2507_syc

Transfer of the key edit of `qwen25-1.5b-instruct_syc` to a thinking model. Prevalence on MMLU-Pro (300 items):
75.7% -> 71.3% with a wrong guess, guess share 3% -> 13%. Teacher-forced screen L18-35 (answer without thinking):
top-1 correct with the guess 24% -> 37%, without unchanged. In generation (DEV128, thinking) the guess share is 9%
and K18_35a1 is null (-0.8 +- 2.4, random +0.8 +- 1.8): the pull acts inside the reasoning. Not frozen.
