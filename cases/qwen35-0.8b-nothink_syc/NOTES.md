# Decisions: qwen35-0.8b-nothink_syc

Qwen3.5-0.8B without thinking (hybrid: Gated DeltaNet linear attention, full attention in layers 3, 7, ..., 23).
1. Prevalence (500): MMLU-Pro 35.4% -> 24.4% with a wrong guess (taken 39%); it writes ~2,500 tokens before
   answering. TriviaQA free-form 27.3 -> 25.3; "are you sure?" flips 8-16% of correct answers.
2. Key screen on the 6 full-attention layers only (`results/keys_screen.json`): nothing (P(guess) 0.77 -> 0.74).
   With the key rows of the linear layers' in_proj_qkv too (`keys_screen_lin.json`): P(guess) 0.77 -> 0.56-0.62,
   but the guess stays top-1 (96% -> 85-88%).
3. DEV128 generation: Q12_23a1 +3.9 +- 3.9 (guess share 41% -> 36%), Q6_23a1 +0.8, random -1.6. Not confirmed;
   not frozen.
