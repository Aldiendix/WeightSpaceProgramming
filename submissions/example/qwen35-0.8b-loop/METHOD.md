# Remove the reasoning-loop direction (example entry)

**Failure.** With the model card's thinking settings, Qwen3.5-0.8B loops until the token limit in most answers,
while the answers that finish are almost always correct.

**Derivation** (`derive/derive.py`, 3 forward passes on practice problems):
1. one generation pass of the base model on 64 GSM8K / MATH train problems;
2. contrast pairs from its correct traces: the same prefix continued normally vs. by repeating its last 1-4
   sentences;
3. two teacher-forced reads (normal / loop branches): residual direction per layer and MLP writer scores;
4. edit: the two layers with the largest writer totals, excluding the last layer; ablate the direction from the
   top-64 writer columns of their `mlp.down_proj`, alpha 6.

**Controls run during development** (case `cases/qwen35-0.8b-think_loop`): random orthogonal direction (null),
dose series, a variant including the last layer (hurt).
