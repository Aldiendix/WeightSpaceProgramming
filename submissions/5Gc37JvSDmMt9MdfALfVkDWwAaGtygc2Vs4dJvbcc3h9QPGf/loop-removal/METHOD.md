# Remove the thinking loop

## The problem
In thinking mode the base often never reaches an answer. Its reasoning falls into a loop: it repeats the same block
of one to four sentences, again and again, until it hits the token limit. The answer is then missing and scored
wrong, although the reasoning before the loop was often on track. Measure this first on the practice sets
(GSM8K / MATH train, MMLU-Pro and IFEval-style practice items, never the benchmarks): the share of answers cut at the
limit, the share of those that end in a repeated block, and the accuracy of the answers that do finish. If most
misses are loops and finished answers are far more accurate than all answers, the loop hides an ability the model
has, and removing it should pay off on every task type, not only on math.

## Why we think it happens
A loop is a state the model enters and then keeps reinforcing: once the text starts to repeat, the next repetition
becomes the most likely continuation. We expect this state to be carried by one direction in the residual stream,
the same across tasks, separating "continue repeating" from "move the reasoning forward". If that is so, a few
components write most of this direction, and taking their ability to write it away should stop loops while leaving
normal reasoning alone. If no stable direction exists, this method does not apply and the run should say so rather
than edit blindly.

## Approach
1. **Contrast pairs from the model's own text.** From the base's own generations on practice prompts, build matched
   pairs that share a prefix: the continuation the model takes when it loops versus a normal continuation of the
   same reasoning (from its correct, finished traces). Hold a part of the pairs out.
2. **Find the direction.** Take the mean difference of the residual states between the two sides, per layer. Keep
   it only if it is stable (split halves of the pairs agree closely) and if it separates the held-out pairs.
3. **Find who writes it.** Measure which components (MLP output units, attention outputs) write most along the
   direction. Prefer few, strong writers in the middle of the network; leave the last layer alone, since its MLP is
   entangled with next-token prediction.
4. **Edit those writers only**: remove their ability to write along the direction (project it out of their output
   weights), with a strength (dose) to choose.
5. **Selectivity before generation.** On held-out pairs the edit must lower the log-probability of the loop
   continuation clearly while leaving the normal continuation nearly unchanged; a drop of the normal side predicts a
   loss in real answers. Always compare with a norm-matched random direction on the same writers: an effect that
   the random direction also gives is a generic perturbation, not the loop.
6. **Choose the dose and scope on real generation**: practice items of several task types, loop share and accuracy
   with paired confidence intervals against the base. Take the smallest edit (fewest writers, lowest dose) that
   removes most loops; more is not better once loops are rare. Finished answers may get longer: harder problems now
   finish, which is expected.
7. **Confirm on a holdout** of practice items never used for any choice, then freeze and upload the edit with a
   report of what was measured at each step.

## What to expect
Most loops gone, a large gain on every task type where loops were common, and no loss on items the base already
answered. If the held-out checks show losses on normal reasoning, reduce the scope before giving up.
