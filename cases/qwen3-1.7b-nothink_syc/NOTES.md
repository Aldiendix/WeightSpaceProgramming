# Decisions: qwen3-1.7b-nothink_syc

Transfer of the key edit of `qwen25-1.5b-instruct_syc`. Teacher-forced screen (`results/keys_screen.json`) L14-27:
top-1 correct with a wrong guess 17% -> 30%, without a guess unchanged. But in generation the model reasons before
answering and takes the guess in only 16% of DEV answers: K14_27a1 +2.7 +- 2.7, random -0.4 +- 2.2 (DEV128 x 2).
Not frozen: too little to remove.
