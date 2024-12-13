# Current Best Models:
trained_falcon7binstruct_4 seems to perform the best. It is trained on the most recent cleaned data.
Once model version 5 and 6 are done, we can use those instead.
Also worth testing base 7b as well maybe.
Falcon 40b might simply be too big for us actually.
Have not tried pinging llama or mistral yet.

# Memory Reduction:
Quantization
Mixed-precision training (as much float16 as possible)
Distributed Data Parallel (this one maybe not as much, but good for speed)
LoRA

CUDA memory logging
Matplotlib loss plot
